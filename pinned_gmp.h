#pragma once
#include <cuda_runtime.h>
#include <gmp.h>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <set>
#include <memory>
#include "host_topology.h"

// Optional GMP allocation policy. Large buffers use portable pinned memory
// and a bounded reuse cache so GPU transfers do not fault pageable buffers.
// Install before constructing any GMP objects; clear after all are destroyed.
namespace pinned_gmp {
inline bool option(const char*name,bool fallback){const char*s=getenv(name);return s?atoi(s)!=0:fallback;}
struct Arena;
struct alignas(16) Header {size_t bytes; Arena*owner; bool pinned,registered,mapped; char padding[45];};
static_assert(sizeof(Header)==64);
inline std::mutex mutex;
inline std::multimap<size_t,Header*> cache;
inline size_t cached_bytes=0, cache_limit=0;
inline std::atomic<uint64_t> hits{0},misses{0},fallbacks{0},allocation_ns{0},allocation_bytes{0};
// Bound individual reservations; CUDA copies must stay within one registered
// range. Separate arenas also prevent coalescing across registration boundaries.
inline constexpr size_t reservation_chunk=384ull<<30;
inline void free_pinned(Header*h){
    if(h->mapped){size_t n=h->bytes;if(h->registered)cudaHostUnregister(h);munmap(h,n);}else cudaFreeHost(h);
}
inline Header* raw_allocate(size_t total){
    Header*h=nullptr;
    auto start=std::chrono::steady_clock::now();++misses;allocation_bytes+=total;
    bool registered=option("GPUPI_MGPU_REGISTER",true);
    bool pageable=getenv("GPUPI_MGPU_PAGEABLE")&&atoi(getenv("GPUPI_MGPU_PAGEABLE"));
    if(pageable)registered=false;
    bool mapped=registered||pageable;
    if(mapped){
        total=(total+4095)&~size_t(4095);
        h=(Header*)mmap(nullptr,total,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(h==MAP_FAILED)h=nullptr;
        if(h){
            madvise(h,total,MADV_HUGEPAGE);
            size_t pages=total/4096,count=std::min<size_t>(32,std::max<size_t>(1,total/(32ull<<20)));
            size_t base=host_topology::enabled()?host_topology::ticket.fetch_add(count):0;
            std::vector<std::thread>threads;
            for(size_t t=1;t<count;t++)threads.emplace_back([=]{host_topology::run(base+t,[&]{size_t a=pages*t/count,b=pages*(t+1)/count;std::memset((char*)h+a*4096,0,(b-a)*4096);});});
            host_topology::run(base,[&]{std::memset(h,0,(pages/count)*4096);});for(auto&t:threads)t.join();
            if(registered){
                cudaError_t status=cudaHostRegister(h,total,cudaHostRegisterPortable);
                if(status!=cudaSuccess){
                    if(getenv("GPUPI_MGPU_PROFILE"))fprintf(stderr,"host_register_failed bytes=%zu error=%s\n",total,cudaGetErrorString(status));
                    cudaGetLastError();munmap(h,total);h=nullptr;
                }
            }
        }
    }else if(cudaHostAlloc((void**)&h,total,cudaHostAllocPortable)!=cudaSuccess){cudaGetLastError();h=nullptr;}
    if(!h)++fallbacks;
    allocation_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    if(h){h->bytes=total;h->pinned=true;h->registered=registered;h->mapped=mapped;h->owner=nullptr;}
    return h;
}
struct Arena {Header*base;size_t bytes,live=0;bool registered,mapped;};
inline std::map<size_t,std::set<Header*>> blocks;
inline std::map<Header*,size_t> ranges;
inline std::map<Header*,std::unique_ptr<Arena>> arenas;
inline size_t arena_free_bytes=0,next_arena=64ull<<20,arena_reserved=0,arena_peak=0;
inline size_t arena_live_bytes=0,arena_live_peak=0;
inline void insert_block(Header*h,size_t n,Arena*owner){
    h->bytes=n;h->owner=owner;h->pinned=true;h->registered=owner->registered;h->mapped=owner->mapped;
    blocks[n].insert(h);ranges[h]=n;arena_free_bytes+=n;
}
inline void erase_block(Header*h,size_t n){
    auto it=blocks.find(n);it->second.erase(h);if(it->second.empty())blocks.erase(it);ranges.erase(h);arena_free_bytes-=n;
}
inline bool exact_blocks(){return option("GPUPI_MGPU_EXACT_BLOCKS",true);}
inline Header* split_block(Header*h,size_t have,size_t want,Arena*owner){
    if(exact_blocks()){if(have>want)insert_block((Header*)((char*)h+want),have-want,owner);}
    else while(have>want){have/=2;insert_block((Header*)((char*)h+have),have,owner);}
    h->bytes=want;h->owner=owner;h->pinned=true;h->registered=owner->registered;h->mapped=owner->mapped;++owner->live;
    arena_live_bytes+=want;arena_live_peak=std::max(arena_live_peak,arena_live_bytes);return h;
}
inline Header* arena_allocate(size_t total){
    size_t want=1;if(exact_blocks())want=(total+65535)&~size_t(65535);else while(want<total)want*=2;
    size_t capacity;
    {
        std::lock_guard<std::mutex>lock(mutex);auto it=blocks.lower_bound(want);
        if(it!=blocks.end()){
            size_t have=it->first;Header*h=*it->second.begin();Arena*owner=h->owner;
            erase_block(h,have);++hits;return split_block(h,have,want,owner);
        }
        size_t limit=std::max<size_t>(64ull<<20,std::min<size_t>(32ull<<30,cache_limit/4));
        capacity=std::max(want,std::min(next_arena,limit));next_arena=std::min(limit,next_arena*2);
    }
    Header*h=raw_allocate(capacity);if(!h)return nullptr;
    auto region=std::make_unique<Arena>();region->base=h;region->bytes=capacity;region->registered=h->registered;region->mapped=h->mapped;
    std::lock_guard<std::mutex>lock(mutex);Arena*owner=region.get();arenas.emplace(h,std::move(region));
    arena_reserved+=capacity;arena_peak=std::max(arena_peak,arena_reserved);
    return split_block(h,capacity,want,owner);
}
// Register an initial arena while GPUs are idle. This avoids changing their
// host mappings in the middle of independent kernels on several devices.
inline void reserve(size_t bytes){
    while(bytes){
        // The optional buddy allocator requires power-of-two arena sizes.
        size_t chunk=std::min<size_t>(bytes,exact_blocks()?reservation_chunk:256ull<<30),capacity=1;
        if(exact_blocks())capacity=(chunk+65535)&~size_t(65535);else while(capacity<chunk)capacity*=2;
        Header*h=raw_allocate(capacity);if(!h)return;
        auto region=std::make_unique<Arena>();region->base=h;region->bytes=capacity;region->registered=h->registered;region->mapped=h->mapped;
        std::lock_guard<std::mutex>lock(mutex);Arena*owner=region.get();arenas.emplace(h,std::move(region));
        arena_reserved+=capacity;arena_peak=std::max(arena_peak,arena_reserved);insert_block(h,capacity,owner);
        bytes-=chunk;
    }
}
inline void free_arena(Arena*region){
    if(region->mapped){if(region->registered)cudaHostUnregister(region->base);munmap(region->base,region->bytes);}else cudaFreeHost(region->base);
}
inline void arena_release(Header*h){
    std::unique_ptr<Arena>evicted;
    {
        std::lock_guard<std::mutex>lock(mutex);Arena*owner=h->owner;size_t n=h->bytes;--owner->live;arena_live_bytes-=n;
        if(exact_blocks()){
            auto it=ranges.lower_bound(h);
            if(it!=ranges.begin()){
                auto previous=std::prev(it);Header*p=previous->first;size_t bytes=previous->second;
                if(p->owner==owner&&(char*)p+bytes==(char*)h){erase_block(p,bytes);h=p;n+=bytes;}
            }
            it=ranges.lower_bound(h);
            if(it!=ranges.end()){
                Header*p=it->first;size_t bytes=it->second;
                if(p->owner==owner&&(char*)h+n==(char*)p){erase_block(p,bytes);n+=bytes;}
            }
        }else while(n<owner->bytes){
            size_t offset=(char*)h-(char*)owner->base;
            Header*buddy=(Header*)((char*)owner->base+(offset^n));auto it=blocks.find(n);
            if(it==blocks.end()||!it->second.count(buddy))break;
            erase_block(buddy,n);if(buddy<h)h=buddy;n*=2;
        }
        if(!owner->live&&arena_free_bytes+n>cache_limit){
            auto it=arenas.find(owner->base);evicted=std::move(it->second);arenas.erase(it);arena_reserved-=owner->bytes;
        }else insert_block(h,n,owner);
    }
    if(evicted)free_arena(evicted.get());
}
inline void* allocate(size_t bytes){
    size_t total=bytes+sizeof(Header);Header*h=nullptr;
    if(bytes>=1048576){
        if(option("GPUPI_MGPU_ARENA",true))h=arena_allocate(total);
        else{
            {std::lock_guard<std::mutex> lock(mutex);auto it=cache.lower_bound(total);
                if(it!=cache.end()&&it->first<=total*2){h=it->second;cached_bytes-=it->first;cache.erase(it);++hits;}}
            if(!h)h=raw_allocate(total);
        }
    }
    if(!h){h=(Header*)std::malloc(total);if(!h)std::abort();h->bytes=total;h->pinned=false;h->owner=nullptr;}
    return h+1;
}
inline void release(void*p,size_t){
    if(!p)return;Header*h=(Header*)p-1;
    if(!h->pinned){std::free(h);return;}
    if(h->owner){arena_release(h);return;}
    {std::lock_guard<std::mutex> lock(mutex);
        if(h->bytes<=cache_limit-cached_bytes){cache.emplace(h->bytes,h);cached_bytes+=h->bytes;return;}}
    free_pinned(h);
}
inline void* resize(void*p,size_t old,size_t bytes){
    if(p&&bytes<=((Header*)p-1)->bytes-sizeof(Header))return p;
    void*q=allocate(bytes);if(p)std::memcpy(q,p,std::min(old,bytes));release(p,old);return q;
}
inline void install(size_t limit){cache_limit=limit;mp_set_memory_functions(allocate,resize,release);}
struct Guard {~Guard();};
inline void clear(){std::lock_guard<std::mutex>lock(mutex);for(auto&p:cache)free_pinned(p.second);cache.clear();cached_bytes=0;for(auto&p:arenas)free_arena(p.second.get());arenas.clear();blocks.clear();ranges.clear();arena_free_bytes=arena_reserved=0;}
inline Guard::~Guard(){if(getenv("GPUPI_MGPU_PROFILE"))fprintf(stderr,"pinned hits=%llu misses=%llu fallback=%llu allocated_GB=%.3f allocation_s=%.6f arena_peak_GB=%.3f arena_live_peak_GB=%.3f\n",(unsigned long long)hits.load(),(unsigned long long)misses.load(),(unsigned long long)fallbacks.load(),allocation_bytes.load()/1e9,allocation_ns.load()/1e9,arena_peak/1e9,arena_live_peak/1e9);clear();}
}
