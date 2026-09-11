#include "pinned_gmp.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Isolate registration failures and startup cost without allocating GPU data.
int main(int argc,char**argv){
    if(argc!=2){fprintf(stderr,"Usage: reserve_bench GiB\n");return 2;}
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess)return 2;
    count=std::min(count,8);
    for(int d=0;d<count;d++)if(cudaSetDevice(d)!=cudaSuccess||cudaFree(nullptr)!=cudaSuccess)return 2;
    cudaSetDevice(0);host_topology::automatic=true;
    size_t bytes=strtoull(argv[1],nullptr,10)<<30;
    if(!bytes)return 2;
    pinned_gmp::cache_limit=bytes;
    auto start=std::chrono::steady_clock::now();
    pinned_gmp::reserve(bytes);
    bool success=pinned_gmp::arena_reserved>=bytes;
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    printf("gpus=%d bytes=%zu success=%d arenas=%zu seconds=%.6f\n",count,bytes,success,pinned_gmp::arenas.size(),seconds);fflush(stdout);
    if(success){
        for(auto&entry:pinned_gmp::arenas){
            auto*h=entry.second->base;size_t end=entry.second->bytes;
            for(size_t position:std::initializer_list<size_t>{4096,end-4}){
                auto*word=(uint32_t*)((char*)h+position);*word=(uint32_t)(position/4)^0xa5197u;
                cudaPointerAttributes attr{};
                if(cudaPointerGetAttributes(&attr,word)!=cudaSuccess||attr.type!=cudaMemoryTypeHost)return 3;
                for(int d=0;d<count;d++){
                    cudaSetDevice(d);uint32_t*device=nullptr,got=0;
                    if(cudaMalloc(&device,4)!=cudaSuccess||cudaMemcpy(device,word,4,cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(&got,device,4,cudaMemcpyDeviceToHost)!=cudaSuccess||got!=*word)return 3;
                    cudaFree(device);
                }
            }
        }
        puts("PASS: registered range endpoints transfer correctly on every GPU");
        size_t n=std::min<size_t>(100ull<<30,bytes/8);std::vector<void*>allocations;
        for(int i=0;i<5;i++){
            void*p=pinned_gmp::allocate(n);allocations.push_back(p);
            auto*owner=((pinned_gmp::Header*)p-1)->owner;
            if(!owner||(char*)p+n>(char*)owner->base+owner->bytes)return 3;
        }
        for(int i:{2,0,4,1,3})pinned_gmp::release(allocations[i],n);
        puts("PASS: pool allocations stay within one registration range");
    }
    pinned_gmp::clear();return success?0:1;
}
