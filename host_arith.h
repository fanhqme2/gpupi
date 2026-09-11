#pragma once
#include <gmpxx.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include "host_topology.h"

namespace host_arith {
inline size_t threshold(){const char*s=getenv("GPUPI_HOST_MIN");return s?strtoull(s,nullptr,10):(1ull<<20);}
template<class Fn>void parallel(size_t n,Fn fn){
    const char*s=getenv("GPUPI_HOST_THREADS");size_t count=s?strtoull(s,nullptr,10):std::max<size_t>(1,std::min<size_t>(32,std::thread::hardware_concurrency()/2));
    count=n<threshold()?1:std::max<size_t>(1,std::min<size_t>(count,32));
    count=std::min(count,std::max<size_t>(1,n));
    size_t base=host_topology::enabled()?host_topology::ticket.fetch_add(count):0;
    std::vector<std::thread>threads;
    for(size_t t=1;t<count;t++)threads.emplace_back([=,&fn]{host_topology::run(base+t,[&]{fn(n*t/count,n*(t+1)/count,t);});});
    host_topology::run(base,[&]{fn(0,n/count,0);});for(auto&t:threads)t.join();
}
inline void copy(void*dst,const void*src,size_t bytes){parallel((bytes+7)/8,[&](size_t a,size_t b,size_t){a*=8;b=std::min(b*8,bytes);if(a<b)std::memcpy((char*)dst+a,(const char*)src+a,b-a);});}
// Add a positive limb slice into existing storage, with exact chunk carries.
inline void add_at(mp_limb_t*out,size_t n,const mp_limb_t*src,size_t sn){
    if(!sn)return;
    struct Chunk {size_t l=0,r=0;mp_limb_t carry=0;};Chunk chunks[32];
    parallel(n,[&](size_t l,size_t r,size_t t){
        size_t m=l<sn?std::min(r,sn)-l:0;
        mp_limb_t carry=m?mpn_add(out+l,out+l,r-l,src+l,m):0;
        chunks[t]={l,r,carry};
    });
    mp_limb_t incoming=0;
    for(size_t t=0;t<32&&chunks[t].r;t++){
        mp_limb_t propagated=incoming;
        for(size_t j=chunks[t].l;j<chunks[t].r&&propagated;j++){++out[j];propagated=out[j]==0;}
        incoming=chunks[t].carry|propagated;
    }
    if(incoming)throw std::overflow_error("host add exceeds output capacity");
}
template<class Z>void add(Z&a,const Z&b,bool subtract=false){
    if(&a==&b){if(subtract)a=0;else a<<=1;return;}
    size_t an=mpz_size(a.get_mpz_t()),bn=mpz_size(b.get_mpz_t()),n=std::max(an,bn);
    if(n<threshold()||!an||!bn){if(subtract)a-=b;else a+=b;return;}
    int as=mpz_sgn(a.get_mpz_t()),bs=mpz_sgn(b.get_mpz_t())*(subtract?-1:1);
    bool difference=as!=bs,reverse=difference&&mpz_cmpabs(a.get_mpz_t(),b.get_mpz_t())<0;
    int sign=reverse?bs:as;
    mp_limb_t*out=mpz_limbs_modify(a.get_mpz_t(),n+1);const mp_limb_t*bp=mpz_limbs_read(b.get_mpz_t());
    if(n>an)parallel(n-an,[&](size_t l,size_t r,size_t){std::memset(out+an+l,0,(r-l)*8);});out[n]=0;
    struct Chunk {size_t begin=0,end=0;mp_limb_t flag=0;};Chunk chunks[32];size_t count=0;
    parallel(n,[&](size_t l,size_t r,size_t t){
        size_t m=l<bn?std::min(r,bn)-l:0;mp_limb_t flag=0;
        if(reverse)flag=mpn_sub_n(out+l,bp+l,out+l,r-l);
        else if(m)flag=difference?mpn_sub(out+l,out+l,r-l,bp+l,m):mpn_add(out+l,out+l,r-l,bp+l,m);
        chunks[t]={l,r,flag};
    });
    while(count<32&&chunks[count].end)++count;
    mp_limb_t incoming=0;
    for(size_t t=0;t<count;t++){
        mp_limb_t propagated=incoming;
        for(size_t j=chunks[t].begin;j<chunks[t].end&&propagated;j++){
            mp_limb_t old=out[j];out[j]=difference?old-1:old+1;propagated=difference?(old==0):(out[j]==0);
        }incoming=chunks[t].flag|propagated;
    }
    if(!difference)out[n]=incoming;
    mpz_limbs_finish(a.get_mpz_t(),(mp_size_t)(n+1)*sign);
}
template<class Z>void shift(Z&a,int64_t bits){
    if(!bits||a==0)return;
    size_t an=mpz_size(a.get_mpz_t());
    if(an<threshold()||a<0){if(bits>=0)a<<=bits;else a>>=-bits;return;}
    uint64_t amount=bits>=0?bits:-bits,words=amount/64;unsigned part=amount%64;
    if(bits<0&&words>=an){a=0;return;}
    size_t n=bits>=0?an+words+(part!=0):an-words;Z result;
    auto*out=mpz_limbs_write(result.get_mpz_t(),n);const auto*src=mpz_limbs_read(a.get_mpz_t());
    parallel(n,[&](size_t l,size_t r,size_t){
        if(bits<0){for(size_t j=l;j<r;j++){size_t i=j+words;out[j]=part?(src[i]>>part)|((i+1<an?src[i+1]:0)<<(64-part)):src[i];}}
        else{for(size_t j=l;j<r;j++){if(j<words){out[j]=0;continue;}size_t i=j-words;mp_limb_t lo=i<an?src[i]:0,hi=i&&i-1<an?src[i-1]:0;out[j]=part?(lo<<part)|(hi>>(64-part)):lo;}}
    });mpz_limbs_finish(result.get_mpz_t(),n);a=std::move(result);
}
}
