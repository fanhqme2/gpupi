// Full-output analytic multiplication check, including maximal coefficients,
// long carry chains and 64-bit shifts. k=29 fits a 32 GiB GPU;
// k=30/31 should run on the 96 GiB server to avoid host-memory spill.
#include "batch_arith.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);} }while(0)
__global__ void check_product(const uint32_t *c, uint32_t a, uint32_t b, unsigned *bad) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<(size_t)a+b;i+=(size_t)gridDim.x*blockDim.x){
        uint32_t want=i==0?1:i<a?0:i<b?0xffffffffu:i==b?0xfffffffeu:0xffffffffu;
        if(c[i]!=want) atomicMin(bad,(unsigned)i);
    }
}
int main(int argc,char**argv){
    unsigned k=argc>1?atoi(argv[1]):29;
    if(k<13||k>31)return 2;
    auto ctx=batch_mp_init(); if(!ctx)return 1;
    unsigned *bad; CHECK(cudaMallocManaged(&bad,4));
    uint32_t n=1u<<(k-1);
    auto a=batch_mp_array_create(1,n); auto b=batch_mp_array_create(1,n); auto c=batch_mp_array_create(1,2*n);
    CHECK(cudaMemset(a.data,255,(size_t)n*4));CHECK(cudaMemset(b.data,255,(size_t)n*4));
    for(int t=0;t<4;++t){
        auto aa=a; auto bb=t==0?a:b; if(t==2)bb.length-=7;
        if(t==3){aa.length=n/2;bb.length=n/2+n/32;}
        auto cc=c;cc.length=aa.length+bb.length;
        *bad=UINT32_MAX;
        cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));CHECK(cudaEventRecord(start));
        CHECK(batch_mp_mul(ctx,aa,bb,cc));CHECK(cudaEventRecord(stop));CHECK(cudaEventSynchronize(stop));float ms;CHECK(cudaEventElapsedTime(&ms,start,stop));
        check_product<<<65535,256>>>(c.data,std::min(aa.length,bb.length),std::max(aa.length,bb.length),bad);CHECK(cudaDeviceSynchronize());
        printf("k=%u case=%s a=%u b=%u ms=%.3f first_bad=%u\n",k,t==0?"square":t==1?"multiply":t==2?"unequal":"cyclic",aa.length,bb.length,ms,*bad);fflush(stdout);
        if(*bad!=UINT32_MAX)return 1;
        CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));
    }
    // Two full-sized outputs share B, including the 2^31 transform cache.
    auto d=batch_mp_array_create(1,2*n);
    auto shorter=a;shorter.length=n-7;
    auto dout=d;dout.length=2*n-7;
    *bad=UINT32_MAX;
    CHECK(batch_mp_mul_shared_b(ctx,a,shorter,b,c,dout));
    check_product<<<65535,256>>>(c.data,n,n,bad);CHECK(cudaDeviceSynchronize());
    if(*bad!=UINT32_MAX)return 1;
    check_product<<<65535,256>>>(d.data,n-7,n,bad);CHECK(cudaDeviceSynchronize());
    if(*bad!=UINT32_MAX)return 1;
    batch_mp_array_release(d);
    printf("k=%u shared transform: both complete outputs passed\n",k);
    uint64_t bits=0;CHECK(batch_mp_bitlength_max(ctx,a,&bits));
    if(bits!=(uint64_t)n*32){fprintf(stderr,"bitlength mismatch: %llu\n",(unsigned long long)bits);return 1;}
    if (k >= 29) {
        uint32_t word=0xdeadbeef, got[2]={};
        CHECK(cudaMemcpy(a.data,&word,4,cudaMemcpyHostToDevice));
        auto one=a;one.length=1;
        const int64_t shift=(1ll<<33)+5;
        CHECK(batch_mp_shift_bits(ctx,one,c,shift));
        auto two=b;two.length=2;
        CHECK(batch_mp_shift_bits(ctx,c,two,-shift));
        CHECK(cudaMemcpy(got,b.data,8,cudaMemcpyDeviceToHost));
        if(got[0]!=word || got[1])return 1;
        CHECK(batch_mp_bitlength_max(ctx,c,&bits));
        if(bits!=(uint64_t)shift+32)return 1;
        CHECK(batch_mp_shift_bits(ctx,c,two,INT64_MIN));
        CHECK(cudaMemcpy(got,b.data,8,cudaMemcpyDeviceToHost));
        if(got[0]||got[1])return 1;
        CHECK(batch_mp_shift_bits(ctx,one,two,INT64_MAX));
        CHECK(cudaMemcpy(got,b.data,8,cudaMemcpyDeviceToHost));
        if(got[0]||got[1])return 1;
        printf("64-bit shifts and bitlength passed\n");
    }
    batch_mp_array_release(a);batch_mp_array_release(b);batch_mp_array_release(c);cudaFree(bad);batch_mp_destroy(ctx);
}
