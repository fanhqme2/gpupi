#include "batch_arith.h"
#include <gmp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
int main(){
 auto ctx=batch_mp_init();if(!ctx)return 1;
 std::mt19937 rng(1871);
 mpz_t divisor,q,input;mpz_inits(divisor,q,input,nullptr);
 for(unsigned count:{0u,1u,2u,4u,11u,22u,32u}){
  std::vector<uint32_t> factors(count);
  mpz_set_ui(divisor,1);
  for(auto &b:factors){b=rng()|1u;mpz_mul_ui(divisor,divisor,b);}
  for(unsigned length:{64u,129u,512u,2048u})for(unsigned n:{1u,33u,65u}){
   unsigned stride=length+3;
   std::vector<uint32_t> host((size_t)n*stride,0),expected(host.size(),0),words(length-count-1);
   for(unsigned row=0;row<n;++row){
    for(auto &w:words)w=rng();
    mpz_import(q,words.size(),-1,4,0,0,words.data());mpz_mul(input,q,divisor);
    size_t written=0;
    mpz_export(host.data()+(size_t)row*stride,&written,-1,4,0,0,input);
    if(written>length)return 1;
    mpz_export(expected.data()+(size_t)row*stride,&written,-1,4,0,0,q);
    for(unsigned j=length;j<stride;++j)host[(size_t)row*stride+j]=expected[(size_t)row*stride+j]=0xa5a5a5a5u;
   }
   auto a=batch_mp_array_create(n,stride);if(!a.data)return 1;a.length=length;
   CHECK(cudaMemcpy(a.data,host.data(),host.size()*4,cudaMemcpyHostToDevice));
   CHECK(batch_mp_exactdiv_small_many(ctx,a,factors.data(),count));
   CHECK(cudaMemcpy(host.data(),a.data,host.size()*4,cudaMemcpyDeviceToHost));
   if(host!=expected){fprintf(stderr,"Mismatch count=%u n=%u length=%u\n",count,n,length);return 1;}
   batch_mp_array_release(a);
  }
  printf("PASS exact division chain count=%u through 2048 limbs\n",count);fflush(stdout);
 }
 auto a=batch_mp_array_create(1,4);uint32_t invalid[]={3,0,5};
 if(batch_mp_exactdiv_small_many(ctx,a,invalid,3)!=cudaErrorInvalidValue)return 1;
 invalid[1]=2;if(batch_mp_exactdiv_small_many(ctx,a,invalid,3)!=cudaErrorInvalidValue)return 1;
 if(batch_mp_exactdiv_small_many(ctx,a,nullptr,1)!=cudaErrorInvalidValue)return 1;
 if(batch_mp_exactdiv_small_many(ctx,a,invalid,33)!=cudaErrorInvalidValue)return 1;
 batch_mp_array_release(a);mpz_clears(divisor,q,input,nullptr);batch_mp_destroy(ctx);
 puts("All exact division chain checks passed");
}
