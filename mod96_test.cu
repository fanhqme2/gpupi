#include "ntt_mod96.cuh"
#include <gmp.h>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t e) { if(e != cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(1); } }
__global__ void calculate(const uint3 *a, const uint3 *b, uint3 *out, unsigned n) {
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        out[4*i]=mul_mod(a[i],b[i]); out[4*i+1]=add_mod(a[i],b[i]); out[4*i+2]=sub_mod(a[i],b[i]); out[4*i+3]=scale_inverse_power2(a[i],i%33);
    }
}
__global__ void throughput(uint3 *a, const uint3 *b, unsigned n) {
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        uint3 x=a[i],y=b[i];
        #pragma unroll 1
        for(int j=0;j<64;++j) { x=mul_mod(x,y); y=add_mod(y,x); }
        a[i]=x;
    }
}
int main() {
    constexpr unsigned N=131072;
    std::mt19937 r(1789);
    std::vector<uint3>a(N),b(N),out(4*N);
    std::vector<uint3>edge={{0,0,0},{1,0,0},{0xffffffff,0,0},{0,1,0},{0,0,1},
        {0,0xffffffff,0xffffffff},{1,0xffffffff,0xfffffffe},{0xffffffff,0xfffffffe,0xffffffff},
        {0,0,0xffffffff},{0xffffffff,0xffffffff,0xfffffffe}};
    for(unsigned i=0;i<N;++i) { a[i]={(uint32_t)r(),(uint32_t)r(),(uint32_t)r()}; b[i]={(uint32_t)r(),(uint32_t)r(),(uint32_t)r()}; }
    for(unsigned i=0;i<edge.size();++i) for(unsigned j=0;j<edge.size();++j) { a[i*edge.size()+j]=edge[i];b[i*edge.size()+j]=edge[j]; }
    // Exercise every scaling exponent at each carry/canonicalization boundary.
    const unsigned edge_start = ((edge.size()*edge.size()+32)/33)*33;
    for(unsigned i=0;i<edge.size();++i) for(unsigned k=0;k<=32;++k) {
        a[edge_start+i*33+k]=edge[i]; b[edge_start+i*33+k]=edge.back();
    }
    uint3 *da,*db,*dout;check(cudaMalloc(&da,N*sizeof(uint3)));check(cudaMalloc(&db,N*sizeof(uint3)));check(cudaMalloc(&dout,4*N*sizeof(uint3)));
    check(cudaMemcpy(da,a.data(),N*sizeof(uint3),cudaMemcpyHostToDevice));check(cudaMemcpy(db,b.data(),N*sizeof(uint3),cudaMemcpyHostToDevice));
    calculate<<<512,256>>>(da,db,dout,N);check(cudaMemcpy(out.data(),dout,4*N*sizeof(uint3),cudaMemcpyDeviceToHost));
    mpz_t x,y,z,p,actual;mpz_inits(x,y,z,p,actual,nullptr);mpz_set_str(p,"ffffffffffffffff00000001",16);
    for(unsigned i=0;i<N;++i) {
        mpz_import(x,3,-1,4,0,0,&a[i]);mpz_import(y,3,-1,4,0,0,&b[i]);
        for(int op=0;op<4;++op) {
            if(op==0)mpz_mul(z,x,y);else if(op==1)mpz_add(z,x,y);else if(op==2) mpz_sub(z,x,y); else { mpz_set_ui(z,2);mpz_powm_ui(z,z,i%33,p);mpz_invert(z,z,p);mpz_mul(z,z,x); }
            mpz_mod(z,z,p);mpz_import(actual,3,-1,4,0,0,&out[4*i+op]);
            if(mpz_cmp(actual,z)) { gmp_fprintf(stderr,"FAIL i=%u op=%d a=%Zx b=%Zx got=%Zx expected=%Zx\n",i,op,x,y,actual,z);return 1; }
        }
    }
    puts("PASS: 131072 modular multiply/add/subtract/scaling cases against GMP, including boundary pairs");
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    throughput<<<512,256>>>(da,db,N);check(cudaDeviceSynchronize());
    for(int trial=0;trial<5;++trial) {
        check(cudaEventRecord(start));for(int j=0;j<20;++j)throughput<<<512,256>>>(da,db,N);
        check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));float ms;check(cudaEventElapsedTime(&ms,start,stop));printf("throughput_trial=%d ms=%.3f\n",trial,ms);
    }
    mpz_clears(x,y,z,p,actual,nullptr);cudaFree(da);cudaFree(db);cudaFree(dout);cudaEventDestroy(start);cudaEventDestroy(stop);
}
