#include <cuda_runtime.h>
#include <gmp.h>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
int main(){
    constexpr size_t bytes=1ull<<30,limbs=bytes/8;
    for(int pinned=0;pinned<2;pinned++){
        mp_limb_t*a,*b,*c;
        auto alloc=[&](mp_limb_t**p){if(pinned){if(cudaHostAlloc((void**)p,bytes,cudaHostAllocPortable)!=cudaSuccess)exit(1);}else *p=(mp_limb_t*)malloc(bytes);};
        alloc(&a);alloc(&b);alloc(&c);memset(a,1,bytes);memset(b,2,bytes);memset(c,0,bytes);
        for(int threads:{1,8,32}){
            auto t=std::chrono::steady_clock::now();
            for(int trial=0;trial<4;trial++){
                std::vector<std::thread>jobs;
                for(int j=0;j<threads;j++)jobs.emplace_back([&,j]{size_t start=limbs*j/threads,end=limbs*(j+1)/threads;mpn_add_n(c+start,a+start,b+start,end-start);});
                for(auto&job:jobs)job.join();
            }
            double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
            printf("pinned=%d threads=%d add_output_GBps=%.3f seconds=%.6f\n",pinned,threads,4*bytes/s/1e9,s);fflush(stdout);
        }
        if(pinned){cudaFreeHost(a);cudaFreeHost(b);cudaFreeHost(c);}else{free(a);free(b);free(c);}
    }
}
