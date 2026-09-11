#include <cuda_runtime.h>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(int argc,char**argv){
    size_t bytes=(argc>1?strtoull(argv[1],nullptr,10):1)*(1ull<<30);
    int devices=argc>2?atoi(argv[2]):1;
    for(int d=0;d<devices;d++){check(cudaSetDevice(d));check(cudaFree(nullptr));}
    for(int mode=0;mode<4;mode++)for(int trial=0;trial<2;trial++){
        void*p=nullptr;auto t=std::chrono::steady_clock::now();
        if(!mode)check(cudaHostAlloc(&p,bytes,cudaHostAllocPortable));
        else{
            p=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
            if(p==MAP_FAILED)return 2;
            if(mode>=2)madvise(p,bytes,MADV_HUGEPAGE);
            std::vector<std::thread>jobs;
            for(int j=0;j<32;j++)jobs.emplace_back([&,j]{std::memset((char*)p+bytes*j/32,0,bytes*(j+1)/32-bytes*j/32);});
            for(auto&job:jobs)job.join();
            check(cudaHostRegister(p,bytes,mode==3?cudaHostRegisterDefault:cudaHostRegisterPortable));
        }
        double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        printf("mode=%d trial=%d GB=%.3f alloc_s=%.6f GBps=%.3f\n",mode,trial,bytes/1e9,s,bytes/s/1e9);fflush(stdout);
        if(!mode)check(cudaFreeHost(p));else{check(cudaHostUnregister(p));munmap(p,bytes);}
    }
}
