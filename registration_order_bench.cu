#include <cuda_runtime.h>
#include "pinned_gmp.h"
#include <chrono>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>
static void cu(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(int argc,char**argv){
    int mode=argc>1?atoi(argv[1]):0,g=0;cu(cudaGetDeviceCount(&g));g=std::min(g,8);
    size_t bytes=(argc>2?strtoull(argv[2],nullptr,10):16)*(1ull<<30),stride=((bytes-sizeof(pinned_gmp::Header))/g)&~size_t(4095);
    std::vector<void*>device(g);pinned_gmp::Header*h=nullptr;
    auto parallel=[&](auto fn){std::vector<std::thread>jobs;for(int d=0;d<g;d++)jobs.emplace_back([&,d]{cu(cudaSetDevice(d));fn(d);});for(auto&t:jobs)t.join();};
    auto initialize=[&]{parallel([&](int d){cu(cudaMalloc(&device[d],stride));});};
    auto start=std::chrono::steady_clock::now();
    if(mode){h=pinned_gmp::raw_allocate(bytes);initialize();}else{initialize();h=pinned_gmp::raw_allocate(bytes);}
    if(!h)return 2;
    auto allocated=std::chrono::steady_clock::now();
    parallel([&](int d){cu(cudaMemcpy(device[d],(char*)(h+1)+d*stride,stride,cudaMemcpyHostToDevice));uint32_t word=1;cu(cudaMemcpy(&word,device[d],4,cudaMemcpyDeviceToHost));if(word)exit(3);});
    auto finish=std::chrono::steady_clock::now();
    printf("register_first=%d gpus=%d GB=%.3f init_and_alloc_s=%.6f transfer_s=%.6f total_s=%.6f\n",mode,g,bytes/1e9,std::chrono::duration<double>(allocated-start).count(),std::chrono::duration<double>(finish-allocated).count(),std::chrono::duration<double>(finish-start).count());
    parallel([&](int d){cu(cudaFree(device[d]));});pinned_gmp::free_pinned(h);
}
