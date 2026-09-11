#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <thread>
#define CU(x) do { cudaError_t check_error=(x); if(check_error!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(check_error)); exit(1);} } while(0)
int main(int argc,char**argv){
    int n=0; CU(cudaGetDeviceCount(&n));
    size_t bytes=argc>1?strtoull(argv[1],0,10):256ull<<20;
    std::vector<void*> a(n),b(n);
    for(int i=0;i<n;i++){CU(cudaSetDevice(i));CU(cudaMalloc(&a[i],bytes));CU(cudaMalloc(&b[i],bytes));CU(cudaMemset(a[i],i+1,bytes));CU(cudaDeviceSynchronize());}
    puts("source,destination,peer_access,GB_per_second");
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){
        CU(cudaSetDevice(j)); int access=0; if(i!=j){CU(cudaDeviceCanAccessPeer(&access,j,i));if(access){auto e=cudaDeviceEnablePeerAccess(i,0);if(e!=cudaSuccess && e!=cudaErrorPeerAccessAlreadyEnabled)CU(e);}}
        CU(cudaMemcpyPeer(b[j],j,a[i],i,bytes));CU(cudaDeviceSynchronize());
        auto t=std::chrono::steady_clock::now();
        for(int r=0;r<4;r++)CU(cudaMemcpyPeerAsync(b[j],j,a[i],i,bytes));
        CU(cudaDeviceSynchronize());
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        printf("%d,%d,%d,%.3f\n",i,j,access,4*bytes/sec/1e9);fflush(stdout);
    }
    void*h; CU(cudaHostAlloc(&h,bytes,cudaHostAllocPortable));
    for(int i=0;i<n;i++){
        CU(cudaSetDevice(i));
        for(int dir=0;dir<2;dir++){
            auto t=std::chrono::steady_clock::now();
            for(int r=0;r<4;r++)CU(cudaMemcpyAsync(dir?h:b[i],dir?a[i]:h,bytes,dir?cudaMemcpyDeviceToHost:cudaMemcpyHostToDevice));
            CU(cudaDeviceSynchronize());
            printf("host_gpu_%d,%s,%.3f\n",i,dir?"D2H":"H2D",4*bytes/std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/1e9);
        }
        CU(cudaFree(a[i]));CU(cudaFree(b[i]));
    }
    CU(cudaFreeHost(h));
}
