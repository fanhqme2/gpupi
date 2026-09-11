#include <gmp.h>
#include <sched.h>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <chrono>
int main(){
    cpu_set_t allowed;sched_getaffinity(0,sizeof(allowed),&allowed);
    std::map<int,std::vector<int>> packages;std::set<std::pair<int,int>> seen;
    for(int cpu=0;cpu<CPU_SETSIZE;cpu++)if(CPU_ISSET(cpu,&allowed)){
        int package=-1,core=-1;std::string base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
        std::ifstream(base+"physical_package_id")>>package;std::ifstream(base+"core_id")>>core;
        if(seen.insert({package,core}).second)packages[package].push_back(cpu);
    }
    std::vector<int> cpus;
    for(size_t i=0;;i++){bool found=false;for(auto&p:packages)if(i<p.second.size()){cpus.push_back(p.second[i]);found=true;}if(!found)break;}
    constexpr size_t bytes=1ull<<30,n=bytes/8;
    for(bool bound:{false,true}){
        auto*a=(mp_limb_t*)mmap(nullptr,bytes*3,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0),*b=a+n,*c=b+n;
        if(a==MAP_FAILED)return 1;madvise(a,bytes*3,MADV_HUGEPAGE);
        auto parallel=[&](int count,auto fn){std::vector<std::thread>jobs;
            for(int j=0;j<count;j++)jobs.emplace_back([&,j]{if(bound&&!cpus.empty()){cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpus[j%cpus.size()],&mask);sched_setaffinity(0,sizeof(mask),&mask);}fn(n*j/count,n*(j+1)/count);});
            for(auto&t:jobs)t.join();
        };
        parallel(32,[&](size_t l,size_t r){std::memset(a+l,1,(r-l)*8);std::memset(b+l,2,(r-l)*8);std::memset(c+l,0,(r-l)*8);});
        for(int threads:{8,32,64}){
            auto t=std::chrono::steady_clock::now();
            for(int i=0;i<8;i++)parallel(threads,[&](size_t l,size_t r){mpn_add_n(c+l,a+l,b+l,r-l);});
            double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
            printf("bound=%d threads=%d output_GBps=%.3f seconds=%.6f\n",bound,threads,8*bytes/s/1e9,s);fflush(stdout);
        }munmap(a,bytes*3);
    }
}
