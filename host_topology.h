#pragma once
#include <sched.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace host_topology {
inline bool automatic=false;
inline bool enabled(){const char*s=getenv("GPUPI_HOST_AFFINITY");return s?atoi(s)!=0:automatic;}
inline std::atomic<size_t> ticket{0};
inline const std::vector<int>& cores(){
    static const auto result=[] {
        cpu_set_t allowed;CPU_ZERO(&allowed);sched_getaffinity(0,sizeof(allowed),&allowed);
        std::map<int,std::vector<int>> packages;std::set<std::pair<int,int>> seen;
        std::vector<int> fallback;
        for(int cpu=0;cpu<CPU_SETSIZE;cpu++)if(CPU_ISSET(cpu,&allowed)){
            fallback.push_back(cpu);int package=-1,core=-1;
            std::string base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
            std::ifstream(base+"physical_package_id")>>package;std::ifstream(base+"core_id")>>core;
            if(package>=0&&core>=0&&seen.insert({package,core}).second)packages[package].push_back(cpu);
        }
        std::vector<int> cpus;
        for(size_t i=0;;i++){bool found=false;for(auto&p:packages)if(i<p.second.size()){cpus.push_back(p.second[i]);found=true;}if(!found)break;}
        return cpus.empty()?fallback:cpus;
    }();return result;
}
template<class Fn>void run(size_t index,Fn fn){
    if(!enabled()||cores().empty()){fn();return;}
    cpu_set_t saved,mask;CPU_ZERO(&saved);sched_getaffinity(0,sizeof(saved),&saved);CPU_ZERO(&mask);CPU_SET(cores()[index%cores().size()],&mask);
    sched_setaffinity(0,sizeof(mask),&mask);
    try{fn();}catch(...){sched_setaffinity(0,sizeof(saved),&saved);throw;}
    sched_setaffinity(0,sizeof(saved),&saved);
}
}
