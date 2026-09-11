#include "distributed_ntt.h"
#include <cuda_runtime.h>
#include <gmpxx.h>
#include "pinned_gmp.h"
#include <vector>
#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <atomic>
#include "host_arith.h"
int main(int argc,char**argv){try{
    pinned_gmp::Guard guard;if(getenv("GPUPI_MGPU_PINNED"))pinned_gmp::install(16ull<<30);
    int g=1;cudaGetDeviceCount(&g);if(getenv("GPUPI_GPUS"))g=atoi(getenv("GPUPI_GPUS"));
    DistributedNTT engine(g);gmp_randclass random(gmp_randinit_default);random.seed(847);
    if(argc>1&&(std::string(argv[1])=="--max"||std::string(argv[1])=="--wrap")){
        bool wrap=std::string(argv[1])=="--wrap";
        int k=argc>2?atoi(argv[2]):24;if(k<10||k>(wrap?32:31))throw std::runtime_error("invalid maximal test exponent");
        size_t size=(size_t(1)<<k);if(wrap)size=size/8*5;
        size_t na=size-17,nb=size-43;
        mpz_class a,b,out;
        auto fill=[](mpz_class&z,size_t n){auto*p=mpz_limbs_write(z.get_mpz_t(),(n+1)/2);host_arith::parallel((n+1)/2,[&](size_t l,size_t r,size_t){std::memset(p+l,255,(r-l)*8);});if(n&1)p[n/2]=0xffffffff;mpz_limbs_finish(z.get_mpz_t(),(n+1)/2);};
        fill(a,na);fill(b,nb);
        for(bool square:{false,true}){
            size_t n2=square?na:nb,total=na+n2,shorter=std::min(na,n2),longer=std::max(na,n2);
            size_t first=wrap?(total/2)&~size_t(1):0,count=total-first;
            auto*p=mpz_limbs_write(out.get_mpz_t(),count/2);auto t=std::chrono::steady_clock::now();
            if(wrap)engine.multiply_window((const uint32_t*)mpz_limbs_read(a.get_mpz_t()),na,(const uint32_t*)mpz_limbs_read((square?a:b).get_mpz_t()),n2,(uint32_t*)p,first);
            else engine.multiply((const uint32_t*)mpz_limbs_read(a.get_mpz_t()),na,(const uint32_t*)mpz_limbs_read((square?a:b).get_mpz_t()),n2,(uint32_t*)p);
            double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
            std::atomic<bool>failed{false};auto*words=(uint32_t*)p;
            host_arith::parallel(count,[&](size_t l,size_t r,size_t){for(size_t j=l;j<r;j++){
                size_t i=j+first;
                uint32_t expected=i==0?1:i<shorter?0:i<longer?0xffffffffu:i==longer?0xfffffffeu:0xffffffffu;
                if(words[j]!=expected){failed=true;break;}
            }});
            if(failed)throw std::runtime_error("maximal product limb mismatch");
            mpz_limbs_finish(out.get_mpz_t(),count/2);
            printf("maximal k=%d gpus=%d square=%d wrap=%d seconds=%.6f ALL %zu LIMBS PASS\n",k,g,square,wrap,s,count);fflush(stdout);
        }return 0;
    }
    bool wrap_random=argc>1&&std::string(argv[1])=="--wrap-random";
    int maxk=wrap_random?(argc>2?atoi(argv[2]):22):argc>1?atoi(argv[1]):22;
    if(maxk<10||maxk>31)throw std::runtime_error("random test exponent must be 10..31");
    for(int k=10;k<=maxk;k++){
        size_t size=(size_t(1)<<k);if(wrap_random)size=size/8*5;
        size_t na=size-17,nb=size-43;
        mpz_class a=random.get_z_bits(na*32),b=random.get_z_bits(nb*32),out;
        auto*p=mpz_limbs_write(out.get_mpz_t(),(na+nb+1)/2);
        auto t=std::chrono::steady_clock::now();
        if(wrap_random)engine.multiply_window((const uint32_t*)mpz_limbs_read(a.get_mpz_t()),na,(const uint32_t*)mpz_limbs_read(b.get_mpz_t()),nb,(uint32_t*)p,0);
        else engine.multiply((const uint32_t*)mpz_limbs_read(a.get_mpz_t()),na,(const uint32_t*)mpz_limbs_read(b.get_mpz_t()),nb,(uint32_t*)p);
        mpz_limbs_finish(out.get_mpz_t(),(na+nb)/2);
        double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        if(k<=25){mpz_class ref=a*b;if(out!=ref)throw std::runtime_error("random product mismatch at k="+std::to_string(k));
            if(wrap_random)for(size_t first:{na/3,na+nb-17}){size_t count=na+nb-first;mpz_class high;auto*dst=mpz_limbs_write(high.get_mpz_t(),(count+1)/2);dst[(count-1)/2]=0;
                engine.multiply_window((const uint32_t*)mpz_limbs_read(a.get_mpz_t()),na,(const uint32_t*)mpz_limbs_read(b.get_mpz_t()),nb,(uint32_t*)dst,first);
                mpz_limbs_finish(high.get_mpz_t(),(count+1)/2);if(high!=(ref>>(first*32)))throw std::runtime_error("random cyclic window mismatch");
            }
        }
        else for(unsigned long mod:{1000000007ul,1000000009ul,2147483647ul})if(mpz_fdiv_ui(out.get_mpz_t(),mod)!=(uint64_t)mpz_fdiv_ui(a.get_mpz_t(),mod)*mpz_fdiv_ui(b.get_mpz_t(),mod)%mod)throw std::runtime_error("checksum mismatch");
        printf("distributed k=%d gpus=%d seconds=%.6f PASS\n",k,g,s);fflush(stdout);
    }
}catch(const std::exception&e){fprintf(stderr,"ERROR: %s\n",e.what());return 1;}}
