#include "distributed_ntt.h"
#include "ntt_mod96.cuh"
#include "batch_mul_ntt.h"
#include <cub/device/device_scan.cuh>
#include <vector>
#include <thread>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <gmpxx.h>
#include "host_arith.h"

namespace dntt {
inline void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ __forceinline__ uint3 root(uint32_t index,const NTTPrecomputedTables&t,bool inverse=false){
    if(inverse)index=__brev(-__brev(index));
    uint3 w=t.roots_table_lv2[index&65535];
    return index>>16?mul_mod(w,t.roots_table_lv1[index>>16]):w;
}
// The outer radix acts along columns; each device initially owns a disjoint
// row range. The transpose gives each device one complete frequency column.
template<bool inverse>
__global__ void outer(const void*input,uint3*output,size_t tile,int lg,NTTPrecomputedTables t){
    int g=1<<lg;
    for(size_t j=(size_t)blockIdx.x*blockDim.x+threadIdx.x;j<tile;j+=(size_t)blockDim.x*gridDim.x){
        uint3 v[8];
        for(int c=0;c<g;c++)v[c]=inverse?((const uint3*)input)[c*tile+j]:make_uint3(((const uint32_t*)input)[c*tile+j],0,0);
        for(int iteration=0;iteration<lg;iteration++){
            int lv=inverse?iteration:lg-1-iteration,step=1<<lv;
            for(int x=0;x<g/2;x++){
                int ix=((x>>lv)<<lv)+x;
                uint3 u=v[ix],w=v[ix+step];uint3 tw=root((ix>>(lv+1))*2,t,inverse);
                if(inverse){v[ix]=add_mod(u,w);v[ix+step]=mul_mod(sub_mod(u,w),tw);}
                else{w=mul_mod(w,tw);v[ix]=add_mod(u,w);v[ix+step]=sub_mod(u,w);}
            }
        }
        for(int c=0;c<g;c++)output[c*tile+j]=v[c];
    }
}
template<int rb,bool inverse>
__global__ void inner_radix(uint3*parts,int k,int i,uint32_t rank,NTTPrecomputedTables t){
    constexpr int R=1<<rb;
    __shared__ uint3 v[R][16];
    uint32_t step=1u<<(k-rb-i);
    for(size_t j=(size_t)threadIdx.x+blockIdx.x*blockDim.x;j<(size_t(1)<<(k-rb));j+=(size_t)gridDim.x*blockDim.x){
        size_t group=j>>(k-rb-i),offset=group<<(k-i);
        uint32_t seq=(uint32_t)group|(rank<<i),pos=j&(step-1);
        for(int x=threadIdx.y;x<R;x+=blockDim.y)v[x][threadIdx.x]=parts[offset+pos+(size_t)step*x];
        __syncthreads();
        for(int iteration=0;iteration<rb;iteration++){
            int lv=inverse?iteration:rb-1-iteration,stride=1<<lv;
            for(int x=threadIdx.y;x<R/2;x+=blockDim.y){
                int ix=((x>>lv)<<lv)+x;
                uint32_t ri=((seq<<(rb-1-lv))|(ix>>(lv+1)))*2;
                uint3 a=v[ix][threadIdx.x],b=v[ix+stride][threadIdx.x],w=root(ri,t,inverse);
                if(inverse){v[ix][threadIdx.x]=add_mod(a,b);v[ix+stride][threadIdx.x]=mul_mod(sub_mod(a,b),w);}
                else{b=mul_mod(b,w);v[ix][threadIdx.x]=add_mod(a,b);v[ix+stride][threadIdx.x]=sub_mod(a,b);}
            }__syncthreads();
        }
        for(int x=threadIdx.y;x<R;x+=blockDim.y)parts[offset+pos+(size_t)step*x]=v[x][threadIdx.x];
        __syncthreads();
    }
}
template<bool square>
__global__ void inner_product(uint3*a,const uint3*b,int k,int i,uint32_t rank,int lg,NTTPrecomputedTables t){
    __shared__ uint3 va[1024],vb[1024];
    int depth=k-i,size=1<<depth;
    for(size_t j=(size_t)blockIdx.x<<depth;j<(size_t(1)<<k);j+=(size_t)gridDim.x<<depth){
        uint32_t seq=(j>>depth)|(rank<<i);
        for(int x=threadIdx.x;x<size;x+=blockDim.x){va[x]=a[j+x];if constexpr(!square)vb[x]=b[j+x];}__syncthreads();
        for(int lv=depth-1;lv>=0;--lv){
            for(int x=threadIdx.x;x<size/2;x+=blockDim.x){
                int ix=((x>>lv)<<lv)+x,stride=1<<lv;
                uint3 w=root(((seq<<(depth-1-lv))|(ix>>(lv+1)))*2,t);
                uint3 u=va[ix],v=mul_mod(va[ix+stride],w);va[ix]=add_mod(u,v);va[ix+stride]=sub_mod(u,v);
                if constexpr(!square){u=vb[ix];v=mul_mod(vb[ix+stride],w);vb[ix]=add_mod(u,v);vb[ix+stride]=sub_mod(u,v);}
            }__syncthreads();
        }
        for(int x=threadIdx.x;x<size;x+=blockDim.x)va[x]=scale_inverse_power2(mul_mod(va[x],square?va[x]:vb[x]),k+lg);
        __syncthreads();
        for(int lv=0;lv<depth;++lv){
            for(int x=threadIdx.x;x<size/2;x+=blockDim.x){
                int ix=((x>>lv)<<lv)+x,stride=1<<lv;
                uint3 w=root(((seq<<(depth-1-lv))|(ix>>(lv+1)))*2,t,true),u=va[ix],v=va[ix+stride];
                va[ix]=add_mod(u,v);va[ix+stride]=mul_mod(sub_mod(u,v),w);
            }__syncthreads();
        }
        for(int x=threadIdx.x;x<size;x+=blockDim.x)a[j+x]=va[x];__syncthreads();
    }
}
__global__ void boundaries(const uint3*a,uint3*tail,size_t tile,int g){
    int c=threadIdx.x/2,j=threadIdx.x%2;if(c<g)tail[c*2+j]=a[(c+1)*tile-2+j];
}
__global__ void unfold(uint3*a,const uint3*reversed,size_t n){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x)a[i]=sub_mod(a[i],reversed[n-1-i]);
}
// Six-bit maps on incoming carries {0,1,2}. Composition is associative.
__host__ __device__ inline uint32_t compose(uint32_t after,uint32_t before){
    return ((after>>(2*(before&3)))&3)|(((after>>(2*((before>>2)&3)))&3)<<2)|(((after>>(2*((before>>4)&3)))&3)<<4);
}
struct Summary {uint32_t function,head;};
struct ScanOp {
    __host__ __device__ Summary operator()(Summary a,Summary b)const{return {b.head?b.function:compose(b.function,a.function),a.head|b.head};}
};
__device__ inline uint64_t coefficient_sum(const uint3*a,const uint3*prefix,size_t i,size_t tile){
    size_t c=i/tile,j=i%tile;
    uint3 prev=j?a[i-1]:prefix[c*2+1],prev2=j>1?a[i-2]:prefix[c*2+(j?1:0)];
    // At j=1, the coefficient two positions back is the preceding segment's
    // final coefficient; at j=0 it is the preceding segment's penultimate.
    return (uint64_t)a[i].x+prev.y+prev2.z;
}
template<bool emit>
__global__ void carry(const uint3*a,const uint3*prefix,Summary*summary,const Summary*scan,
    const uint32_t*incoming,uint32_t*out,size_t local,size_t tile){
    uint32_t lane=threadIdx.x&31;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<local;i+=(size_t)gridDim.x*blockDim.x){
        uint64_t sum=coefficient_sum(a,prefix,i,tile);uint32_t low=sum,hi=sum>>32;
        uint32_t f=hi|((hi+(low==0xffffffffu))<<2)|((hi+(low>=0xfffffffeu))<<4);
        for(int d=1;d<32;d*=2){uint32_t previous=__shfl_up_sync(0xffffffffu,f,d);if(lane>=d)f=compose(f,previous);}
        size_t warp=i/32,first=(i/tile)*(tile/32);
        if constexpr(emit){
            uint32_t c=incoming[i/tile];
            if(warp!=first)c=(scan[warp-1].function>>(2*c))&3;
            uint32_t previous=__shfl_up_sync(0xffffffffu,f,1);
            if(lane)c=(previous>>(2*c))&3;
            out[i]=low+c;
        }else if(lane==31)summary[warp]={f,warp==first};
    }
}
__global__ void last_maps(const Summary*scan,uint32_t*maps,size_t tile,int g){
    int c=threadIdx.x;if(c<g)maps[c]=scan[(c+1)*(tile/32)-1].function;
}
struct Device {
    uint3 *a=nullptr,*b=nullptr,*temp=nullptr,*edge=nullptr;
    uint32_t *maps=nullptr;size_t capacity=0;
    NTTPrecomputedTables tables{};
    cudaStream_t copies[8]{};
};

}

struct DistributedNTT::Impl {
    int g,lg=0;std::vector<dntt::Device> d;std::mutex mutex,window_mutex;
    struct End {uint32_t carry=0;uint3 prev2{},prev{};};
    template<class Fn>void parallel(Fn fn){
        std::vector<std::thread> threads;std::vector<std::exception_ptr>errors(g);
        for(int r=0;r<g;r++)threads.emplace_back([&,r]{try{dntt::cu(cudaSetDevice(r));fn(r,d[r]);dntt::cu(cudaDeviceSynchronize());}catch(...){errors[r]=std::current_exception();}});
        for(auto&t:threads)t.join();for(auto&e:errors)if(e)std::rethrow_exception(e);
    }
    explicit Impl(int n):g(n),d(n){
        if(n<1||n>8||(n&(n-1)))throw std::runtime_error("distributed NTT needs 1, 2, 4 or 8 devices");
        while((1<<lg)<n)++lg;
        parallel([&](int r,dntt::Device&w){
            bool enable=getenv("GPUPI_DNTT_ENABLE_PEER")&&atoi(getenv("GPUPI_DNTT_ENABLE_PEER"));
            for(int j=0;j<g;j++)if(j!=r){int access;dntt::cu(cudaDeviceCanAccessPeer(&access,r,j));if(!access)throw std::runtime_error("distributed NTT requires peer access");if(enable){auto e=cudaDeviceEnablePeerAccess(j,0);if(e!=cudaErrorPeerAccessAlreadyEnabled)dntt::cu(e);else cudaGetLastError();}}
            dntt::cu(cudaMalloc(&w.tables.roots_table_lv1,65536*sizeof(uint3)));
            dntt::cu(cudaMalloc(&w.tables.roots_table_lv2,65536*sizeof(uint3)));
            dntt::cu(cudaMalloc(&w.tables.inv2n_table,33*sizeof(uint3)));
            init_ntt_precomputed_tables(&w.tables);
            dntt::cu(cudaMalloc(&w.edge,4*g*sizeof(uint3)));
            dntt::cu(cudaMalloc(&w.maps,2*g*sizeof(uint32_t)));
            for(int j=0;j<g;j++)dntt::cu(cudaStreamCreateWithFlags(&w.copies[j],cudaStreamNonBlocking));
        });
    }
    ~Impl(){try{parallel([&](int,dntt::Device&w){cudaFree(w.a);cudaFree(w.b);cudaFree(w.temp);cudaFree(w.edge);cudaFree(w.maps);cudaFree(w.tables.roots_table_lv1);cudaFree(w.tables.roots_table_lv2);cudaFree(w.tables.inv2n_table);for(int j=0;j<g;j++)cudaStreamDestroy(w.copies[j]);});}catch(...){}}
    void reserve(size_t local){parallel([&](int,dntt::Device&w){if(local>w.capacity){dntt::cu(cudaFree(w.a));dntt::cu(cudaFree(w.b));dntt::cu(cudaFree(w.temp));w.a=w.b=w.temp=nullptr;w.capacity=0;dntt::cu(cudaMalloc(&w.a,local*12));dntt::cu(cudaMalloc(&w.b,local*12));dntt::cu(cudaMalloc(&w.temp,local*12));w.capacity=local;}});}
    void transpose(bool b,size_t tile){
        int streams=getenv("GPUPI_DNTT_COPY_STREAMS")?atoi(getenv("GPUPI_DNTT_COPY_STREAMS")):8;
        parallel([&](int r,dntt::Device&w){for(int step=0;step<g;step++){
            int src=(r+step)%g;cudaStream_t stream=streams?w.copies[streams>1?src:0]:nullptr;
            dntt::cu(cudaMemcpyPeerAsync(w.temp+src*tile,r,(b?d[src].b:d[src].a)+r*tile,src,tile*12,stream));
        }});
        for(auto&w:d)std::swap(b?w.b:w.a,w.temp);
    }
    End run(const uint32_t*a,size_t na,const uint32_t*b,size_t nb,uint32_t*out,
            size_t first=0,const uint3*reverse_tail=nullptr,size_t wrap=0,uint3*raw=nullptr,size_t raw_count=0){
        std::lock_guard<std::mutex>lock(mutex);
        if(first>na+nb)throw std::runtime_error("product window starts past end");
        if(na>(1ull<<32)||nb>(1ull<<32)||na+nb-wrap>(1ull<<32))throw std::runtime_error("distributed NTT exceeds 2^32 points");
        size_t K=std::max<size_t>(4096,g*g*32);int total_k=0;while(K<na+nb-wrap)K*=2;
        if(wrap&&(K!=na+nb-wrap||na>K||nb>K))throw std::runtime_error("invalid cyclic product dimensions");
        while((size_t(1)<<total_k)<K)++total_k;
        if(total_k>32)throw std::runtime_error("96-bit field NTT length exceeds 2^32");
        auto t0=std::chrono::steady_clock::now();
        size_t local=K/g,tile=local/g;int k=total_k-lg;reserve(local);
        auto t1=std::chrono::steady_clock::now();
        bool square=a==b&&na==nb;
        for(int operand=0;operand<(square?1:2);operand++){
            const uint32_t*src=operand?b:a;size_t n=operand?nb:na;
            parallel([&](int r,dntt::Device&w){
                uint32_t*raw=(uint32_t*)w.temp;dntt::cu(cudaMemset(raw,0,local*4));
                for(int c=0;c<g;c++){size_t offset=c*local+r*tile;if(offset<n)dntt::cu(cudaMemcpyAsync(raw+c*tile,src+offset,std::min(tile,n-offset)*4,cudaMemcpyHostToDevice));}
                dntt::outer<false><<<std::min<size_t>(65535,(tile+255)/256),256>>>(raw,operand?w.b:w.a,tile,lg,w.tables);dntt::cu(cudaGetLastError());
            });
            transpose(operand,tile);
        }
        auto t2=std::chrono::steady_clock::now();
        std::vector<std::pair<int,int>>schedule;int i=0;
        while(k-i>10){int rb=k-i>=18?8:5;schedule.push_back({i,rb});i+=rb;}
        parallel([&](int rank,dntt::Device&w){
            for(auto stage:schedule)for(int operand=0;operand<(square?1:2);operand++){
                auto*p=operand?w.b:w.a;
                if(stage.second==8)dntt::inner_radix<8,false><<<std::min<size_t>(65535,(local/256+15)/16),dim3(16,32)>>>(p,k,stage.first,rank,w.tables);
                else dntt::inner_radix<5,false><<<std::min<size_t>(65535,(local/32+15)/16),dim3(16,8)>>>(p,k,stage.first,rank,w.tables);
            }
            if(square)dntt::inner_product<true><<<std::min<size_t>(65535,local>>(k-i)),512>>>(w.a,w.b,k,i,rank,lg,w.tables);
            else dntt::inner_product<false><<<std::min<size_t>(65535,local>>(k-i)),512>>>(w.a,w.b,k,i,rank,lg,w.tables);
            for(auto it=schedule.rbegin();it!=schedule.rend();++it){
                if(it->second==8)dntt::inner_radix<8,true><<<std::min<size_t>(65535,(local/256+15)/16),dim3(16,32)>>>(w.a,k,it->first,rank,w.tables);
                else dntt::inner_radix<5,true><<<std::min<size_t>(65535,(local/32+15)/16),dim3(16,8)>>>(w.a,k,it->first,rank,w.tables);
            }dntt::cu(cudaGetLastError());
        });
        auto t3=std::chrono::steady_clock::now();
        transpose(false,tile);
        std::vector<std::vector<uint3>>tails(g,std::vector<uint3>(2*g));
        parallel([&](int rank,dntt::Device&w){
            dntt::outer<true><<<std::min<size_t>(65535,(tile+255)/256),256>>>(w.a,w.temp,tile,lg,w.tables);
            if(wrap)for(int c=0;c<g;c++){
                size_t offset=c*local+rank*tile;
                if(offset<wrap-1){size_t n=std::min(tile,wrap-1-offset);
                    dntt::cu(cudaMemcpyAsync(w.b+c*tile,reverse_tail+wrap-1-offset-n,n*12,cudaMemcpyHostToDevice));
                    dntt::unfold<<<std::min<size_t>(65535,(n+255)/256),256>>>(w.temp+c*tile,w.b+c*tile,n);
                }
            }
            dntt::boundaries<<<1,32>>>(w.temp,w.edge,tile,g);
            dntt::cu(cudaMemcpy(tails[rank].data(),w.edge,2*g*12,cudaMemcpyDeviceToHost));
            if(raw)for(int c=0;c<g;c++){size_t offset=c*local+rank*tile;if(offset<raw_count)dntt::cu(cudaMemcpyAsync(raw+offset,w.temp+c*tile,std::min(tile,raw_count-offset)*12,cudaMemcpyDeviceToHost));}
        });
        for(auto&w:d)std::swap(w.a,w.temp);
        if(raw)return {};
        auto t4=std::chrono::steady_clock::now();
        std::vector<std::vector<uint32_t>>maps(g,std::vector<uint32_t>(g));
        size_t warps=local/32;
        parallel([&](int rank,dntt::Device&w){
            std::vector<uint3>prefix(2*g,make_uint3(0,0,0));
            for(int c=0;c<g;c++){int previous_rank=rank?rank-1:g-1,previous_c=rank?c:c-1;if(previous_c>=0){prefix[c*2]=tails[previous_rank][previous_c*2];prefix[c*2+1]=tails[previous_rank][previous_c*2+1];}}
            dntt::cu(cudaMemcpy(w.edge,prefix.data(),2*g*12,cudaMemcpyHostToDevice));
            auto*summary=(dntt::Summary*)w.temp;auto*scan=summary+warps;
            void*scratch=scan+warps;size_t bytes=0;
            dntt::carry<false><<<std::min<size_t>(65535,(local+255)/256),256>>>(w.a,w.edge,summary,nullptr,nullptr,nullptr,local,tile);
            dntt::cu(cub::DeviceScan::InclusiveScan(nullptr,bytes,summary,scan,dntt::ScanOp(),warps));
            if(bytes+2*warps*sizeof(dntt::Summary)>w.capacity*12)throw std::runtime_error("scan scratch exceeds arena");
            dntt::cu(cub::DeviceScan::InclusiveScan(scratch,bytes,summary,scan,dntt::ScanOp(),warps));
            dntt::last_maps<<<1,32>>>(scan,w.maps,tile,g);
            dntt::cu(cudaMemcpy(maps[rank].data(),w.maps,g*4,cudaMemcpyDeviceToHost));
        });
        uint32_t incoming=0;
        for(int c=0;c<g;c++)for(int r=0;r<g;r++){uint32_t f=maps[r][c];maps[r][c]=incoming;incoming=(f>>(2*incoming))&3;}
        auto t5=std::chrono::steady_clock::now();
        if(incoming&&!wrap)throw std::runtime_error("nonzero carry beyond product length");
        parallel([&](int rank,dntt::Device&w){
            dntt::cu(cudaMemcpy(w.maps,maps[rank].data(),g*4,cudaMemcpyHostToDevice));
            auto*scan=(dntt::Summary*)w.temp+warps;uint32_t*result=(uint32_t*)w.b;
            dntt::carry<true><<<std::min<size_t>(65535,(local+255)/256),256>>>(w.a,w.edge,nullptr,scan,w.maps,result,local,tile);
            for(int c=0;c<g;c++){
                size_t offset=c*local+rank*tile,start=std::max(offset,first),end=std::min(offset+tile,na+nb);
                if(start<end)dntt::cu(cudaMemcpyAsync(out+start-first,result+c*tile+start-offset,(end-start)*4,cudaMemcpyDeviceToHost));
            }
        });
        if(getenv("GPUPI_DNTT_PROFILE")){
            auto sec=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};auto t6=std::chrono::steady_clock::now();
            fprintf(stderr,"dntt k=%d gpus=%d allocate=%.6f forward=%.6f inner=%.6f inverse=%.6f scan=%.6f output=%.6f total=%.6f\n",total_k,g,sec(t0,t1),sec(t1,t2),sec(t2,t3),sec(t3,t4),sec(t4,t5),sec(t5,t6),sec(t0,t6));
        }
        return {incoming,tails[g-1][2*g-2],tails[g-1][2*g-1]};
    }
    void window(const uint32_t*a,size_t na,const uint32_t*b,size_t nb,uint32_t*out,size_t first){
        std::lock_guard<std::mutex>lock(window_mutex);
        auto start=std::chrono::steady_clock::now();
        size_t L=na+nb,K=1;while(K*2<L)K*=2;
        if(first>L)throw std::runtime_error("product window starts past end");
        if(K<4096||L-K<=1||L-K>K/4||na>K||nb>K){run(a,na,b,nb,out,first);return;}
        size_t wrap=L-K,t=wrap-1;
        mpz_class reversed_a,reversed_b,coefficients;
        auto*ra=(uint32_t*)mpz_limbs_write(reversed_a.get_mpz_t(),(t+1)/2);
        bool square=a==b&&na==nb;
        auto*rb=square?ra:(uint32_t*)mpz_limbs_write(reversed_b.get_mpz_t(),(t+1)/2);
        host_arith::parallel(t,[&](size_t l,size_t r,size_t){for(size_t i=l;i<r;i++){ra[i]=a[na-1-i];if(!square)rb[i]=b[nb-1-i];}});
        auto*tail=(uint3*)mpz_limbs_write(coefficients.get_mpz_t(),(t*12+7)/8);
        run(ra,t,rb,t,nullptr,0,nullptr,0,tail,t);
        reversed_a=mpz_class();reversed_b=mpz_class();
        End end=run(a,na,b,nb,out,first,tail,wrap);
        // Continue the ordinary base-2^32 carry through the corrected tail.
        // Each CPU chunk first assumes zero incoming carry; a short boundary
        // pass adds the preceding chunk's carry exactly, including long runs.
        std::unique_ptr<uint32_t[]> temporary;
        uint32_t*words;
        if(first<=K)words=out+K-first;
        else{temporary.reset(new uint32_t[wrap]);words=temporary.get();}
        struct Chunk {size_t l=0,r=0;uint32_t carry=0;};Chunk chunks[32];
        auto coefficient=[&](int64_t i){return i<0?(i==-1?end.prev:end.prev2):size_t(i)<t?tail[t-1-i]:make_uint3(0,0,0);};
        host_arith::parallel(wrap,[&](size_t l,size_t r,size_t thread){
            uint32_t carry=0;uint3 previous=coefficient((int64_t)l-1),previous2=coefficient((int64_t)l-2);
            for(size_t i=l;i<r;i++){
                uint3 current=coefficient(i);uint64_t sum=(uint64_t)current.x+previous.y+previous2.z+carry;
                words[i]=sum;carry=sum>>32;previous2=previous;previous=current;
            }chunks[thread]={l,r,carry};
        });
        uint32_t incoming=end.carry;
        for(int c=0;c<32&&chunks[c].r;c++){
            uint32_t propagated=incoming;
            for(size_t j=chunks[c].l;j<chunks[c].r&&propagated;j++){uint64_t sum=(uint64_t)words[j]+propagated;words[j]=sum;propagated=sum>>32;}
            incoming=chunks[c].carry+propagated;
        }
        if(incoming)throw std::runtime_error("carry past cyclic product tail");
        if(first>K)host_arith::copy(out,words+first-K,(L-first)*4);
        if(getenv("GPUPI_DNTT_PROFILE"))fprintf(stderr,"cyclic words=%zu first=%zu tail=%zu seconds=%.6f\n",L,first,wrap,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    }
};
DistributedNTT::DistributedNTT(int n):impl(new Impl(n)){}
DistributedNTT::~DistributedNTT()=default;
void DistributedNTT::multiply(const uint32_t*a,size_t na,const uint32_t*b,size_t nb,uint32_t*out,size_t first){std::lock_guard<std::mutex>lock(impl->window_mutex);impl->run(a,na,b,nb,out,first);}
void DistributedNTT::multiply_window(const uint32_t*a,size_t na,const uint32_t*b,size_t nb,uint32_t*out,size_t first){impl->window(a,na,b,nb,out,first);}
