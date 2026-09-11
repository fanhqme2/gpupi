// Host orchestration with exact, bounded NTT products on independent GPUs.
// Include the existing pipeline to reuse its tuned binary-splitting and
// decimal kernels. The single-GPU executable remains independently usable.
#define main single_gpu_main
#include "pi_bs_gpu.cu"
#undef main
#include <gmpxx.h>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <atomic>
#include "pinned_gmp.h"
#include "distributed_ntt.h"
#include "host_arith.h"
#include "wide_int.h"
#include <array>
#include <unistd.h>

static std::atomic<uint64_t> upload_us{0}, arithmetic_us{0}, download_us{0}, leaf_count{0};

namespace mgpu {
using Z = wide::Z;
using Clock = std::chrono::steady_clock;
static std::array<std::mutex,8> device_mutex;
static void cu(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
static uint64_t setting(const char*name,uint64_t fallback){const char*s=getenv(name);return s?strtoull(s,nullptr,10):fallback;}
static uint64_t system_memory(){long pages=sysconf(_SC_PHYS_PAGES),page=sysconf(_SC_PAGESIZE);return pages>0&&page>0?(uint64_t)pages*page:32ull<<30;}

struct Worker {
    int device;
    BatchMPContext *ctx=nullptr;
    uint32_t *a=nullptr,*b=nullptr,*c=nullptr;
    size_t ca=0,cb=0,cc=0;
    ConvertDecimalPowerTable decimal_table{};
    explicit Worker(int d):device(d){cu(cudaSetDevice(d));ctx=batch_mp_init();if(!ctx)throw std::runtime_error("GPU context initialization failed");}
    ~Worker(){convert_decimal_release_powers(&decimal_table);cudaFree(a);cudaFree(b);cudaFree(c);batch_mp_destroy(ctx);}
    void reserve(uint32_t*&p,size_t&cap,size_t n){if(n>cap){cu(cudaFree(p));p=nullptr;cap=0;cu(cudaMalloc(&p,n));cap=n;}}
    void reset(){convert_decimal_release_powers(&decimal_table);cu(cudaFree(a));cu(cudaFree(b));cu(cudaFree(c));a=b=c=nullptr;ca=cb=cc=0;batch_mp_destroy(ctx);ctx=batch_mp_init();if(!ctx)throw std::runtime_error("GPU context reset failed");}
};
class Pool {
    std::mutex mutex;
    std::condition_variable ready;
    std::queue<std::function<void(Worker&)>> queue;
    bool stop=false;
    std::vector<std::thread> threads;
public:
    int count;
    std::vector<size_t> free_memory;
    std::shared_future<void> reservation;
    explicit Pool(int n):count(n),free_memory(n){
        std::vector<std::future<void>> starts;
        for(int i=0;i<n;i++){
            auto promise=std::make_shared<std::promise<void>>(); starts.push_back(promise->get_future());
            threads.emplace_back([this,i,promise]{
                try {
                    Worker worker(i);size_t total;cu(cudaMemGetInfo(&free_memory[i],&total));promise->set_value();
                    for(;;){std::function<void(Worker&)> job;{
                        std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return stop||!queue.empty();});
                        if(stop&&queue.empty())break;job=std::move(queue.front());queue.pop();
                    }std::lock_guard<std::mutex>device_lock(device_mutex[i]);job(worker);}
                }catch(...){try{promise->set_exception(std::current_exception());}catch(...){std::terminate();}}
            });
        }
        try{for(auto&f:starts)f.get();}catch(...){shutdown();throw;}
    }
    void shutdown(){{std::lock_guard<std::mutex> lock(mutex);stop=true;}ready.notify_all();for(auto&t:threads)if(t.joinable())t.join();}
    ~Pool(){shutdown();if(reservation.valid())reservation.wait();}
    template<class Fn> auto submit(Fn fn)->std::future<decltype(fn(std::declval<Worker&>()))>{
        using R=decltype(fn(std::declval<Worker&>()));
        auto task=std::make_shared<std::packaged_task<R(Worker&)>>(std::move(fn));auto f=task->get_future();
        {std::lock_guard<std::mutex> lock(mutex);queue.emplace([task](Worker&w){(*task)(w);});}ready.notify_one();return f;
    }
};

// Read-only GMP views avoid copying the low and high halves at each split.
struct View {
    const mp_limb_t *p=nullptr; size_t n=0;
    View()=default;
    View(const Z&z):p(mpz_limbs_read(z.get_mpz_t())),n(mpz_size(z.get_mpz_t())){}
    View(const mp_limb_t*q,size_t m):p(q),n(m){while(n&&p[n-1]==0)--n;}
    View low(size_t k)const{return View(p,std::min(n,k));}
    View high(size_t k)const{return n>k?View(p+k,n-k):View();}
    wide::Rep raw()const{return {0,(int64_t)n,const_cast<mp_limb_t*>(p)};}
    bool same(View b)const{return p==b.p&&n==b.n;}
};
static Z plus(View a,View b){Z c;auto x=a.raw(),y=b.raw();mpz_add(c.get_mpz_t(),&x,&y);return c;}
// Extract a bit interval directly, without copying and shifting a full input.
static Z bit_slice(View v,uint64_t start,uint64_t count){
    Z z;if(!count||start/64>=v.n)return z;
    count=std::min<uint64_t>(count,v.n*64-start);
    size_t n=(count+63)/64,first=start/64;unsigned shift=start%64;
    auto*out=mpz_limbs_write(z.get_mpz_t(),n);
    host_arith::parallel(n,[&](size_t l,size_t r,size_t){for(size_t j=l;j<r;j++){
        size_t i=first+j;out[j]=shift?(v.p[i]>>shift)|((i+1<v.n?v.p[i+1]:0)<<(64-shift)):v.p[i];
    }});
    if(count%64)out[n-1]&=(mp_limb_t(1)<<(count%64))-1;
    mpz_limbs_finish(z.get_mpz_t(),n);return z;
}

struct Product {
    size_t split=0;
    uint64_t pre_drop=0;
    Z sa,sb;
    std::future<Z> leaf;
    std::unique_ptr<Product> lo,hi,mid;
    ~Product(){if(leaf.valid())leaf.wait();}
    Z collect_high(uint64_t drop){
        if(pre_drop){if(drop<pre_drop)throw std::runtime_error("product window cannot recover discarded bits");Z z=leaf.get();host_arith::shift(z,-int64_t(drop-pre_drop));return z;}
        if(!lo || drop<split*64){Z z=collect();host_arith::shift(z,-int64_t(drop));return z;}
        Z l=lo->collect(),h=hi->collect(),m=mid->collect();
        sa=Z();sb=Z();
        host_arith::add(m,l,true);host_arith::add(m,h,true);
        host_arith::shift(l,-int64_t(split*64));host_arith::add(m,l);l=Z();
        host_arith::shift(h,split*64);host_arith::add(m,h);h=Z();
        host_arith::shift(m,-int64_t(drop-split*64));return m;
    }
    Z collect(){
        if(!lo)return leaf.get();
        Z l=lo->collect(),h=hi->collect(),m=mid->collect();
        sa=Z();sb=Z();
        host_arith::add(m,l,true);
        host_arith::add(m,h,true);
        // Place disjoint low/high products directly, then add the middle
        // diagonal. Avoid materializing two enormous shifted temporaries.
        size_t ln=mpz_size(l.get_mpz_t()),hn=mpz_size(h.get_mpz_t()),mn=mpz_size(m.get_mpz_t());
        size_t n=std::max({ln,2*split+hn,split+mn})+1;
        auto*out=mpz_limbs_modify(l.get_mpz_t(),n);
        host_arith::parallel(n-ln,[&](size_t a,size_t b,size_t){std::memset(out+ln+a,0,(b-a)*8);});
        host_arith::copy(out+2*split,mpz_limbs_read(h.get_mpz_t()),hn*8);
        if(mn)host_arith::add_at(out+split,n-split,mpz_limbs_read(m.get_mpz_t()),mn);
        mpz_limbs_finish(l.get_mpz_t(),n);
        return l;
    }
};
class Multiply {
public:
    Pool &pool;
    size_t grain;
    bool distributed_enabled;
    bool serial_products=setting("GPUPI_MGPU_SERIAL_PRODUCTS",0);
    std::unique_ptr<DistributedNTT> distributed;
    std::once_flag distributed_init;
    size_t distributed_min=setting("GPUPI_DISTRIBUTED_MIN",(1ull<<31)+1);
    explicit Multiply(Pool&p):pool(p){
        uint64_t free=*std::min_element(p.free_memory.begin(),p.free_memory.end());
        uint64_t scratch=(36ull<<32)/p.count;
        distributed_enabled=setting("GPUPI_DISTRIBUTED",p.count>1&&(p.count&(p.count-1))==0&&free>scratch+(4ull<<30));
        uint64_t margin=(2ull<<30)+(distributed_enabled?scratch:0),budget=free>margin?free-margin:0;
        grain=1ull<<29;while(grain>1024&&grain*144>budget)grain/=2;
        grain=setting("GPUPI_MGPU_GRAIN",grain);
        if(grain<1024||grain>(1ull<<29))throw std::runtime_error("GPUPI_MGPU_GRAIN must be 1024..2^29 (64-bit limbs)");
    }
    std::unique_ptr<Product> plan(View a,View b,uint64_t drop=0){
        if(a.n<b.n)std::swap(a,b);
        auto node=std::make_unique<Product>();
        size_t words=(a.n+b.n)*2,K=1;while(K*2<words)K*=2;
        if(drop&&drop>=words*32){node->pre_drop=drop;node->leaf=pool.submit([](Worker&){return Z();});return node;}
        bool cyclic=setting("GPUPI_DNTT_CYCLIC",1)&&K>=4096&&words-K>1&&words-K<=K/4&&a.n*2<=K&&b.n*2<=K;
        if(distributed_enabled && (pool.count&(pool.count-1))==0 && pool.count<=8 &&
           a.n && b.n && words>=distributed_min && (words<=(1ull<<32)-2||(cyclic&&K<=(1ull<<32)))){
            node->pre_drop=drop;
            node->leaf=std::async(std::launch::async,[this,a,b,cyclic,drop]{
                std::vector<std::unique_lock<std::mutex>>locks;
                for(int i=0;i<pool.count;i++)locks.emplace_back(device_mutex[i]);
                std::call_once(distributed_init,[&]{distributed=std::make_unique<DistributedNTT>(pool.count);});
                size_t first=drop/32,n=((a.n+b.n)*2-first+1)/2;
                Z z;auto*out=mpz_limbs_write(z.get_mpz_t(),n);if(n)out[n-1]=0;
                if(cyclic)distributed->multiply_window((const uint32_t*)a.p,a.n*2,(const uint32_t*)b.p,b.n*2,(uint32_t*)out,first);
                else distributed->multiply((const uint32_t*)a.p,a.n*2,(const uint32_t*)b.p,b.n*2,(uint32_t*)out,first);
                mpz_limbs_finish(z.get_mpz_t(),n);host_arith::shift(z,-int64_t(drop%32));return z;
            });return node;
        }
        if(a.n+b.n<=2*grain || !a.n || !b.n){
            node->pre_drop=drop;
            node->leaf=pool.submit([a,b,drop](Worker&w){
                Z c;
                if(!a.n||!b.n)return c;
                if(std::min(a.n,b.n)<1024){auto x=a.raw(),y=b.raw();mpz_mul(c.get_mpz_t(),&x,&y);host_arith::shift(c,-int64_t(drop));return c;}
                const size_t na=a.n*2,nb=b.n*2,nc=na+nb;
                if(nc>(1ull<<31))throw std::runtime_error("NTT leaf exceeds 2^31 limbs");
                bool square=a.same(b);
                w.reserve(w.a,w.ca,na*4);if(!square)w.reserve(w.b,w.cb,nb*4);w.reserve(w.c,w.cc,nc*4);
                auto t0=Clock::now();
                cu(cudaMemcpy(w.a,a.p,na*4,cudaMemcpyHostToDevice));
                if(!square)cu(cudaMemcpy(w.b,b.p,nb*4,cudaMemcpyHostToDevice));
                BatchMPArray aa{w.a,(uint32_t)na,1,(uint32_t)na};
                BatchMPArray bb{square?w.a:w.b,(uint32_t)nb,1,(uint32_t)nb};
                BatchMPArray out{w.c,(uint32_t)nc,1,(uint32_t)nc};
                auto t1=Clock::now();
                cu(batch_mp_mul(w.ctx,aa,bb,out));
                bool profiling=getenv("GPUPI_MGPU_PROFILE");if(profiling)cu(cudaDeviceSynchronize());
                auto t2=Clock::now();
                size_t first=drop/32,n=(nc-first+1)/2;
                mp_limb_t*dst=mpz_limbs_write(c.get_mpz_t(),n);if(n)dst[n-1]=0;
                cu(cudaMemcpy(dst,w.c+first,(nc-first)*4,cudaMemcpyDeviceToHost));
                if(profiling){
                    upload_us+=(uint64_t)(std::chrono::duration<double>(t1-t0).count()*1e6);
                    arithmetic_us+=(uint64_t)(std::chrono::duration<double>(t2-t1).count()*1e6);
                    download_us+=(uint64_t)(seconds(t2)*1e6);++leaf_count;
                }
                mpz_limbs_finish(c.get_mpz_t(),n);host_arith::shift(c,-int64_t(drop%32));
                return c;
            });return node;
        }
        size_t k=(std::max(a.n,b.n)+1)/2;node->split=k;
        node->sa=plus(a.low(k),a.high(k));
        bool square=a.same(b);
        if(!square)node->sb=plus(b.low(k),b.high(k));
        node->lo=plan(a.low(k),b.low(k));node->hi=plan(a.high(k),b.high(k));
        node->mid=plan(View(node->sa),View(square?node->sa:node->sb));
        return node;
    }
    Z operator()(const Z&a,const Z&b){auto p=plan(View(a),View(b));return p->collect();}
};

static Z download(BatchMPArray a){
    Z z;size_t n=((size_t)a.length+1)/2;
    auto*p=mpz_limbs_write(z.get_mpz_t(),n);if(n)p[n-1]=0;
    cu(cudaMemcpy(p,a.data,(size_t)a.length*4,cudaMemcpyDeviceToHost));
    mpz_limbs_finish(z.get_mpz_t(),n);return z;
}
// Mantissa precision is shared across a computation, with 512 guard bits.
// Exponents preserve the implicit powers of two in the Chudnovsky triples.
// Every positive operation rounds down by less than one mantissa ulp.
struct F {
    Z m; int64_t e=0;size_t skip=0;
    static uint64_t precision;
    F()=default;
    F(Z v,int64_t exponent=0):m(std::move(v)),e(exponent){trim();}
    View value()const{return View(mpz_limbs_read(m.get_mpz_t())+skip,mpz_size(m.get_mpz_t())-skip);}
    uint64_t bits()const{return m==0?0:mpz_sizeinbase(m.get_mpz_t(),2)-skip*64;}
    Z materialize()const{Z z;auto v=value();auto*p=mpz_limbs_write(z.get_mpz_t(),v.n);host_arith::copy(p,v.p,v.n*8);mpz_limbs_finish(z.get_mpz_t(),(mp_size_t)v.n*mpz_sgn(m.get_mpz_t()));return z;}
    void trim(){if(m==0){e=0;skip=0;return;}uint64_t b=bits();if(b>precision){size_t words=(b-precision+63)/64;skip+=words;e+=words*64;}}
    void materialize_inplace(){if(skip){m=materialize();skip=0;}}
};
uint64_t F::precision=4096;
struct FProduct {
    std::unique_ptr<Product> plan;int64_t e=0;uint64_t drop=0;
    F collect(){return plan?F(plan->collect_high(drop),e):F();}
};
static FProduct fplan(Multiply&mul,const F&a,const F&b){
    if(a.m==0||b.m==0)return FProduct();
    uint64_t bits=a.bits()+b.bits();
    uint64_t drop=bits>F::precision?((bits-F::precision+63)&~63ull):0;
    auto plan=mul.plan(a.value(),b.value(),drop);
    return {std::move(plan),a.e+b.e+(int64_t)drop,drop};
}
static F fadd(F a,F b){
    if(a.m==0)return b;if(b.m==0)return a;
    int64_t top=std::max(a.e+(int64_t)a.bits(),b.e+(int64_t)b.bits());
    int64_t exponent=std::max(std::min(a.e,b.e),top-(int64_t)F::precision);
    auto align=[&](F&v){
        v.materialize_inplace();
        host_arith::shift(v.m,v.e-exponent);
    };
    align(a);align(b);host_arith::add(a.m,b.m);a.e=exponent;a.trim();return a;
}
struct Tree {F p,q,r;uint64_t blocks=0;};
static Tree subtree(Worker&w,uint64_t offset,uint32_t blocks,bool need_p,const std::shared_future<void>&reservation={}){
    BatchMPArray p{},q{},r{}; Tree t;t.blocks=blocks;
    try{cu(binary_split_batched(w.ctx,blocks,q,r,nullptr,offset,need_p?&p:nullptr));
        // Let the first GPU wave run while the host arena is initialized.
        // Wait before any GMP allocation or download can use that arena.
        if(reservation.valid())reservation.get();
        if(need_p)t.p=F(download(p),-int64_t(blocks)*17*15);t.q=F(download(q));t.r=F(download(r),-int64_t(blocks)*17*15);
    }catch(...){batch_mp_array_release(p);batch_mp_array_release(q);batch_mp_array_release(r);throw;}
    batch_mp_array_release(p);batch_mp_array_release(q);batch_mp_array_release(r);
    return t;
}
static Tree merge(Multiply&mul,Tree a,Tree b,bool need_p){
    Tree t;t.blocks=a.blocks+b.blocks;
    if(mul.serial_products){
        // One distributed product can already occupy all GPUs. Collect it
        // before planning the next to bound retained Karatsuba temporaries.
        t.q=fplan(mul,a.q,b.q).collect();
        F rq=fplan(mul,a.r,b.q).collect();
        F pr=fplan(mul,a.p,b.r).collect();
        if(a.blocks&1)pr.m.negate();
        t.r=fadd(std::move(rq),std::move(pr));
        if(need_p)t.p=fplan(mul,a.p,b.p).collect();
        return t;
    }
    auto qp=fplan(mul,a.q,b.q),rqp=fplan(mul,a.r,b.q),prp=fplan(mul,a.p,b.r);
    auto pp=need_p?fplan(mul,a.p,b.p):FProduct();
    t.q=qp.collect();
    F rq=rqp.collect(),pr=prp.collect();
    if(a.blocks&1)pr.m.negate();
    t.r=fadd(std::move(rq),std::move(pr));
    if(need_p)t.p=pp.collect();
    return t;
}

static Tree binary_split(Pool&pool,Multiply&mul,uint64_t digits){
    auto leaf_start=Clock::now();
    uint64_t blocks=(uint64_t)ceil(((double)digits+64)/14.18/17);
    // Multiples of eight preserve the guaranteed 136-term cancellations.
    blocks=(blocks+7)&~7ull;
    size_t free=*std::min_element(pool.free_memory.begin(),pool.free_memory.end());
    uint32_t cap=setting("GPUPI_MGPU_TREE_BLOCKS",free>(80ull<<30)?1u<<23:free>(40ull<<30)?1u<<22:1u<<21);
    if(cap<8||(cap&(cap-1))||cap>(1u<<23))throw std::runtime_error("tree block cap must be a power of two in 8..2^23");
    if(setting("GPUPI_MGPU_ROUND_TREES",1) && blocks>=(uint64_t)cap*4)blocks=(blocks+cap-1)/cap*cap;
    std::vector<std::future<Tree>> pending;
    uint64_t offset=0;
    for(uint64_t left=blocks;left;){
        uint32_t n=cap;while(n>left)n/=2;
        int batch=1;
        if(setting("GPUPI_MGPU_BALANCE_TREES",digits>=15000000000ull&&pool.count==8)&&blocks>=4ull*cap&&left>=8ull*pool.count){
            while((uint64_t)n*pool.count>left)n/=2;batch=pool.count;
        }
        for(int i=0;i<batch;i++){
            bool need_p=left!=n;
            bool reclaim=pool.count<8&&mul.distributed_enabled;
            pending.push_back(pool.submit([offset,n,need_p,reclaim,reservation=pool.reservation](Worker&w){auto tree=subtree(w,offset,n,need_p,reservation);if(reclaim)w.reset();return tree;}));
            offset+=n;left-=n;
        }
    }
    std::vector<Tree> nodes;for(auto&f:pending)nodes.push_back(f.get());
    fprintf(stderr,"binary_split leaves=%zu terms=%llu leaves_s=%.6f\n",nodes.size(),(unsigned long long)blocks*17,seconds(leaf_start));
    while(nodes.size()>1){
        auto merge_start=Clock::now();
        std::vector<std::future<Tree>> merges;
        // Large transforms already occupy every GPU. Serialize their parent
        // merges at high precision to reduce retained products and allocation.
        bool serial=nodes.size()<=setting("GPUPI_MGPU_SERIAL_MERGES",digits>=75000000000ull&&pool.count==8?32:0);
        for(size_t i=0;i<nodes.size();i+=2){
            merges.push_back(std::async(serial?std::launch::deferred:std::launch::async,[&,i]{
                if(i+1==nodes.size())return std::move(nodes[i]);
                return merge(mul,std::move(nodes[i]),std::move(nodes[i+1]),i+2<nodes.size());
            }));
        }
        std::vector<Tree> next;for(auto&f:merges)next.push_back(f.get());nodes=std::move(next);
        if(getenv("GPUPI_MGPU_PROFILE"))fprintf(stderr,"merge nodes=%zu seconds=%.6f\n",nodes.size(),seconds(merge_start));
    }return std::move(nodes[0]);
}

// The same exact reciprocal refinement used by the original device pipeline.
static Z inverse(Multiply&mul,const Z&x,uint64_t L){
    std::vector<uint64_t> levels{L};
    while(levels.back()>32768){uint64_t m=((levels.back()-2)/4)&~31ull;levels.push_back(levels.back()-2*m);}
    auto chunk=[&](uint64_t l){uint64_t start=(L-l)/2;return bit_slice(View(x),start,mpz_size(x.get_mpz_t())*64-start);};
    uint64_t l0=levels.back();Z total=1,y,res,x0=chunk(l0);total<<=l0;
    mpz_fdiv_qr(y.get_mpz_t(),res.get_mpz_t(),total.get_mpz_t(),x0.get_mpz_t());
    for(size_t i=levels.size()-1;i>0;--i){
        uint64_t l2=levels[i-1],m=(l2-levels[i])/2;
        Z xn=chunk(l2),x2=bit_slice(View(xn),0,m);
        Z xy=mul(x2,y);host_arith::shift(res,m);host_arith::add(res,xy,true);
        bool neg=res<0;if(neg)res.negate();
        Z y2=mul(res,y);host_arith::shift(y2,-int64_t(l2-2*m));if(neg)++y2;
        if(l2!=L){Z prod=mul(y2,xn);host_arith::shift(res,m);if(neg){host_arith::add(prod,res,true);res=std::move(prod);}else host_arith::add(res,prod,true);}
        host_arith::shift(y,m);host_arith::add(y,y2,neg);
    }return y;
}
static std::pair<Z,Z> pell(Multiply&mul,uint64_t digits){
    Z p=4001,q=40;
    for(double valid=7.5;valid<digits;valid*=2){
        auto pq=mul.plan(View(p),View(q));auto pp=mul.plan(View(p),View(p));
        Z nq=pq->collect(),np=pp->collect();host_arith::shift(nq,1);host_arith::shift(np,1);--np;
        p=std::move(np);q=std::move(nq);
    }return {std::move(p),std::move(q)};
}
static Z fraction(Pool&pool,Multiply&mul,uint64_t digits,uint64_t&bits){
    auto begin=Clock::now();Tree t=binary_split(pool,mul,digits);
    fprintf(stderr,"binary_split_s=%.6f\n",seconds(begin));
    uint64_t prec=F::precision;
    begin=Clock::now();auto sq=pell(mul,digits+128);
    fprintf(stderr,"sqrt_s=%.6f\n",seconds(begin));begin=Clock::now();
    F sp(std::move(sq.first)),sqf(std::move(sq.second));
    auto pp=fplan(mul,sp,t.q),qp=fplan(mul,sqf,t.r);
    F p=pp.collect(),q=qp.collect();sp={};sqf={};
    t={};
    uint64_t L=2*q.bits()-1;
    size_t saved_threshold=mul.distributed_min;
    mul.distributed_min=setting("GPUPI_SERIAL_DISTRIBUTED_MIN",1ull<<28);
    q.materialize_inplace();Z inv=inverse(mul,q.m,L);
    bits=prec;
    int64_t drop=int64_t(L)-int64_t(bits)-p.e+q.e+8;
    auto plan=mul.plan(p.value(),View(inv),drop>=0?drop:0);
    Z out;
    if(drop>=0)out=plan->collect_high(drop);else{out=plan->collect();out<<=-drop;}
    mul.distributed_min=saved_threshold;
    fprintf(stderr,"division_s=%.6f\n",seconds(begin));return out;
}

struct DecimalNode {Z value;uint64_t bits,digits,offset;};
struct DecimalOutput {
    struct Deleter {void(*free)(void*,size_t)=nullptr;size_t bytes=0;void operator()(char*p)const{if(p)free(p,bytes);}};
    std::unique_ptr<char,Deleter> buffer{nullptr,Deleter{}};size_t length=0;
    DecimalOutput()=default;
    explicit DecimalOutput(size_t n):length(n){
        void*(*allocate)(size_t);void(*release)(void*,size_t);mp_get_memory_functions(&allocate,nullptr,&release);
        buffer=std::unique_ptr<char,Deleter>((char*)allocate(n+1),Deleter{release,n+1});buffer.get()[n]=0;
    }
    size_t size()const{return length;}
    const char*data()const{return buffer?buffer.get():"";}
    const char*c_str()const{return data();}
    void resize(size_t n){length=n;if(buffer)buffer.get()[n]=0;}
};
static DecimalOutput decimal(Pool&pool,Multiply&mul,Z x,uint64_t bits,uint64_t digits){
    auto phase=Clock::now();
    uint64_t total=round_up_decimal_digits(digits);uint32_t leaf=kDecimalLeafDigits;
    uint64_t cap=setting("GPUPI_MGPU_DECIMAL_DIGITS",400000000);
    cap=std::max<uint64_t>(cap,leaf);
    uint32_t split=0;while((total>>split)>cap || (1u<<split)<(unsigned)pool.count)++split;
    uint32_t all=__builtin_ctzll(total/leaf);split=std::min(split,all);
    std::vector<Z> powers;
    size_t saved_threshold=mul.distributed_min;
    mul.distributed_min=setting("GPUPI_SERIAL_DISTRIBUTED_MIN",1ull<<28);
    if(split){Z z;mpz_ui_pow_ui(z.get_mpz_t(),5,leaf);powers.push_back(std::move(z));
        for(uint32_t i=1;i<all;i++)powers.push_back(mul(powers.back(),powers.back()));}
    mul.distributed_min=saved_threshold;
    if(getenv("GPUPI_MGPU_PROFILE"))fprintf(stderr,"decimal powers_s=%.6f\n",seconds(phase));phase=Clock::now();
    std::vector<DecimalNode> nodes;nodes.push_back({std::move(x),bits,total,0});
    for(uint32_t level=0;level<split;level++){
        mul.distributed_min=nodes.size()==1?setting("GPUPI_SERIAL_DISTRIBUTED_MIN",1ull<<28):saved_threshold;
        std::vector<std::unique_ptr<Product>> products;
        for(auto&node:nodes){
            uint64_t mb=node.bits-node.digits/2;
            // Unmasked bits above mb disappear in the final reduction modulo
            // 2^cb. This makes the low operand a read-only limb view.
            products.push_back(mul.plan(View(node.value).low((mb+63)/64),View(powers[all-1-level])));
        }
        std::vector<DecimalNode> next;
        for(size_t i=0;i<nodes.size();i++){
            auto&node=nodes[i];
            uint64_t d=node.digits/2,cb=(uint64_t)ceil((double)d*3.32192809488736234787)+34,mb=node.bits-d;
            Z product=products[i]->collect();
            Z left=bit_slice(View(node.value),node.bits-cb,cb);
            Z right=bit_slice(View(product),mb-cb,cb);
            next.push_back({std::move(left),cb,d,node.offset});
            next.push_back({std::move(right),cb,d,node.offset+d});
        }nodes=std::move(next);
    }
    mul.distributed_min=saved_threshold;
    powers.clear();
    if(getenv("GPUPI_MGPU_PROFILE"))fprintf(stderr,"decimal split_s=%.6f\n",seconds(phase));phase=Clock::now();
    DecimalOutput output(total);std::vector<std::future<void>> done;
    for(auto&node:nodes)done.push_back(pool.submit([&node,&output,leaf,all,split](Worker&w){
        size_t n=(node.bits+31)/32;w.reserve(w.a,w.ca,n*4);w.reserve(w.c,w.cc,node.digits);
        BatchMPArray input{w.a,(uint32_t)n,1,(uint32_t)n};char*result=(char*)w.c;
        cu(cudaMemset(input.data,0,n*4));cu(cudaMemcpy(input.data,mpz_limbs_read(node.value.get_mpz_t()),std::min(n*4,mpz_size(node.value.get_mpz_t())*8),cudaMemcpyHostToDevice));
        if(!w.decimal_table.powers||w.decimal_table.leaf_digits!=leaf||w.decimal_table.levels!=all-split){
            convert_decimal_release_powers(&w.decimal_table);cu(convert_decimal_precompute_powers(w.ctx,&w.decimal_table,leaf,all-split));
        }
        cu(convert_decimal_equal_split(w.ctx,input,node.bits,node.digits,leaf,&w.decimal_table,result));
        cu(cudaMemcpy(output.buffer.get()+node.offset,result,node.digits,cudaMemcpyDeviceToHost));
    }));
    try{for(auto&f:done)f.get();}catch(...){for(auto&f:done)if(f.valid())f.wait();throw;}
    if(getenv("GPUPI_MGPU_PROFILE"))fprintf(stderr,"decimal leaves_s=%.6f\n",seconds(phase));
    output.resize(digits);return output;
}

static void test(Pool&pool,Multiply&mul){
    pool.submit([](Worker&w){
        for(uint64_t offset:{0ull,42000000ull,90000000ull,105000000ull,205000000ull,252645120ull,252645135ull,252645136ull,419430368ull,500000000ull})for(bool head:{false,true}){
            BatchMPArray p=batch_mp_array_create(16,4),q=batch_mp_array_create(16,5),r=batch_mp_array_create(16,6);
            if(head)construct_leaf_17<true,true,true><<<1,32>>>(p.data,q.data,r.data,16,offset);
            else construct_leaf_17<false,true,true><<<1,32>>>(p.data,q.data,r.data,16,offset);
            cu(cudaGetLastError());
            std::vector<uint32_t> pp(64),qq(80),rr(96);
            cu(cudaMemcpy(pp.data(),p.data,pp.size()*4,cudaMemcpyDeviceToHost));
            cu(cudaMemcpy(qq.data(),q.data,qq.size()*4,cudaMemcpyDeviceToHost));
            cu(cudaMemcpy(rr.data(),r.data,rr.size()*4,cudaMemcpyDeviceToHost));
            batch_mp_array_release(p);batch_mp_array_release(q);batch_mp_array_release(r);
            for(int j=0;j<16;j++){
                uint64_t i=offset*17+(head?j*17+16:j+(j>>4));
                Z a,b,c;mpz_import(a.get_mpz_t(),4,-1,4,0,0,pp.data()+j*4);
                mpz_import(b.get_mpz_t(),5,-1,4,0,0,qq.data()+j*5);
                mpz_import(c.get_mpz_t(),6,-1,4,0,0,rr.data()+j*6);
                Z ap=Z(6*i+1)*Z(2*i+1)*Z(6*i+5),bq=Z(i+1)*Z(i+1)*Z(i+1)*10005*10005*3335;
                if(a*bq!=b*ap || c*3335!=b*Z((unsigned long)(545140134ull*i+13591409)))throw std::runtime_error("high-index leaf mismatch");
            }
        }
    }).get();
    pool.submit([](Worker&w){
        BatchMPArray q{},r{};
        for(uint64_t offset:std::array<uint64_t,2>{(1ull<<33)/17,UINT64_MAX})
            if(binary_split_batched(w.ctx,32,q,r,nullptr,offset)!=cudaErrorInvalidValue)throw std::runtime_error("leaf range guard mismatch");
    }).get();
    uint64_t old_precision=F::precision;F::precision=1048576;
    struct ExactTree {Z p,q,r;uint32_t terms;};
    std::function<ExactTree(uint64_t,uint32_t)> reference=[&](uint64_t i,uint32_t n)->ExactTree{
        if(n==1){Z base=Z(i+1)*Z(i+1)*Z(i+1)*10005*10005;
            return {Z(6*i+1)*Z(2*i+1)*Z(6*i+5),base*3335,base*Z((unsigned long)(545140134ull*i+13591409)),1};}
        auto a=reference(i,n/2),b=reference(i+n/2,n-n/2);
        Z r=a.r*b.q;r<<=15ull*b.terms;
        Z pr=a.p*b.r;if(a.terms&1)r-=pr;else r+=pr;
        return {a.p*b.p,a.q*b.q,std::move(r),n};
    };
    for(uint64_t offset:{0ull,20000000ull,40000000ull,80000000ull,105000000ull,205000000ull,252645120ull,252645136ull,419430368ull,500000000ull}){
        auto got=pool.submit([offset](Worker&w){return subtree(w,offset,32,true);}).get();
        auto ref=reference((uint64_t)offset*17,32*17);
        auto ratio=[&](const F&a,const Z&b){
            Z lhs=a.materialize()*ref.q,rhs=b*got.q.materialize();
            int64_t shift=a.e-got.q.e+(int64_t)ref.terms*15;
            if(shift>=0)lhs<<=shift;else rhs<<=-shift;
            return lhs==rhs;
        };
        if(!ratio(got.p,ref.p)||!ratio(got.r,ref.r))throw std::runtime_error("offset subtree mismatch");
    }
    F::precision=old_precision;
    gmp_randclass random(gmp_randinit_default);random.seed(717);
    for(int i=0;i<100;i++){
        Z a=random.get_z_bits(4096+i*619),b=random.get_z_bits(4096+i*517);
        Z slice=bit_slice(View(a),i*97,131+i*311),slice_ref=(a>>(i*97))&((Z(1)<<(131+i*311))-1);
        if(slice!=slice_ref)throw std::runtime_error("bit slice mismatch");
        if(i&1)a=-a;if(i&2)b=-b;
        Z x=a,ref=a+b;host_arith::add(x,b);if(x!=ref)throw std::runtime_error("host add mismatch");
        x=a;ref=a-b;host_arith::add(x,b,true);if(x!=ref)throw std::runtime_error("host sub mismatch");
        int64_t shift=i*131-4096;x=a;host_arith::shift(x,shift);ref=a;if(shift>=0)ref<<=shift;else ref>>=-shift;
        if(x!=ref)throw std::runtime_error("host shift mismatch");
        a=(Z(1)<<(4096+i*131))-1;b=1;x=a;host_arith::add(x,b);if(x!=a+b)throw std::runtime_error("host carry chain mismatch");
        x=b;host_arith::add(x,a,true);if(x!=b-a)throw std::runtime_error("host borrow chain mismatch");
    }
    size_t saved=mul.grain;mul.grain=4096;
    for(int i=0;i<35;i++){
        size_t a=i<25?1000+i*21111:((size_t)1<<(i-10));size_t b=(i%3)?a:a/3;
        Z x=random.get_z_bits(a),y=random.get_z_bits(b);if(i%5==0){x=(Z(1)<<a)-1;y=(Z(1)<<b)-1;}
        Z ref=x*y,got=mul(x,y);auto hp=mul.plan(mgpu::View(x),mgpu::View(y));if(hp->collect_high(a/2)!=(ref>>(a/2)))throw std::runtime_error("high product mismatch");if(ref!=got)throw std::runtime_error("multi-GPU multiplication mismatch");
        auto window=mul.plan(View(x),View(y),a/2);if(window->collect_high(a/2)!=(ref>>(a/2)))throw std::runtime_error("pre-truncated product mismatch");
        got=mul(x,x);ref=x*x;if(ref!=got)throw std::runtime_error("multi-GPU square mismatch");
    }
    mul.grain=saved;
    for(int i=0;i<10;i++){Z x=random.get_z_bits(10000+i*10000);mpz_setbit(x.get_mpz_t(),9999+i*10000);uint64_t L=mpz_sizeinbase(x.get_mpz_t(),2)*2-1;Z y=inverse(mul,x,L),r=(Z(1)<<L)/x;if(abs(y-r)>1)throw std::runtime_error("reciprocal mismatch");}
    fprintf(stderr,"PASS: multi-GPU products, squares, carries, unequal operands, recursive reciprocal\n");
}
static void bench(Multiply&mul,int k,int reps){
    if(k<12||k>33||reps<1)throw std::runtime_error("bench exponent must be 12..33, repeats positive");
    if(setting("GPUPI_MGPU_PINNED",1)&&setting("GPUPI_MGPU_ARENA",1))pinned_gmp::reserve(setting("GPUPI_BENCH_RESERVE",0));
    size_t n=setting("GPUPI_BENCH_WORDS",1ull<<k);n=(n+1)&~size_t(1);Z a,b;
    auto*x=mpz_limbs_write(a.get_mpz_t(),n/2),*y=mpz_limbs_write(b.get_mpz_t(),n/2);
    host_arith::parallel(n/2,[&](size_t l,size_t r,size_t){for(size_t i=l;i<r;i++){x[i]=0xfedcba9876543211ull+i*0x9e3779b97f4a7c15ull;y[i]=0x123456789abcdef1ull+i*0xbf58476d1ce4e5b9ull;}});
    mpz_limbs_finish(a.get_mpz_t(),n/2);mpz_limbs_finish(b.get_mpz_t(),n/2);
    for(int r=0;r<reps;r++){
        auto t=Clock::now();Z c=mul(a,b);double elapsed=seconds(t);
        for(unsigned long p:{1000000007ul,1000000009ul,2147483647ul}){
            unsigned long expected=(uint64_t)mpz_fdiv_ui(a.get_mpz_t(),p)*mpz_fdiv_ui(b.get_mpz_t(),p)%p;
            if(mpz_fdiv_ui(c.get_mpz_t(),p)!=expected)throw std::runtime_error("benchmark product checksum mismatch");
        }
        fprintf(stderr,"profile leaves=%llu upload_s=%.6f arithmetic_s=%.6f download_s=%.6f\n",(unsigned long long)leaf_count.exchange(0),upload_us.exchange(0)/1e6,arithmetic_us.exchange(0)/1e6,download_us.exchange(0)/1e6);
        printf("mul k=%d gpus=%d grain=%zu trial=%d seconds=%.6f\n",k,mul.pool.count,mul.grain,r,elapsed);fflush(stdout);
    }
}
} // namespace mgpu

int main(int argc,char**argv){
    try{
        pinned_gmp::Guard pinned_guard;
        if(mgpu::setting("GPUPI_MGPU_PINNED",1))pinned_gmp::install(mgpu::setting("GPUPI_MGPU_PIN_CACHE",std::min<uint64_t>(512ull<<30,mgpu::system_memory()/4)));
        int available=0;mgpu::cu(cudaGetDeviceCount(&available));int n=mgpu::setting("GPUPI_GPUS",std::min(available,8));
        if(n<1||n>available||n>8)throw std::runtime_error("GPUPI_GPUS must be between 1 and min(8, visible GPU count)");
        mgpu::Pool pool(n);mgpu::Multiply mul(pool);
        if(argc==2&&strcmp(argv[1],"--test")==0){mgpu::test(pool,mul);return 0;}
        if(argc>=3&&strcmp(argv[1],"--bench-mul")==0){mgpu::bench(mul,atoi(argv[2]),argc>3?atoi(argv[3]):3);return 0;}
        if(argc<2||argc>3)throw std::runtime_error("Usage: pi_mgpu DIGITS [--benchmark] | --test");
        char*end=nullptr;errno=0;uint64_t digits=strtoull(argv[1],&end,10);
        if(errno||end==argv[1]||*end||digits>100000000000ull)throw std::runtime_error("DIGITS must be between 0 and 100000000000");
        bool benchmark=argc==3&&strcmp(argv[2],"--benchmark")==0;
        if(argc==3&&!benchmark)throw std::runtime_error("Unknown option");
        // Workers are still idle: retain the larger initial arena across stages.
        if(digits>=75000000000ull&&mgpu::setting("GPUPI_MGPU_PINNED",1))pinned_gmp::cache_limit=mgpu::setting("GPUPI_MGPU_PIN_CACHE",std::min<uint64_t>(1024ull<<30,mgpu::system_memory()/2));
        host_topology::automatic=digits>=30000000000ull&&n==8;
        mgpu::F::precision=((uint64_t)ceil((double)digits*3.32192809488736234787)+512+63)/64*64;
        auto t=mgpu::Clock::now();uint64_t bits=0;
        uint64_t reserve=digits>=75000000000ull?640ull<<30:digits>=30000000000ull?320ull<<30:digits>=15000000000ull?(n==8?144ull:224ull)<<30:digits>=5000000000ull?48ull<<30:0;
        if(mgpu::system_memory()<reserve*2)reserve=0;
        reserve=mgpu::setting("GPUPI_MGPU_RESERVE",reserve);
        if(mgpu::setting("GPUPI_MGPU_PINNED",1)&&mgpu::setting("GPUPI_MGPU_ARENA",1)){
            if(reserve&&mgpu::setting("GPUPI_MGPU_ASYNC_RESERVE",1))pool.reservation=std::async(std::launch::async,[reserve]{mgpu::cu(cudaSetDevice(0));pinned_gmp::reserve(reserve);}).share();
            else pinned_gmp::reserve(reserve);
        }
        if(getenv("GPUPI_MGPU_PROFILE"))fprintf(stderr,"reserve_%s_s=%.6f\n",pool.reservation.valid()?"launch":"sync",mgpu::seconds(t));
        auto x=mgpu::fraction(pool,mul,digits,bits);
        auto dt=mgpu::Clock::now();auto output=digits?mgpu::decimal(pool,mul,std::move(x),bits,digits):mgpu::DecimalOutput();
        fprintf(stderr,"decimal_s=%.6f\nelapsed_ms=%.3f digits=%llu gpus=%d\n",mgpu::seconds(dt),mgpu::seconds(t)*1000,(unsigned long long)digits,n);
        if(benchmark)printf("DEC = 3%.*s...%s\n",(int)std::min<size_t>(64,output.size()),output.c_str(),output.c_str()+(output.size()>64?output.size()-64:0));
        else{putchar('3');if(fwrite(output.data(),1,output.size(),stdout)!=output.size())throw std::runtime_error("output write failed");putchar('\n');}
    }catch(const std::exception&e){fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
