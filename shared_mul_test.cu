#include "batch_arith.h"
#include <gmp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

static void run(BatchMPContext *ctx, uint32_t n, uint32_t la, uint32_t lc,
                uint32_t lb, bool broadcast, bool maximal) {
    const uint32_t sa=la+3, sc=lc+5, sb=lb+7, so=la+lb+3, sp=lc+lb+3;
    std::mt19937 rng(1234+la+lc+lb);
    std::vector<uint32_t> a((size_t)n*sa), c((size_t)n*sc), b((size_t)(broadcast?1:n)*sb);
    for (auto *v : {&a,&b,&c}) for (auto &x : *v) x=maximal?UINT32_MAX:rng();
    auto A=batch_mp_array_create(n,sa), C=batch_mp_array_create(n,sc);
    auto B=batch_mp_array_create(broadcast?1:n,sb);
    auto O=batch_mp_array_create(n,so), P=batch_mp_array_create(n,sp);
    if(!A.data||!B.data||!C.data||!O.data||!P.data) std::exit(1);
    CHECK(cudaMemcpy(A.data,a.data(),a.size()*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(C.data,c.data(),c.size()*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B.data,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(O.data,0xa5,(size_t)n*so*4)); CHECK(cudaMemset(P.data,0xa5,(size_t)n*sp*4));
    A.length=la; C.length=lc; B.length=lb; B.batch_size=n; if(broadcast)B.stride=0;
    O.length=la+lb; P.length=lc+lb;
    CHECK(batch_mp_mul_shared_b(ctx,A,C,B,O,P)); CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> o((size_t)n*so), p((size_t)n*sp);
    CHECK(cudaMemcpy(o.data(),O.data,o.size()*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p.data(),P.data,p.size()*4,cudaMemcpyDeviceToHost));
    mpz_t x,y,z,actual; mpz_inits(x,y,z,actual,nullptr);
    for(uint32_t row=0;row<n;++row) {
        mpz_import(y,lb,-1,4,0,0,b.data()+(size_t)(broadcast?0:row)*sb);
        for(int which=0;which<2;++which) {
            uint32_t len=which?lc:la, stride=which?sc:sa, out_stride=which?sp:so;
            const auto &in=which?c:a; const auto &out=which?p:o;
            mpz_import(x,len,-1,4,0,0,in.data()+(size_t)row*stride); mpz_mul(z,x,y);
            mpz_import(actual,len+lb,-1,4,0,0,out.data()+(size_t)row*out_stride);
            if(mpz_cmp(z,actual)) { fprintf(stderr,"Mismatch n=%u a=%u c=%u b=%u row=%u output=%d\n",n,la,lc,lb,row,which);std::exit(1); }
            for(uint32_t j=len+lb;j<out_stride;++j)
                if(out[(size_t)row*out_stride+j]!=0xa5a5a5a5u) std::exit(1);
        }
    }
    mpz_clears(x,y,z,actual,nullptr);
    for(auto v:{A,B,C,O,P})batch_mp_array_release(v);
    printf("PASS n=%u a=%u c=%u b=%u broadcast=%d maximal=%d\n",n,la,lc,lb,broadcast,maximal);fflush(stdout);
}
static void run_merge(BatchMPContext *ctx, bool maximal, bool omit_last_p) {
    constexpr uint32_t pairs = 5;
    auto p = batch_mp_array_create(2 * pairs - omit_last_p, 8500, 8507);
    auto q = batch_mp_array_create(2 * pairs, 17300, 17307);
    auto r = batch_mp_array_create(2 * pairs, 21000, 21007);
    auto shifted = batch_mp_array_create(pairs, r.length + 128);
    auto po = batch_mp_array_create(pairs - omit_last_p, p.length * 2);
    auto qo = batch_mp_array_create(pairs, q.length * 2);
    auto ro = batch_mp_array_create(pairs, shifted.length + q.length + 1);
    std::mt19937 rng(153);
    for (auto col : {p, q, r}) {
        std::vector<uint32_t> words((size_t)col.stride * col.batch_size);
        for (auto &w : words) w = maximal ? UINT32_MAX : rng();
        CHECK(cudaMemcpy(col.data, words.data(), words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    auto even_r = r;
    even_r.stride *= 2;
    even_r.batch_size = pairs;
    CHECK(batch_mp_shift_bits(ctx, even_r, shifted, 128 * 32));
    CHECK(batch_mp_split_merge(ctx, p, q, r, shifted, po, qo, ro));
    auto read = [](mpz_t x, BatchMPArray v, uint32_t row) {
        std::vector<uint32_t> h(v.length);
        CHECK(cudaMemcpy(h.data(), v.data + (size_t)row * v.stride,
            h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        mpz_import(x, h.size(), -1, sizeof(uint32_t), 0, 0, h.data());
    };
    mpz_t a, b, c, d, expected, actual;
    mpz_inits(a, b, c, d, expected, actual, nullptr);
    for (uint32_t i = 0; i < pairs; ++i) {
        read(a, p, 2*i);
        read(b, q, 2*i+1);
        if (i < po.batch_size) {
            read(c, p, 2*i+1);
            mpz_mul(expected, a, c);
            read(actual, po, i);
            if (mpz_cmp(expected, actual)) std::exit(1);
        }
        read(c, q, 2*i);
        mpz_mul(expected, b, c);
        read(actual, qo, i);
        if (mpz_cmp(expected, actual)) std::exit(1);
        read(c, r, 2*i);
        read(d, r, 2*i+1);
        mpz_mul(expected, b, c);
        mpz_mul_2exp(expected, expected, 128 * 32);
        mpz_addmul(expected, a, d);
        read(actual, ro, i);
        if (mpz_cmp(expected, actual)) std::exit(1);
    }
    mpz_clears(a, b, c, d, expected, actual, nullptr);
    for (auto col : {p, q, r, shifted, po, qo, ro}) batch_mp_array_release(col);
    printf("PASS split merge maximal=%d omit_last_p=%d\n", maximal, omit_last_p);
}

int main() {
    auto ctx=batch_mp_init();if(!ctx)return 1;
    run_merge(ctx, false, false);
    run_merge(ctx, true, true);
    // Fallbacks, every medium schedule, linear/cyclic combinations, both tails,
    // varying strides, broadcast B, and a workspace batch boundary.
    for(uint32_t k=6;k<=25;++k) {
        uint32_t half=1u<<(k-1);
        run(ctx,3,half-7,half-3,half-1,false,false);
        run(ctx,3,half,half+33,half,false,true);
        if(k<=18)run(ctx,3,half+1,half+31,half,true,false);
    }
    run(ctx,1025,8192,8225,8192,false,false);
    run(ctx,1025,8193,8225,8192,true,true);
    // The shorter transform is the even-frequency prefix of the longer one.
    // Check both output orders and the cache stride across distinct rows.
    run(ctx,5,6000,12500,7500,false,false);
    run(ctx,5,12500,6000,7500,true,true);
    // Wrap lengths near one quarter of K exercise the fused reverse-tail NTT.
    run(ctx, 3, 82211, 84510, 79011, false, true);
    run(ctx, 3, 82211, 84510, 79011, true, false);
    run(ctx,3,8192,32768,8192,false,false); // unequal NTT lengths: fallback
    // Force overlap-add, including unequal last tiles, independent rows,
    // broadcast input and maximal limbs carrying across tile boundaries.
    batch_mp_limit_large_products(ctx, 192u << 10);
    run(ctx, 3, 11210, 21413, 12227, false, false);
    run(ctx, 3, 11210, 21413, 12227, true, true);
    batch_mp_destroy(ctx); puts("All shared multiplication checks passed");
}
