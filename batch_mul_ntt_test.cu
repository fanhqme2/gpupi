#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <gmp.h>

#include "batch_mul_ntt.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

static void words_to_mpz(mpz_t out, const uint32_t* words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

static bool run_configuration(
    uint32_t N,
    uint32_t L,
    uint32_t sampled_total_words,
    int case_id,
    NTTPrecomputedTables tables,
    std::mt19937_64& rng
) {
    const size_t in_words = (size_t)N * L;
    const size_t out_words = in_words * 2;
    const size_t ws_bytes = batch_mul_ntt_workspace_size(N, L);

    std::vector<uint32_t> h_A(in_words);
    std::vector<uint32_t> h_B(in_words);
    std::vector<uint32_t> h_out(out_words, 0);

    std::uniform_int_distribution<uint32_t> word_dist(0u, 0xffffffffu);
    for (size_t i = 0; i < in_words; ++i) {
        h_A[i] = word_dist(rng);
        h_B[i] = word_dist(rng);
    }

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    uint32_t* d_out = nullptr;
    uint32_t* d_ws = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, in_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_B, in_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, out_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), in_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), in_words * sizeof(uint32_t), cudaMemcpyHostToDevice));

    batch_mul_ntt(d_A, d_B, d_out, d_ws, tables, N, L);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    size_t bad_idx = 0;

    mpz_t a, b, expected, got;
    mpz_inits(a, b, expected, got, NULL);
    for (size_t i = 0; i < N; ++i) {
        words_to_mpz(a, &h_A[i * L], L);
        words_to_mpz(b, &h_B[i * L], L);
        mpz_mul(expected, a, b);
        words_to_mpz(got, &h_out[i * ((size_t)L * 2)], (size_t)L * 2);
        if (mpz_cmp(expected, got) != 0) {
            ok = false;
            bad_idx = i;
            break;
        }
    }

    if (!ok) {
        printf("FAILED: case=%d sampled(N*L)=%u L=%u N=%u actual(N*L)=%zu idx=%zu\n",
               case_id, sampled_total_words, L, N, in_words, bad_idx);
        gmp_printf("A[%zu] = %Zx\n", bad_idx, a);
        gmp_printf("B[%zu] = %Zx\n", bad_idx, b);
        gmp_printf("Got    = %Zx\n", got);
        gmp_printf("Expect = %Zx\n", expected);
    } else {
        printf("PASSED: case=%d sampled(N*L)=%u L=%u N=%u actual(N*L)=%zu\n",
               case_id, sampled_total_words, L, N, in_words);
    }

    mpz_clears(a, b, expected, got, NULL);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ws));

    return ok;
}

int main() {
    constexpr uint32_t N = 5;
    constexpr uint32_t L = 16373;

    // Keep seed aligned with the quick-finder so this case is reproducible.
    std::mt19937_64 rng(123456789ull);

    uint3* d_lv1 = nullptr;
    uint3* d_lv2 = nullptr;
    uint3* d_inv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lv1, 65536 * sizeof(uint3)));
    CUDA_CHECK(cudaMalloc(&d_lv2, 65536 * sizeof(uint3)));
    CUDA_CHECK(cudaMalloc(&d_inv, 33 * sizeof(uint3)));

    NTTPrecomputedTables tables;
    tables.roots_table_lv1 = d_lv1;
    tables.roots_table_lv2 = d_lv2;
    tables.inv2n_table = d_inv;

    init_ntt_precomputed_tables(&tables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Repro case: N=%u L=%u\n", N, L);
    bool ok = run_configuration(N, L, N * L, 0, tables, rng);

    CUDA_CHECK(cudaFree(d_lv1));
    CUDA_CHECK(cudaFree(d_lv2));
    CUDA_CHECK(cudaFree(d_inv));

    return ok ? 0 : 1;
}
