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

static uint32_t floor_log2_u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << (r + 1)) <= x && r < 30) ++r;
    return r;
}

static uint32_t sample_log_uniform_u32(uint32_t lo, uint32_t hi, std::mt19937_64& rng) {
    uint32_t min_bin = floor_log2_u32(lo);
    uint32_t max_bin = floor_log2_u32(hi);
    std::uniform_int_distribution<uint32_t> bin_dist(min_bin, max_bin);
    uint32_t b = bin_dist(rng);
    uint32_t b_lo = std::max(lo, 1u << b);
    uint32_t b_hi = std::min(hi, (b == 31) ? 0xffffffffu : ((1u << (b + 1)) - 1u));
    std::uniform_int_distribution<uint32_t> val_dist(b_lo, b_hi);
    return val_dist(rng);
}

static bool test_configuration(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    NTTPrecomputedTables tables,
    std::mt19937_64& rng,
    bool verbose = false
) {
    const uint32_t L = L_a + L_b;
    const size_t size_A_words = (size_t)N * L_a;
    const size_t size_B_words = (size_t)N * L_b;
    const size_t size_ret_words = (size_t)N * L;

    if (verbose) {
        printf("Testing N=%u, L_a=%u, L_b=%u (N*L=%zu)...\n", N, L_a, L_b, (size_t)N * L);
    }

    std::vector<uint32_t> h_A(size_A_words);
    std::vector<uint32_t> h_B(size_B_words);
    std::vector<uint32_t> h_ret(size_ret_words, 0);

    std::uniform_int_distribution<uint32_t> word_dist(0u, 0xffffffffu);
    for (size_t i = 0; i < size_A_words; ++i) h_A[i] = word_dist(rng);
    for (size_t i = 0; i < size_B_words; ++i) h_B[i] = word_dist(rng);

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    uint32_t* d_ret = nullptr;
    uint32_t* d_workspace = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, size_A_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ret, size_ret_words * sizeof(uint32_t)));

    size_t workspace_size = batch_mul_ntt_workspace_size(N, L_a, L_b);
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B_words * sizeof(uint32_t), cudaMemcpyHostToDevice));

    batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_ret.data(), d_ret, size_ret_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (uint32_t i = 0; i < N; ++i) {
        mpz_t a, b, expected, got;
        mpz_inits(a, b, expected, got, NULL);

        words_to_mpz(a, &h_A[(size_t)i * L_a], L_a);
        words_to_mpz(b, &h_B[(size_t)i * L_b], L_b);
        mpz_mul(expected, a, b);
        words_to_mpz(got, &h_ret[(size_t)i * L], L);

        if (mpz_cmp(expected, got) != 0) {
            pass = false;
            if (verbose) {
                printf("  Mismatch at index %u\n", i);
                gmp_printf("  A = %Zx\n", a);
                gmp_printf("  B = %Zx\n", b);
                gmp_printf("  Expected = %Zx\n", expected);
                gmp_printf("  Got      = %Zx\n", got);
            }
            mpz_clears(a, b, expected, got, NULL);
            break;
        }
        mpz_clears(a, b, expected, got, NULL);
    }

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_ret));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));

    return pass;
}

static bool run_suite(
    const char* name,
    uint32_t total_limit,
    int rounds,
    bool fixed_total,
    NTTPrecomputedTables tables,
    std::mt19937_64& rng
) {
    constexpr uint32_t L_MIN = 512;
    constexpr uint32_t L_MAX = 16384;

    printf("%s\n", name);
    std::uniform_int_distribution<uint32_t> total_dist(L_MIN, total_limit);

    for (int t = 0; t < rounds; ++t) {
        uint32_t target_total = fixed_total ? total_limit : total_dist(rng);
        uint32_t max_L = std::min(L_MAX, target_total);
        uint32_t L = sample_log_uniform_u32(L_MIN, max_L, rng);
        std::uniform_int_distribution<uint32_t> la_dist(1u, L - 1);
        uint32_t L_a = la_dist(rng);
        uint32_t L_b = L - L_a;
        uint32_t N = target_total / L;
        if (N == 0) N = 1;

        if (!test_configuration(N, L_a, L_b, tables, rng, true)) {
            return false;
        }
    }
    return true;
}

int main() {
    std::mt19937_64 rng(100ull);
    bool all_passed = true;

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

    printf("=== Correctness Tests ===\n\n");

    all_passed = run_suite("Small size tests (N*(L_a+L_b) <= 16384):", 16384, 12, false, tables, rng) && all_passed;
    printf("\n");
    if (!all_passed) goto done;

    all_passed = run_suite("Medium size tests (N*(L_a+L_b) <= 2^20):", (1u << 20), 8, false, tables, rng) && all_passed;
    printf("\n");
    if (!all_passed) goto done;

    all_passed = run_suite("Large size tests (N*(L_a+L_b) ~ 1e7):", 10000000u, 3, true, tables, rng) && all_passed;
    printf("\n");

done:
    CUDA_CHECK(cudaFree(d_lv1));
    CUDA_CHECK(cudaFree(d_lv2));
    CUDA_CHECK(cudaFree(d_inv));

    printf("=== Summary ===\n");
    printf("All correctness tests: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}
