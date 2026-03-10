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

static void benchmark_configuration(uint32_t L_a, uint32_t L_b, uint64_t target_NL, NTTPrecomputedTables tables) {
    uint32_t N = (uint32_t)(target_NL / L_a);
    if (N == 0) N = 1;
    const uint32_t L = L_a + L_b;

    printf("Benchmarking L_a=%u, L_b=%u, N=%u (N*L_a=%llu)...\n",
           L_a, L_b, N, (unsigned long long)((uint64_t)N * L_a));

    const size_t size_A = (size_t)N * L_a * sizeof(uint32_t);
    const size_t size_B = (size_t)N * L_b * sizeof(uint32_t);
    const size_t size_ret = (size_t)N * L * sizeof(uint32_t);
    const size_t workspace_size = batch_mul_ntt_workspace_size(N, L_a, L_b);

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    uint32_t* d_ret = nullptr;
    uint32_t* d_workspace = nullptr;

    cudaError_t err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_A failed: %s)\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_B, size_B);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_B failed: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc(&d_ret, size_ret);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_ret failed: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }
    if (workspace_size > 0) {
        err = cudaMalloc(&d_workspace, workspace_size);
        if (err != cudaSuccess) {
            printf("  SKIPPED (cudaMalloc workspace failed: %s)\n", cudaGetErrorString(err));
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_ret);
            return;
        }
    }
    printf("  params size %dM, workspace size %dM\n",
        int((size_A + size_B + size_ret) / 1000000),
        int(workspace_size / 1000000));

    // Warmup
    batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;

    const double mul_per_sec_thousand = ((double)N / (avg_ms * 1e-3)) / 1e3;
    const double io_words = (double)N * (double)(L_a + L_b + L);
    const double bandwidth_gb_s = (io_words * sizeof(uint32_t)) / (avg_ms * 1e-3) / 1e9;

    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Multiplications per second: %.2f thousand\n", mul_per_sec_thousand);
    printf("  Approx. bandwidth (A+B+ret): %.2f GB/s\n", bandwidth_gb_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_ret));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
}

int main() {
    std::mt19937_64 rng(100ull);
    bool all_passed = true;
    const uint64_t target_NL = 100000000ull; // keep N * L_a around 1e8

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
    if (!all_passed) goto done;

    printf("Large L tests (N=1,2,3 with N*(L_a+L_b) ~ 1e7):\n");
    {
        constexpr uint32_t target_total = 10000000u;
        const uint32_t large_l_ns[] = {1u, 2u, 3u};
        for (uint32_t N_case : large_l_ns) {
            uint32_t L = target_total / N_case;
            uint32_t L_a = L / 2;
            uint32_t L_b = L - L_a;
            if (!test_configuration(N_case, L_a, L_b, tables, rng, true)) {
                all_passed = false;
                break;
            }
        }
    }
    printf("\n");
    if (!all_passed) goto done;

    printf("=== Benchmark Tests ===\n\n");
    for (uint32_t e = 9; e <= 25; ++e) {
        
        uint32_t L_a = 1u << e;
        uint32_t L_b = L_a;
        benchmark_configuration(L_a, L_b, target_NL, tables);

        
        if (e <= 20 && e >= 14){
            e += 3;
        }else if (e == 12){
            e ++;
        }
    }
    printf("\n");

done:
    CUDA_CHECK(cudaFree(d_lv1));
    CUDA_CHECK(cudaFree(d_lv2));
    CUDA_CHECK(cudaFree(d_inv));

    printf("=== Summary ===\n");
    printf("All correctness tests: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}

/*
Benchmarking L_a=512, L_b=512, N=195312 (N*L_a=99999744)...
  params size 1599M, workspace size 95M
  Average time: 14.757 ms
  Multiplications per second: 13235.17 thousand
  Approx. bandwidth (A+B+ret): 108.42 GB/s
Benchmarking L_a=1024, L_b=1024, N=97656 (N*L_a=99999744)...
  params size 1599M, workspace size 95M
  Average time: 16.080 ms
  Multiplications per second: 6072.97 thousand
  Approx. bandwidth (A+B+ret): 99.50 GB/s
Benchmarking L_a=2048, L_b=2048, N=48828 (N*L_a=99999744)...
  params size 1599M, workspace size 95M
  Average time: 17.992 ms
  Multiplications per second: 2713.80 thousand
  Approx. bandwidth (A+B+ret): 88.93 GB/s
Benchmarking L_a=4096, L_b=4096, N=24414 (N*L_a=99999744)...
  params size 1599M, workspace size 95M
  Average time: 21.378 ms
  Multiplications per second: 1142.03 thousand
  Approx. bandwidth (A+B+ret): 74.84 GB/s
Benchmarking L_a=16384, L_b=16384, N=6103 (N*L_a=99991552)...
  params size 1599M, workspace size 95M
  Average time: 24.358 ms
  Multiplications per second: 250.55 thousand
  Approx. bandwidth (A+B+ret): 65.68 GB/s
Benchmarking L_a=262144, L_b=262144, N=381 (N*L_a=99876864)...
  params size 1598M, workspace size 88M
  Average time: 29.080 ms
  Multiplications per second: 13.10 thousand
  Approx. bandwidth (A+B+ret): 54.95 GB/s
Benchmarking L_a=4194304, L_b=4194304, N=23 (N*L_a=96468992)...
  params size 1543M, workspace size 201M
  Average time: 47.135 ms
  Multiplications per second: 0.49 thousand
  Approx. bandwidth (A+B+ret): 32.75 GB/s
Benchmarking L_a=8388608, L_b=8388608, N=11 (N*L_a=92274688)...
  params size 1476M, workspace size 402M
  Average time: 52.793 ms
  Multiplications per second: 0.21 thousand
  Approx. bandwidth (A+B+ret): 27.97 GB/s
Benchmarking L_a=16777216, L_b=16777216, N=5 (N*L_a=83886080)...
  params size 1342M, workspace size 805M
  Average time: 52.275 ms
  Multiplications per second: 0.10 thousand
  Approx. bandwidth (A+B+ret): 25.68 GB/s
Benchmarking L_a=33554432, L_b=33554432, N=2 (N*L_a=67108864)...
  params size 1073M, workspace size 1610M
  Average time: 44.731 ms
  Multiplications per second: 0.04 thousand
  Approx. bandwidth (A+B+ret): 24.00 GB/s
*/
