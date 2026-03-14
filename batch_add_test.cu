#include <cuda_runtime.h>
#include <gmp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "batch_add.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

namespace {

constexpr uint32_t kInputPadPatternA = 0xA5A5A5A5u;
constexpr uint32_t kInputPadPatternB = 0x5A5A5A5Au;
constexpr uint32_t kOutputPadPattern = 0xDEADBEEFu;

void words_to_mpz(mpz_t out, const uint32_t * words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

bool verify_results_with_gmp(
    const std::vector<uint32_t> & h_A,
    const std::vector<uint32_t> & h_B,
    const std::vector<uint32_t> & h_C,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C,
    uint32_t samples
) {
    const uint32_t calc_len = std::min<uint32_t>(L_c, std::max(L_a, L_b) + 1u);
    const uint32_t sample_count = std::min<uint32_t>(samples, N);
    if (sample_count == 0) {
        return true;
    }
    const uint32_t step = std::max<uint32_t>(1u, N / sample_count);

    for (uint32_t s = 0; s < sample_count; ++s) {
        const uint32_t idx = std::min<uint32_t>(N - 1u, s * step);
        mpz_t a, b, sum, expected, actual;
        mpz_inits(a, b, sum, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + (size_t)idx * stride_A, L_a);
        words_to_mpz(b, h_B.data() + (size_t)idx * stride_B, L_b);
        words_to_mpz(actual, h_C.data() + (size_t)idx * stride_C, calc_len);
        mpz_add(sum, a, b);
        if (calc_len == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, sum, (mp_bitcnt_t)calc_len * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            printf("  GMP spot-check failed at index %u\n", idx);
            gmp_printf("  A        = %Zx\n", a);
            gmp_printf("  B        = %Zx\n", b);
            gmp_printf("  Expected = %Zx\n", expected);
            gmp_printf("  Actual   = %Zx\n", actual);
            mpz_clears(a, b, sum, expected, actual, NULL);
            return false;
        }

        mpz_clears(a, b, sum, expected, actual, NULL);
    }
    return true;
}

bool verify_padding(const std::vector<uint32_t> & words, uint32_t N, uint32_t used, uint32_t stride, uint32_t pad) {
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t * row = words.data() + (size_t)i * stride;
        for (uint32_t j = used; j < stride; ++j) {
            if (row[j] != pad) {
                return false;
            }
        }
    }
    return true;
}

void fill_random_operand(
    std::vector<uint32_t> & words,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    uint32_t pad,
    std::mt19937_64 & rng
) {
    std::uniform_int_distribution<uint32_t> dist(0u, 0xffffffffu);
    std::fill(words.begin(), words.end(), pad);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t * row = words.data() + (size_t)i * stride;
        for (uint32_t j = 0; j < L; ++j) {
            row[j] = dist(rng);
        }
    }
}

bool test_configuration(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C,
    std::mt19937_64 & rng,
    bool verbose = false
) {
    const uint32_t calc_len = std::min<uint32_t>(L_c, std::max(L_a, L_b) + 1u);
    if (verbose) {
        printf("Testing N=%u, L_a=%u, L_b=%u, L_c=%u, stride_A=%u, stride_B=%u, stride_C=%u...\n",
               N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    }

    std::vector<uint32_t> h_A((size_t)N * stride_A);
    std::vector<uint32_t> h_B((size_t)N * stride_B);
    std::vector<uint32_t> h_C((size_t)N * stride_C, kOutputPadPattern);

    fill_random_operand(h_A, N, L_a, stride_A, kInputPadPatternA, rng);
    fill_random_operand(h_B, N, L_b, stride_B, kInputPadPatternB, rng);

    uint32_t * d_A = nullptr;
    uint32_t * d_B = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;

    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t size_B = h_B.size() * sizeof(uint32_t);
    const size_t size_C = h_C.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_add_simple_workspace_size(N, L_a, L_b, L_c);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_add_simple(d_A, d_B, d_C, d_workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, size_B, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = true;
    for (uint32_t i = 0; i < N && pass; ++i) {
        mpz_t a, b, sum, expected, actual;
        mpz_inits(a, b, sum, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + (size_t)i * stride_A, L_a);
        words_to_mpz(b, h_B.data() + (size_t)i * stride_B, L_b);
        words_to_mpz(actual, h_C.data() + (size_t)i * stride_C, calc_len);
        mpz_add(sum, a, b);
        if (calc_len == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, sum, (mp_bitcnt_t)calc_len * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            pass = false;
            if (verbose) {
                printf("  Mismatch at index %u\n", i);
                gmp_printf("  A        = %Zx\n", a);
                gmp_printf("  B        = %Zx\n", b);
                gmp_printf("  Expected = %Zx\n", expected);
                gmp_printf("  Actual   = %Zx\n", actual);
            }
        }

        if (pass) {
            const uint32_t * c_row = h_C.data() + (size_t)i * stride_C;
            for (uint32_t j = calc_len; j < L_c; ++j) {
                if (c_row[j] != 0u) {
                    pass = false;
                    if (verbose) {
                        printf("  Non-zero high padding at index %u word %u: 0x%08x\n", i, j, c_row[j]);
                    }
                    break;
                }
            }
        }

        mpz_clears(a, b, sum, expected, actual, NULL);
    }

    if (pass && !verify_padding(h_A, N, L_a, stride_A, kInputPadPatternA)) {
        pass = false;
        if (verbose) printf("  Input A padding was modified\n");
    }
    if (pass && !verify_padding(h_B, N, L_b, stride_B, kInputPadPatternB)) {
        pass = false;
        if (verbose) printf("  Input B padding was modified\n");
    }
    if (pass && !verify_padding(h_C, N, L_c, stride_C, kOutputPadPattern)) {
        pass = false;
        if (verbose) printf("  Output padding beyond L_c was modified\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }
    return pass;
}

void benchmark_configuration(uint32_t L_a, uint32_t L_b, uint32_t L_c, uint64_t target_words) {
    const uint32_t stride_A = L_a;
    const uint32_t stride_B = L_b;
    const uint32_t stride_C = L_c;
    uint32_t N = (uint32_t)(target_words / std::max<uint64_t>(L_c, 1u));
    if (N == 0) N = 1;

    const size_t size_A_words = (size_t)N * stride_A;
    const size_t size_B_words = (size_t)N * stride_B;
    const size_t size_C_words = (size_t)N * stride_C;
    const size_t size_A = size_A_words * sizeof(uint32_t);
    const size_t size_B = size_B_words * sizeof(uint32_t);
    const size_t size_C = size_C_words * sizeof(uint32_t);
    const size_t workspace_size = batch_add_simple_workspace_size(N, L_a, L_b, L_c);

    printf("Benchmarking L_a=%u, L_b=%u, L_c=%u, N=%u...\n", L_a, L_b, L_c, N);

    uint32_t * d_A = nullptr;
    uint32_t * d_B = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;
    std::vector<uint32_t> h_A_verify;
    std::vector<uint32_t> h_B_verify;
    std::vector<uint32_t> h_C_verify;

    cudaError_t err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_A failed: %s)\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_B, size_B);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_B failed: %s)\n", cudaGetErrorString(err));
        CUDA_CHECK(cudaFree(d_A));
        return;
    }
    err = cudaMalloc(&d_C, size_C);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_C failed: %s)\n", cudaGetErrorString(err));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        return;
    }
    if (workspace_size > 0) {
        err = cudaMalloc(&d_workspace, workspace_size);
        if (err != cudaSuccess) {
            printf("  SKIPPED (cudaMalloc workspace failed: %s)\n", cudaGetErrorString(err));
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            return;
        }
    }

    printf("  params size %4dM, workspace size %4dM ",
           int((size_A + size_B + size_C) / 1000000),
           int(workspace_size / 1000000));

    CUDA_CHECK(cudaMemset(d_A, 0x3c, size_A));
    CUDA_CHECK(cudaMemset(d_B, 0xc3, size_B));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    batch_add_simple(d_A, d_B, d_C, d_workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    batch_add_simple(d_A, d_B, d_C, d_workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    batch_add_simple(d_A, d_B, d_C, d_workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_add_simple(d_A, d_B, d_C, d_workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;
    const double adds_per_sec_k = ((double)N / (avg_ms * 1e-3)) / 1e3;
    const double io_bytes = (double)N * (double)(L_a + L_b + L_c) * sizeof(uint32_t);
    const double bandwidth_gb_s = io_bytes / (avg_ms * 1e-3) / 1e9;

    printf("Average time: %6.3f ms ", avg_ms);
    printf("Add/s: %8.2f K ", adds_per_sec_k);
    printf("Bandwidth (A+B+C): %6.2f GB/s\n", bandwidth_gb_s);

    h_A_verify.resize(size_A_words);
    h_B_verify.resize(size_B_words);
    h_C_verify.resize(size_C_words);
    CUDA_CHECK(cudaMemcpy(h_A_verify.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B_verify.data(), d_B, size_B, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_verify.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    if (!verify_results_with_gmp(
            h_A_verify, h_B_verify, h_C_verify,
            N, L_a, L_b, L_c,
            stride_A, stride_B, stride_C,
            5u)) {
        fprintf(stderr, "  GMP spot-check failed after benchmark\n");
        exit(1);
    }
    printf("  GMP spot-check: PASSED (5 samples)\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
}

}  // namespace

int main() {
    std::mt19937_64 rng(123456789ull);
    bool all_passed = true;

    printf("=== Correctness Tests ===\n\n");

    struct FixedCase {
        uint32_t N;
        uint32_t L_a;
        uint32_t L_b;
        uint32_t L_c;
        uint32_t stride_A;
        uint32_t stride_B;
        uint32_t stride_C;
    };

    const FixedCase fixed_cases[] = {
        {37, 7, 5, 9, 11, 9, 13},
        {19, 17, 23, 12, 20, 29, 16},
        {11, 31, 30, 31, 35, 34, 36},
        {9, 64, 47, 80, 69, 52, 84},
        {128, 300, 300, 301, 300, 300, 301},
        {4, 5000, 4097, 5001, 5000, 4097, 5001},
        {3, 6000, 6000, 7000, 6000, 6000, 7008},
        {2, 8192, 8191, 4096, 8192, 8191, 4096},
        {5, 0, 23, 24, 3, 27, 29},
    };

    for (const FixedCase & cfg : fixed_cases) {
        if (!test_configuration(cfg.N, cfg.L_a, cfg.L_b, cfg.L_c,
                                cfg.stride_A, cfg.stride_B, cfg.stride_C, rng, true)) {
            all_passed = false;
            break;
        }
    }

    if (all_passed) {
        printf("\nRandomized tests:\n");
        std::uniform_int_distribution<uint32_t> n_dist(1u, 128u);
        std::uniform_int_distribution<uint32_t> len_small_dist(0u, 40u);
        std::uniform_int_distribution<uint32_t> len_large_dist(32u, 12000u);
        std::uniform_int_distribution<uint32_t> extra_stride_dist(0u, 8u);

        for (int t = 0; t < 30; ++t) {
            const bool large_case = (t >= 15);
            const uint32_t N = large_case ? std::uniform_int_distribution<uint32_t>(1u, 8u)(rng) : n_dist(rng);
            const uint32_t L_a = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t L_b = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t sum_len = std::max(L_a, L_b) + 1u;
            uint32_t L_c = 0;
            switch (t % 3) {
                case 0: L_c = sum_len == 0 ? 0 : std::max<uint32_t>(1u, sum_len - (sum_len > 0 ? (sum_len / 3u) : 0u)); break;
                case 1: L_c = sum_len; break;
                default: L_c = sum_len + extra_stride_dist(rng) + 7u; break;
            }
            const uint32_t stride_A = L_a + extra_stride_dist(rng);
            const uint32_t stride_B = L_b + extra_stride_dist(rng);
            const uint32_t stride_C = L_c + extra_stride_dist(rng);
            if (!test_configuration(N, L_a, L_b, L_c, stride_A, stride_B, stride_C, rng, true)) {
                all_passed = false;
                break;
            }
        }
    }

    printf("\n=== Benchmark Tests ===\n\n");
    benchmark_configuration(16, 16, 17, 100000000ull);
    benchmark_configuration(8, 1024, 1025, 100000000ull);
    benchmark_configuration(1024, 8, 1025, 100000000ull);
    benchmark_configuration(256, 256, 257, 100000000ull);
    benchmark_configuration(64, 4096, 4097, 100000000ull);
    benchmark_configuration(4096, 64, 4097, 100000000ull);
    benchmark_configuration(1024, 1024, 1025, 100000000ull);
    benchmark_configuration(4096, 4096, 4097, 100000000ull);
    benchmark_configuration(16384, 16384, 16385, 100000000ull);
    benchmark_configuration(524288, 524288, 524289, 100000000ull);

    printf("\nSummary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}
