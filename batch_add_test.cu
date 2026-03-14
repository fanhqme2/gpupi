#include <cuda_runtime.h>
#include <curand.h>
#include <gmp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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

#define CURAND_CHECK(call) do { \
    curandStatus_t _err = (call); \
    if (_err != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, (int)_err); \
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

void print_limb_window(
    const uint32_t * a_row,
    const uint32_t * b_row,
    const uint32_t * c_row,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t mismatch_idx
) {
    const uint32_t window_lo = (mismatch_idx > 3u) ? mismatch_idx - 3u : 0u;
    const uint32_t window_hi = std::min<uint32_t>(L_c, mismatch_idx + 4u);
    uint64_t carry = 0;
    std::vector<uint32_t> expected(L_c);
    for (uint32_t i = 0; i < L_c; ++i) {
        const uint64_t a = (i < L_a) ? a_row[i] : 0u;
        const uint64_t b = (i < L_b) ? b_row[i] : 0u;
        const uint64_t sum = a + b + carry;
        expected[i] = (uint32_t)sum;
        carry = sum >> 32;
    }

    printf("  First mismatched limb: %u\n", mismatch_idx);
    printf("  Window [%u, %u):\n", window_lo, window_hi);
    for (uint32_t i = window_lo; i < window_hi; ++i) {
        printf("    limb %u: A=%08x B=%08x Expected=%08x Actual=%08x\n",
               i,
               (i < L_a) ? a_row[i] : 0u,
               (i < L_b) ? b_row[i] : 0u,
               expected[i],
               c_row[i]);
    }
}

std::vector<uint32_t> make_sample_indices(uint32_t N, uint32_t samples) {
    const uint32_t sample_count = std::min<uint32_t>(samples, N);
    std::vector<uint32_t> indices;
    indices.reserve(sample_count);
    if (sample_count == 0) {
        return indices;
    }

    const uint32_t step = std::max<uint32_t>(1u, N / sample_count);
    for (uint32_t s = 0; s < sample_count; ++s) {
        indices.push_back(std::min<uint32_t>(N - 1u, s * step));
    }
    return indices;
}

void copy_sampled_rows_to_host(
    std::vector<uint32_t> & dst,
    const uint32_t * src,
    uint32_t stride,
    const std::vector<uint32_t> & indices
) {
    dst.resize((size_t)indices.size() * stride);
    for (size_t i = 0; i < indices.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(
            dst.data() + i * stride,
            src + (size_t)indices[i] * stride,
            (size_t)stride * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
    }
}

bool verify_sampled_results_with_gmp(
    const std::vector<uint32_t> & h_A,
    const std::vector<uint32_t> & h_B,
    const std::vector<uint32_t> & h_C,
    const std::vector<uint32_t> & sample_indices,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    const uint32_t calc_len = std::min<uint32_t>(L_c, std::max(L_a, L_b) + 1u);

    for (size_t s = 0; s < sample_indices.size(); ++s) {
        const uint32_t idx = sample_indices[s];
        mpz_t a, b, sum, expected, actual;
        mpz_inits(a, b, sum, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + s * stride_A, L_a);
        words_to_mpz(b, h_B.data() + s * stride_B, L_b);
        words_to_mpz(actual, h_C.data() + s * stride_C, calc_len);
        mpz_add(sum, a, b);
        if (calc_len == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, sum, (mp_bitcnt_t)calc_len * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            printf("  GMP spot-check failed at index %u\n", idx);
            const uint32_t * a_row = h_A.data() + s * stride_A;
            const uint32_t * b_row = h_B.data() + s * stride_B;
            const uint32_t * c_row = h_C.data() + s * stride_C;
            uint32_t mismatch_idx = 0;
            uint64_t carry_words = 0;
            for (; mismatch_idx < calc_len; ++mismatch_idx) {
                const uint64_t av = (mismatch_idx < L_a) ? a_row[mismatch_idx] : 0u;
                const uint64_t bv = (mismatch_idx < L_b) ? b_row[mismatch_idx] : 0u;
                const uint64_t sum_word = av + bv + carry_words;
                if ((uint32_t)sum_word != c_row[mismatch_idx]) {
                    break;
                }
                carry_words = sum_word >> 32;
            }
            if (mismatch_idx == calc_len && calc_len > 0) {
                mismatch_idx = calc_len - 1;
            }
            print_limb_window(a_row, b_row, c_row, L_a, L_b, calc_len, mismatch_idx);
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
                const uint32_t * a_row = h_A.data() + (size_t)i * stride_A;
                const uint32_t * b_row = h_B.data() + (size_t)i * stride_B;
                const uint32_t * c_row = h_C.data() + (size_t)i * stride_C;
                uint32_t mismatch_idx = 0;
                uint64_t carry_words = 0;
                for (; mismatch_idx < calc_len; ++mismatch_idx) {
                    const uint64_t av = (mismatch_idx < L_a) ? a_row[mismatch_idx] : 0u;
                    const uint64_t bv = (mismatch_idx < L_b) ? b_row[mismatch_idx] : 0u;
                    const uint64_t sum_word = av + bv + carry_words;
                    if ((uint32_t)sum_word != c_row[mismatch_idx]) {
                        break;
                    }
                    carry_words = sum_word >> 32;
                }
                if (mismatch_idx == calc_len && calc_len > 0) {
                    mismatch_idx = calc_len - 1;
                }
                print_limb_window(a_row, b_row, c_row, L_a, L_b, calc_len, mismatch_idx);
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

void benchmark_configuration(
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint64_t target_words,
    bool run_spot_check
) {
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
    curandGenerator_t rand_gen = nullptr;
    std::vector<uint32_t> sample_indices;
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

    if (run_spot_check){
        CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, 0x4d595df4ULL));
        CURAND_CHECK(curandGenerate(rand_gen, d_A, size_A_words));
        CURAND_CHECK(curandGenerate(rand_gen, d_B, size_B_words));
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));

        sample_indices = make_sample_indices(N, 5u);
        copy_sampled_rows_to_host(h_A_verify, d_A, stride_A, sample_indices);
        copy_sampled_rows_to_host(h_B_verify, d_B, stride_B, sample_indices);
    }

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

    if (run_spot_check) {
        copy_sampled_rows_to_host(h_C_verify, d_C, stride_C, sample_indices);
        if (!verify_sampled_results_with_gmp(
                h_A_verify, h_B_verify, h_C_verify,
                sample_indices, L_a, L_b, L_c,
                stride_A, stride_B, stride_C)) {
            fprintf(stderr, "  GMP spot-check failed after benchmark\n");
            exit(1);
        }
        printf("  GMP spot-check: PASSED (%zu samples)\n", sample_indices.size());
        CURAND_CHECK(curandDestroyGenerator(rand_gen));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
}

}  // namespace

int main(int argc, char **argv) {
    bool skip_benchmark_spot_check = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--skip-spot-check") == 0) {
            skip_benchmark_spot_check = true;
            continue;
        }
        fprintf(stderr, "Usage: %s [--skip-spot-check]\n", argv[0]);
        return 1;
    }

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
    const bool run_benchmark_spot_check = !skip_benchmark_spot_check;
    benchmark_configuration(1, 1, 1, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(1, 1, 2, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(2, 2, 3, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(4, 4, 5, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(8, 8, 9, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(16, 16, 17, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(24, 24, 25, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(30, 30, 31, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(31, 31, 32, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(32, 32, 33, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(63, 63, 64, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(64, 64, 65, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(127, 127, 128, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(128, 128, 129, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(8, 1024, 1025, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(1024, 8, 1025, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(256, 256, 257, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(64, 4096, 4097, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(4096, 64, 4097, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(1024, 1024, 1025, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(4096, 4096, 4097, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(16384, 16384, 16385, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(262144, 262144, 262145, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(524288, 524288, 524289, 100000000ull, run_benchmark_spot_check);
    for (int i = 20; i <= 27; i ++){
        benchmark_configuration(1 << i, 1 << i, (1 << i), 100000000ull, run_benchmark_spot_check);
    }

    printf("\nSummary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}

/*
Benchmarking L_a=1, L_b=1, L_c=1, N=100000000...
  params size 1200M, workspace size    0M Average time:  0.782 ms Add/s: 127874624.88 K Bandwidth (A+B+C): 1534.50 GB/s
Benchmarking L_a=1, L_b=1, L_c=2, N=50000000...
  params size  800M, workspace size    0M Average time:  0.546 ms Add/s: 91613535.10 K Bandwidth (A+B+C): 1465.82 GB/s
Benchmarking L_a=2, L_b=2, L_c=3, N=33333333...
  params size  933M, workspace size    0M Average time:  0.611 ms Add/s: 54558028.85 K Bandwidth (A+B+C): 1527.62 GB/s
Benchmarking L_a=4, L_b=4, L_c=5, N=20000000...
  params size 1040M, workspace size    0M Average time:  0.699 ms Add/s: 28593046.61 K Bandwidth (A+B+C): 1486.84 GB/s
Benchmarking L_a=8, L_b=8, L_c=9, N=11111111...
  params size 1111M, workspace size    0M Average time:  0.937 ms Add/s: 11860229.26 K Bandwidth (A+B+C): 1186.02 GB/s
Benchmarking L_a=16, L_b=16, L_c=17, N=5882352...
  params size 1152M, workspace size    0M Average time:  0.735 ms Add/s: 8002380.87 K Bandwidth (A+B+C): 1568.47 GB/s
Benchmarking L_a=24, L_b=24, L_c=25, N=4000000...
  params size 1168M, workspace size    0M Average time:  0.747 ms Add/s: 5352214.45 K Bandwidth (A+B+C): 1562.85 GB/s
Benchmarking L_a=30, L_b=30, L_c=31, N=3225806...
  params size 1174M, workspace size    0M Average time:  0.768 ms Add/s: 4202814.36 K Bandwidth (A+B+C): 1529.82 GB/s
Benchmarking L_a=31, L_b=31, L_c=32, N=3125000...
  params size 1175M, workspace size    0M Average time:  0.838 ms Add/s: 3728917.50 K Bandwidth (A+B+C): 1402.07 GB/s
Benchmarking L_a=32, L_b=32, L_c=33, N=3030303...
  params size 1175M, workspace size    0M Average time:  0.840 ms Add/s: 3606028.05 K Bandwidth (A+B+C): 1399.14 GB/s
Benchmarking L_a=63, L_b=63, L_c=64, N=1562500...
  params size 1187M, workspace size    0M Average time:  0.768 ms Add/s: 2033439.30 K Bandwidth (A+B+C): 1545.41 GB/s
Benchmarking L_a=64, L_b=64, L_c=65, N=1538461...
  params size 1187M, workspace size    0M Average time:  0.771 ms Add/s: 1996166.26 K Bandwidth (A+B+C): 1541.04 GB/s
Benchmarking L_a=127, L_b=127, L_c=128, N=781250...
  params size 1193M, workspace size    0M Average time:  0.767 ms Add/s: 1018833.48 K Bandwidth (A+B+C): 1556.78 GB/s
Benchmarking L_a=128, L_b=128, L_c=129, N=775193...
  params size 1193M, workspace size    0M Average time:  0.763 ms Add/s: 1015819.71 K Bandwidth (A+B+C): 1564.36 GB/s
Benchmarking L_a=8, L_b=1024, L_c=1025, N=97560...
  params size  802M, workspace size    0M Average time:  0.547 ms Add/s: 178495.95 K Bandwidth (A+B+C): 1468.66 GB/s
Benchmarking L_a=1024, L_b=8, L_c=1025, N=97560...
  params size  802M, workspace size    0M Average time:  0.517 ms Add/s: 188531.02 K Bandwidth (A+B+C): 1551.23 GB/s
Benchmarking L_a=256, L_b=256, L_c=257, N=389105...
  params size 1196M, workspace size    0M Average time:  0.776 ms Add/s: 501533.17 K Bandwidth (A+B+C): 1542.72 GB/s
Benchmarking L_a=64, L_b=4096, L_c=4097, N=24408...
  params size  806M, workspace size    0M Average time:  0.559 ms Add/s: 43701.24 K Bandwidth (A+B+C): 1443.36 GB/s
Benchmarking L_a=4096, L_b=64, L_c=4097, N=24408...
  params size  806M, workspace size    0M Average time:  0.556 ms Add/s: 43907.97 K Bandwidth (A+B+C): 1450.19 GB/s
Benchmarking L_a=1024, L_b=1024, L_c=1025, N=97560...
  params size 1199M, workspace size    0M Average time:  0.757 ms Add/s: 128841.04 K Bandwidth (A+B+C): 1583.71 GB/s
Benchmarking L_a=4096, L_b=4096, L_c=4097, N=24408...
  params size 1199M, workspace size    0M Average time:  0.751 ms Add/s: 32521.27 K Bandwidth (A+B+C): 1598.62 GB/s
Benchmarking L_a=16384, L_b=16384, L_c=16385, N=6103...
  params size 1199M, workspace size    0M Average time:  0.763 ms Add/s:  7999.44 K Bandwidth (A+B+C): 1572.79 GB/s
Benchmarking L_a=262144, L_b=262144, L_c=262145, N=381...
  params size 1198M, workspace size    0M Average time:  0.834 ms Add/s:   456.88 K Bandwidth (A+B+C): 1437.23 GB/s
Benchmarking L_a=524288, L_b=524288, L_c=524289, N=190...
  params size 1195M, workspace size    0M Average time:  0.967 ms Add/s:   196.44 K Bandwidth (A+B+C): 1235.90 GB/s
Benchmarking L_a=1048576, L_b=1048576, L_c=1048576, N=95...
  params size 1195M, workspace size    0M Average time:  0.789 ms Add/s:   120.34 K Bandwidth (A+B+C): 1514.24 GB/s
Benchmarking L_a=2097152, L_b=2097152, L_c=2097152, N=47...
  params size 1182M, workspace size    0M Average time:  0.879 ms Add/s:    53.45 K Bandwidth (A+B+C): 1345.21 GB/s
Benchmarking L_a=4194304, L_b=4194304, L_c=4194304, N=23...
  params size 1157M, workspace size    0M Average time:  0.865 ms Add/s:    26.60 K Bandwidth (A+B+C): 1338.72 GB/s
Benchmarking L_a=8388608, L_b=8388608, L_c=8388608, N=11...
  params size 1107M, workspace size    0M Average time:  0.830 ms Add/s:    13.25 K Bandwidth (A+B+C): 1333.98 GB/s
Benchmarking L_a=16777216, L_b=16777216, L_c=16777216, N=5...
  params size 1006M, workspace size    0M Average time:  0.759 ms Add/s:     6.59 K Bandwidth (A+B+C): 1327.08 GB/s
Benchmarking L_a=33554432, L_b=33554432, L_c=33554432, N=2...
  params size  805M, workspace size    0M Average time:  0.631 ms Add/s:     3.17 K Bandwidth (A+B+C): 1276.81 GB/s
Benchmarking L_a=67108864, L_b=67108864, L_c=67108864, N=1...
  params size  805M, workspace size    0M Average time:  0.626 ms Add/s:     1.60 K Bandwidth (A+B+C): 1286.32 GB/s
Benchmarking L_a=134217728, L_b=134217728, L_c=134217728, N=1...
  params size 1610M, workspace size    0M Average time:  1.250 ms Add/s:     0.80 K Bandwidth (A+B+C): 1288.31 GB/s
*/