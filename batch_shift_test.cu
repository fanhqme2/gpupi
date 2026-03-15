#include <cuda_runtime.h>
#include <curand.h>
#include <gmp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "batch_shift.h"

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

constexpr uint32_t kInputPadPattern = 0xA5A5A5A5u;
constexpr uint32_t kOutputPadPattern = 0xDEADBEEFu;

void words_to_mpz(mpz_t out, const uint32_t * words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

void mpz_to_words_truncated(std::vector<uint32_t> & out, const mpz_t value, uint32_t L_out) {
    out.assign(L_out, 0u);
    size_t written = 0;
    mpz_export(out.data(), &written, -1, sizeof(uint32_t), 0, 0, value);
    if (written > L_out) {
        out.resize(L_out);
    }
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

void print_limb_window(
    const uint32_t * in_row,
    const uint32_t * out_row,
    const std::vector<uint32_t> & expected,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t mismatch_idx
) {
    const uint32_t lo = (mismatch_idx > 3u) ? mismatch_idx - 3u : 0u;
    const uint32_t hi = std::min<uint32_t>(L_out, mismatch_idx + 4u);
    printf("  First mismatched limb: %u\n", mismatch_idx);
    printf("  Window [%u, %u):\n", lo, hi);
    for (uint32_t i = lo; i < hi; ++i) {
        printf("    limb %u: Expected=%08x Actual=%08x\n", i, expected[i], out_row[i]);
    }
    const uint32_t in_hi = std::min<uint32_t>(L_in, lo + 8u);
    printf("  Input limbs [%u, %u):\n", lo, in_hi);
    for (uint32_t i = lo; i < in_hi; ++i) {
        printf("    in[%u]=%08x\n", i, in_row[i]);
    }
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
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    int32_t shift_bits,
    std::mt19937_64 & rng,
    bool verbose
) {
    if (verbose) {
        printf("Testing N=%u, L_in=%u, L_out=%u, stride_in=%u, stride_out=%u, shift_bits=%d...\n",
               N, L_in, L_out, stride_in, stride_out, shift_bits);
    }

    std::vector<uint32_t> h_A((size_t)N * stride_in);
    std::vector<uint32_t> h_B((size_t)N * stride_out, kOutputPadPattern);
    fill_random_operand(h_A, N, L_in, stride_in, kInputPadPattern, rng);

    uint32_t * d_A = nullptr;
    uint32_t * d_B = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t size_B = h_B.size() * sizeof(uint32_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    batch_shift_bits(d_A, d_B, N, L_in, L_out, stride_in, stride_out, shift_bits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, size_B, cudaMemcpyDeviceToHost));

    bool pass = true;
    std::vector<uint32_t> expected_words;
    for (uint32_t i = 0; i < N && pass; ++i) {
        const uint32_t * in_row = h_A.data() + (size_t)i * stride_in;
        const uint32_t * out_row = h_B.data() + (size_t)i * stride_out;

        mpz_t input, shifted;
        mpz_inits(input, shifted, NULL);
        words_to_mpz(input, in_row, L_in);
        if (shift_bits >= 0) {
            mpz_mul_2exp(shifted, input, (mp_bitcnt_t)shift_bits);
        } else {
            mpz_fdiv_q_2exp(shifted, input, (mp_bitcnt_t)(-shift_bits));
        }
        if (L_out == 0) {
            expected_words.clear();
        } else {
            mpz_fdiv_r_2exp(shifted, shifted, (mp_bitcnt_t)L_out * 32u);
            mpz_to_words_truncated(expected_words, shifted, L_out);
        }

        for (uint32_t j = 0; j < L_out; ++j) {
            if (out_row[j] != expected_words[j]) {
                pass = false;
                if (verbose) {
                    printf("  Mismatch at row %u\n", i);
                    print_limb_window(in_row, out_row, expected_words, L_in, L_out, j);
                }
                break;
            }
        }
        mpz_clears(input, shifted, NULL);
    }

    if (pass && !verify_padding(h_A, N, L_in, stride_in, kInputPadPattern)) {
        pass = false;
        if (verbose) printf("  Input padding was modified\n");
    }
    if (pass && !verify_padding(h_B, N, L_out, stride_out, kOutputPadPattern)) {
        pass = false;
        if (verbose) printf("  Output padding beyond L_out was modified\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }
    return pass;
}

void benchmark_configuration(
    uint32_t L,
    int32_t shift_bits,
    uint64_t target_words,
    bool run_spot_check
) {
    const uint32_t L_in = L;
    const uint32_t L_out = L;
    const uint32_t stride_in = L_in;
    const uint32_t stride_out = L_out;
    uint32_t N = (uint32_t)(target_words / std::max<uint64_t>(L, 1u));
    if (N == 0) N = 1;

    const size_t words_A = (size_t)N * stride_in;
    const size_t words_B = (size_t)N * stride_out;
    const size_t size_A = words_A * sizeof(uint32_t);
    const size_t size_B = words_B * sizeof(uint32_t);

    printf("Benchmarking L=%u, N=%u (N*L=%llu), shift_bits=%d...\n",
           L, N, (unsigned long long)((uint64_t)N * L), shift_bits);

    uint32_t * d_A = nullptr;
    uint32_t * d_B = nullptr;
    curandGenerator_t rand_gen = nullptr;

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

    CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, 0x13579BDF2468ACE0ULL + (uint64_t)(uint32_t)shift_bits));
    CURAND_CHECK(curandGenerate(rand_gen, d_A, words_A));
    CUDA_CHECK(cudaMemset(d_B, 0, size_B));

    if (run_spot_check) {
        const uint32_t sample_N = std::min<uint32_t>(N, 4u);
        std::vector<uint32_t> h_A((size_t)sample_N * stride_in);
        std::vector<uint32_t> h_B((size_t)sample_N * stride_out, 0u);
        CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, h_A.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        batch_shift_bits(d_A, d_B, N, L_in, L_out, stride_in, stride_out, shift_bits);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, h_B.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < sample_N; ++i) {
            mpz_t input, shifted;
            mpz_inits(input, shifted, NULL);
            words_to_mpz(input, h_A.data() + (size_t)i * stride_in, L_in);
            if (shift_bits >= 0) {
                mpz_mul_2exp(shifted, input, (mp_bitcnt_t)shift_bits);
            } else {
                mpz_fdiv_q_2exp(shifted, input, (mp_bitcnt_t)(-shift_bits));
            }
            mpz_fdiv_r_2exp(shifted, shifted, (mp_bitcnt_t)L_out * 32u);
            std::vector<uint32_t> expected;
            mpz_to_words_truncated(expected, shifted, L_out);
            for (uint32_t j = 0; j < L_out; ++j) {
                if (h_B[(size_t)i * stride_out + j] != expected[j]) {
                    fprintf(stderr, "  GMP spot-check failed at row %u limb %u\n", i, j);
                    exit(1);
                }
            }
            mpz_clears(input, shifted, NULL);
        }
        printf("  GMP spot-check: PASSED (%u samples)\n", sample_N);
    }

    batch_shift_bits(d_A, d_B, N, L_in, L_out, stride_in, stride_out, shift_bits);
    batch_shift_bits(d_A, d_B, N, L_in, L_out, stride_in, stride_out, shift_bits);
    batch_shift_bits(d_A, d_B, N, L_in, L_out, stride_in, stride_out, shift_bits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_shift_bits(d_A, d_B, N, L_in, L_out, stride_in, stride_out, shift_bits);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;
    const double shifts_per_sec_m = ((double)N / (avg_ms * 1e-3)) / 1e6;
    const double logical_bytes = (double)N * (double)(L_in + L_out) * sizeof(uint32_t);
    const double bandwidth_gb_s = logical_bytes / (avg_ms * 1e-3) / 1e9;

    printf("  params size %4dM Average time: %6.3f ms Shift/s: %8.3f M Bandwidth (A+B): %7.2f GB/s\n",
           int((size_A + size_B) / 1000000),
           avg_ms,
           shifts_per_sec_m,
           bandwidth_gb_s);

    CURAND_CHECK(curandDestroyGenerator(rand_gen));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
}

}  // namespace

int main(int argc, char **argv) {
    bool skip_benchmark_spot_check = false;
    bool run_correctness = true;
    bool run_benchmarks = true;
    int benchmark_limit = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--skip-spot-check") == 0) {
            skip_benchmark_spot_check = true;
            continue;
        }
        if (std::strcmp(argv[i], "--correctness-only") == 0) {
            run_benchmarks = false;
            continue;
        }
        if (std::strcmp(argv[i], "--benchmark-only") == 0) {
            run_correctness = false;
            continue;
        }
        if (std::strncmp(argv[i], "--benchmark-limit=", 18) == 0) {
            benchmark_limit = std::atoi(argv[i] + 18);
            continue;
        }
        fprintf(stderr, "Usage: %s [--skip-spot-check] [--correctness-only] [--benchmark-only] [--benchmark-limit=N]\n", argv[0]);
        return 1;
    }

    std::mt19937_64 rng(123456789ull);
    bool all_passed = true;

    if (run_correctness) {
        printf("=== Correctness Tests ===\n\n");

        struct FixedCase {
            uint32_t N;
            uint32_t L_in;
            uint32_t L_out;
            uint32_t stride_in;
            uint32_t stride_out;
            int32_t shift_bits;
        };

        const FixedCase fixed_cases[] = {
            {37, 7, 9, 11, 13, 0},
            {19, 17, 12, 20, 16, 1},
            {19, 17, 12, 20, 16, -1},
            {11, 31, 31, 35, 36, 31},
            {11, 31, 31, 35, 36, -31},
            {9, 64, 80, 69, 84, 32},
            {9, 64, 80, 69, 84, -32},
            {7, 64, 80, 69, 84, 63},
            {7, 64, 80, 69, 84, -63},
            {4, 5000, 5001, 5007, 5009, 97},
            {4, 5000, 5001, 5007, 5009, -97},
            {3, 6000, 7000, 6008, 7008, 4096},
            {3, 6000, 7000, 6008, 7008, -4096},
            {2, 8192, 4096, 8192, 4096, 777},
            {2, 8192, 4096, 8192, 4096, -777},
            {5, 0, 24, 3, 29, 17},
            {5, 23, 0, 27, 3, -17},
        };

        for (const FixedCase & cfg : fixed_cases) {
            if (!test_configuration(cfg.N, cfg.L_in, cfg.L_out,
                                    cfg.stride_in, cfg.stride_out, cfg.shift_bits, rng, true)) {
                all_passed = false;
                return 1;
            }
        }

        if (all_passed) {
            printf("\nRandomized tests:\n");
            std::uniform_int_distribution<uint32_t> n_small_dist(1u, 128u);
            std::uniform_int_distribution<uint32_t> n_large_dist(1u, 8u);
            std::uniform_int_distribution<uint32_t> len_small_dist(0u, 80u);
            std::uniform_int_distribution<uint32_t> len_large_dist(256u, 12000u);
            std::uniform_int_distribution<uint32_t> extra_stride_dist(0u, 8u);
            std::uniform_int_distribution<int32_t> shift_small_dist(-160, 160);

            for (int t = 0; t < 36; ++t) {
                const bool large_case = (t >= 18);
                const uint32_t N = large_case ? n_large_dist(rng) : n_small_dist(rng);
                const uint32_t L_in = large_case ? len_large_dist(rng) : len_small_dist(rng);
                uint32_t L_out = 0;
                switch (t % 4) {
                    case 0: L_out = L_in; break;
                    case 1: L_out = L_in + extra_stride_dist(rng) + 1u; break;
                    case 2: L_out = (L_in == 0u) ? 0u : std::max<uint32_t>(1u, L_in / 2u); break;
                    default: L_out = L_in + 37u; break;
                }
                const uint32_t stride_in = L_in + extra_stride_dist(rng);
                const uint32_t stride_out = L_out + extra_stride_dist(rng);
                int32_t shift_bits = shift_small_dist(rng);
                if (large_case) {
                    const int32_t span = (int32_t)(L_in * 32u + 256u);
                    shift_bits = (int32_t)(rng() % (uint64_t)(2 * (uint32_t)span + 1u)) - span;
                }
                if (!test_configuration(N, L_in, L_out, stride_in, stride_out, shift_bits, rng, true)) {
                    all_passed = false;
                    return 1;
                }
            }
        }
    }

    if (run_benchmarks) {
        printf("\n=== Benchmark Tests ===\n\n");
        const bool run_benchmark_spot_check = !skip_benchmark_spot_check;
        const uint32_t benchmark_L[] = {
            1u, 2u, 4u, 8u, 16u, 32u, 64u, 256u, 1024u, 4096u, 16384u, 65536u, 262144u, 1048576u
        };
        const int32_t benchmark_shifts[] = {1, -1, 17, -17, 32, -32, 63, -63};
        int benchmark_count = 0;
        for (uint32_t L : benchmark_L) {
            for (int32_t shift_bits : benchmark_shifts) {
                if (benchmark_limit >= 0 && benchmark_count >= benchmark_limit) {
                    break;
                }
                benchmark_configuration(L, shift_bits, 100000000ull, run_benchmark_spot_check);
                ++benchmark_count;
            }
            if (benchmark_limit >= 0 && benchmark_count >= benchmark_limit) {
                break;
            }
        }
    }

    printf("\nSummary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}
