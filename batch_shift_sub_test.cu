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

#include "batch_shift_sub.h"

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

void compute_reference_shift_sub(
    const uint32_t * a_row,
    const uint32_t * b_row,
    uint32_t * out,
    uint32_t shift,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c
) {
    const uint32_t shift_words = shift >> 5;
    const uint32_t shift_bits = shift & 31u;
    uint64_t borrow = 0;
    for (uint32_t i = 0; i < L_c; ++i) {
        uint32_t shifted_word = 0u;
        if (i >= shift_words) {
            const uint32_t src_lo = i - shift_words;
            if (shift_bits == 0u) {
                shifted_word = (src_lo < L_a) ? a_row[src_lo] : 0u;
            } else {
                const uint32_t lo = (src_lo < L_a) ? a_row[src_lo] : 0u;
                const uint32_t hi = (src_lo > 0u && src_lo - 1u < L_a) ? a_row[src_lo - 1u] : 0u;
                shifted_word = (lo << shift_bits) | (hi >> (32u - shift_bits));
            }
        }
        const uint64_t a = shifted_word;
        const uint64_t b = (i < L_b) ? b_row[i] : 0u;
        const uint64_t diff = a - b - borrow;
        out[i] = (uint32_t)diff;
        borrow = (b + borrow > a) ? 1u : 0u;
    }
}

void print_limb_window(
    const uint32_t * a_row,
    const uint32_t * b_row,
    const uint32_t * c_row,
    uint32_t shift,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t mismatch_idx
) {
    const uint32_t window_lo = (mismatch_idx > 3u) ? mismatch_idx - 3u : 0u;
    const uint32_t window_hi = std::min<uint32_t>(L_c, mismatch_idx + 4u);
    std::vector<uint32_t> expected(L_c);
    compute_reference_shift_sub(a_row, b_row, expected.data(), shift, L_a, L_b, L_c);

    printf("  First mismatched limb: %u\n", mismatch_idx);
    printf("  Window [%u, %u):\n", window_lo, window_hi);
    for (uint32_t i = window_lo; i < window_hi; ++i) {
        uint32_t shifted_word = 0u;
        const uint32_t shift_words = shift >> 5;
        const uint32_t shift_bits = shift & 31u;
        if (i >= shift_words) {
            const uint32_t src_lo = i - shift_words;
            if (shift_bits == 0u) {
                shifted_word = (src_lo < L_a) ? a_row[src_lo] : 0u;
            } else {
                const uint32_t lo = (src_lo < L_a) ? a_row[src_lo] : 0u;
                const uint32_t hi = (src_lo > 0u && src_lo - 1u < L_a) ? a_row[src_lo - 1u] : 0u;
                shifted_word = (lo << shift_bits) | (hi >> (32u - shift_bits));
            }
        }
        printf("    limb %u: shifted(A)=%08x B=%08x Expected=%08x Actual=%08x\n",
               i,
               shifted_word,
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
    uint32_t shift,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    for (size_t s = 0; s < sample_indices.size(); ++s) {
        const uint32_t idx = sample_indices[s];
        mpz_t a, b, shifted, diff, expected, actual;
        mpz_inits(a, b, shifted, diff, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + s * stride_A, L_a);
        words_to_mpz(b, h_B.data() + s * stride_B, L_b);
        words_to_mpz(actual, h_C.data() + s * stride_C, L_c);
        mpz_mul_2exp(shifted, a, shift);
        mpz_sub(diff, shifted, b);
        if (L_c == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, diff, (mp_bitcnt_t)L_c * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            printf("  GMP spot-check failed at index %u\n", idx);
            const uint32_t * a_row = h_A.data() + s * stride_A;
            const uint32_t * b_row = h_B.data() + s * stride_B;
            const uint32_t * c_row = h_C.data() + s * stride_C;
            std::vector<uint32_t> ref(L_c == 0 ? 1u : L_c);
            compute_reference_shift_sub(a_row, b_row, ref.data(), shift, L_a, L_b, L_c);
            uint32_t mismatch_idx = 0;
            for (; mismatch_idx < L_c; ++mismatch_idx) {
                if (c_row[mismatch_idx] != ref[mismatch_idx]) {
                    break;
                }
            }
            if (mismatch_idx == L_c && L_c > 0) {
                mismatch_idx = L_c - 1;
            }
            print_limb_window(a_row, b_row, c_row, shift, L_a, L_b, L_c, mismatch_idx);
            mpz_clears(a, b, shifted, diff, expected, actual, NULL);
            return false;
        }

        mpz_clears(a, b, shifted, diff, expected, actual, NULL);
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
    uint32_t shift,
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
    if (verbose) {
        printf("Testing shift=%u N=%u, L_a=%u, L_b=%u, L_c=%u, stride_A=%u, stride_B=%u, stride_C=%u...\n",
               shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
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
    const size_t workspace_size = batch_shift_sub_simple_workspace_size(N, L_a, L_b, L_c, shift);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_shift_sub_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, size_B, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = true;
    for (uint32_t i = 0; i < N && pass; ++i) {
        mpz_t a, b, shifted, diff, expected, actual;
        mpz_inits(a, b, shifted, diff, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + (size_t)i * stride_A, L_a);
        words_to_mpz(b, h_B.data() + (size_t)i * stride_B, L_b);
        words_to_mpz(actual, h_C.data() + (size_t)i * stride_C, L_c);
        mpz_mul_2exp(shifted, a, shift);
        mpz_sub(diff, shifted, b);
        if (L_c == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, diff, (mp_bitcnt_t)L_c * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            pass = false;
            if (verbose) {
                printf("  Mismatch at index %u\n", i);
                const uint32_t * a_row = h_A.data() + (size_t)i * stride_A;
                const uint32_t * b_row = h_B.data() + (size_t)i * stride_B;
                const uint32_t * c_row = h_C.data() + (size_t)i * stride_C;
                std::vector<uint32_t> ref(L_c == 0 ? 1u : L_c);
                compute_reference_shift_sub(a_row, b_row, ref.data(), shift, L_a, L_b, L_c);
                uint32_t mismatch_idx = 0;
                for (; mismatch_idx < L_c; ++mismatch_idx) {
                    if (c_row[mismatch_idx] != ref[mismatch_idx]) {
                        break;
                    }
                }
                if (mismatch_idx == L_c && L_c > 0) {
                    mismatch_idx = L_c - 1;
                }
                print_limb_window(a_row, b_row, c_row, shift, L_a, L_b, L_c, mismatch_idx);
            }
        }

        mpz_clears(a, b, shifted, diff, expected, actual, NULL);
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
    uint32_t shift,
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
    const size_t workspace_size = batch_shift_sub_simple_workspace_size(N, L_a, L_b, L_c, shift);

    printf("Benchmarking shift=%u L_a=%u, L_b=%u, L_c=%u, N=%u...\n", shift, L_a, L_b, L_c, N);

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

    if (run_spot_check) {
        CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, 0x4d595df4ULL));
        CURAND_CHECK(curandGenerate(rand_gen, d_A, size_A_words));
        CURAND_CHECK(curandGenerate(rand_gen, d_B, size_B_words));
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));

        sample_indices = make_sample_indices(N, 5u);
        copy_sampled_rows_to_host(h_A_verify, d_A, stride_A, sample_indices);
        copy_sampled_rows_to_host(h_B_verify, d_B, stride_B, sample_indices);
    }

    batch_shift_sub_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    batch_shift_sub_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    batch_shift_sub_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_shift_sub_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;
    const double ops_per_sec_k = ((double)N / (avg_ms * 1e-3)) / 1e3;
    const double io_bytes = (double)N * (double)(L_a + L_b + L_c) * sizeof(uint32_t);
    const double bandwidth_gb_s = io_bytes / (avg_ms * 1e-3) / 1e9;

    printf("Average time: %6.3f ms ", avg_ms);
    printf("Shift-subs/s: %8.2f K ", ops_per_sec_k);
    printf("Bandwidth (A+B+C): %6.2f GB/s\n", bandwidth_gb_s);

    if (run_spot_check) {
        copy_sampled_rows_to_host(h_C_verify, d_C, stride_C, sample_indices);
        if (!verify_sampled_results_with_gmp(
                h_A_verify, h_B_verify, h_C_verify,
                sample_indices, shift, L_a, L_b, L_c,
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
        uint32_t shift;
        uint32_t N;
        uint32_t L_a;
        uint32_t L_b;
        uint32_t L_c;
        uint32_t stride_A;
        uint32_t stride_B;
        uint32_t stride_C;
    };

    const FixedCase fixed_cases[] = {
        {0, 37, 7, 5, 9, 11, 9, 13},
        {1, 19, 17, 23, 12, 20, 29, 16},
        {31, 11, 31, 30, 31, 35, 34, 36},
        {32, 9, 64, 47, 80, 69, 52, 84},
        {33, 128, 300, 300, 301, 300, 300, 301},
        {63, 4, 5000, 4097, 5001, 5000, 4097, 5001},
        {65, 3, 6000, 6000, 7000, 6000, 6000, 7008},
        {127, 2, 8192, 8191, 4096, 8192, 8191, 4096},
        {95, 5, 0, 23, 24, 3, 27, 29},
        {47, 5, 23, 0, 24, 27, 3, 29},
    };

    for (const FixedCase & cfg : fixed_cases) {
        if (!test_configuration(cfg.shift, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c,
                                cfg.stride_A, cfg.stride_B, cfg.stride_C, rng, true)) {
            all_passed = false;
            return 1;
        }
    }

    if (all_passed) {
        printf("\nRandomized tests:\n");
        std::uniform_int_distribution<uint32_t> n_dist(1u, 128u);
        std::uniform_int_distribution<uint32_t> len_small_dist(0u, 40u);
        std::uniform_int_distribution<uint32_t> len_large_dist(32u, 12000u);
        std::uniform_int_distribution<uint32_t> extra_stride_dist(0u, 8u);
        std::uniform_int_distribution<uint32_t> shift_small_dist(0u, 96u);
        std::uniform_int_distribution<uint32_t> shift_large_dist(0u, 4096u);

        for (int t = 0; t < 30; ++t) {
            const bool large_case = (t >= 15);
            const uint32_t N = large_case ? std::uniform_int_distribution<uint32_t>(1u, 8u)(rng) : n_dist(rng);
            const uint32_t L_a = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t L_b = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t shift = large_case ? shift_large_dist(rng) : shift_small_dist(rng);
            uint32_t L_c = 0;
            switch (t % 3) {
                case 0: L_c = std::max<uint32_t>(L_b, L_a + (shift >> 5) + ((shift & 31u) != 0u ? 1u : 0u)); break;
                case 1: L_c = std::max<uint32_t>(L_b, L_a + (shift >> 5) + ((shift & 31u) != 0u ? 1u : 0u)) + extra_stride_dist(rng) + 1u; break;
                default: {
                    const uint32_t max_len = std::max<uint32_t>(L_b, L_a + (shift >> 5) + ((shift & 31u) != 0u ? 1u : 0u));
                    L_c = (max_len == 0) ? 0u : std::max<uint32_t>(1u, max_len / 2u);
                    break;
                }
            }
            const uint32_t stride_A = L_a + extra_stride_dist(rng);
            const uint32_t stride_B = L_b + extra_stride_dist(rng);
            const uint32_t stride_C = L_c + extra_stride_dist(rng);
            if (!test_configuration(shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C, rng, true)) {
                all_passed = false;
                return 1;
            }
        }
    }

    printf("\n=== Benchmark Tests ===\n\n");
    const bool run_benchmark_spot_check = !skip_benchmark_spot_check;
    benchmark_configuration(0, 1, 1, 1, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(13, 1, 1, 2, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(7, 2, 2, 3, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(11, 4, 4, 5, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(31, 8, 8, 9, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(32, 16, 16, 17, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(23, 24, 24, 25, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(29, 30, 30, 31, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(31, 31, 31, 32, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(63, 32, 32, 34, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(63, 63, 63, 64, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(64, 64, 64, 65, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(95, 127, 127, 128, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(5, 128, 128, 129, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(8, 1024, 8, 1025, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(8, 8, 1024, 1025, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(17, 256, 256, 257, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(64, 4096, 64, 4097, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(64, 64, 4096, 4097, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(17, 1024, 1024, 1025, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(29, 4096, 4096, 4097, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(37, 16384, 16384, 16385, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(17, 512, 512, 513, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(29, 2048, 2048, 2049, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(31, 8192, 8192, 8193, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(37, 32768, 32768, 32770, 100000000ull, run_benchmark_spot_check);

    if (!all_passed) {
        return 1;
    }

    printf("\nAll tests passed.\n");
    return 0;
}
