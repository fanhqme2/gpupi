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

#include "batch_shift_add.h"

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

uint32_t calc_shifted_len(uint32_t L_a, uint32_t shift) {
    if (L_a == 0u) {
        return 0u;
    }
    return L_a + (shift >> 5) + ((shift & 31u) != 0u ? 1u : 0u);
}

uint32_t calc_len(uint32_t L_a, uint32_t L_b, uint32_t L_c, uint32_t shift) {
    const uint64_t shifted_len = calc_shifted_len(L_a, shift);
    const uint64_t used = std::max<uint64_t>(shifted_len, L_b) + 1u;
    return std::min<uint64_t>(L_c, used);
}

void words_to_mpz(mpz_t out, const uint32_t * words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

void compute_reference_shift_add(
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
    uint64_t carry = 0;
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
        const uint64_t sum = a + b + carry;
        out[i] = (uint32_t)sum;
        carry = sum >> 32;
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
    compute_reference_shift_add(a_row, b_row, expected.data(), shift, L_a, L_b, L_c);

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
    const uint32_t used = calc_len(L_a, L_b, L_c, shift);

    for (size_t s = 0; s < sample_indices.size(); ++s) {
        const uint32_t idx = sample_indices[s];
        mpz_t a, b, shifted, sum, expected, actual;
        mpz_inits(a, b, shifted, sum, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + s * stride_A, L_a);
        words_to_mpz(b, h_B.data() + s * stride_B, L_b);
        words_to_mpz(actual, h_C.data() + s * stride_C, used);
        mpz_mul_2exp(shifted, a, shift);
        mpz_add(sum, shifted, b);
        if (used == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, sum, (mp_bitcnt_t)used * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            printf("  GMP spot-check failed at index %u\n", idx);
            const uint32_t * a_row = h_A.data() + s * stride_A;
            const uint32_t * b_row = h_B.data() + s * stride_B;
            const uint32_t * c_row = h_C.data() + s * stride_C;
            std::vector<uint32_t> ref(used == 0 ? 1u : used);
            compute_reference_shift_add(a_row, b_row, ref.data(), shift, L_a, L_b, used);
            uint32_t mismatch_idx = 0;
            for (; mismatch_idx < used; ++mismatch_idx) {
                if (c_row[mismatch_idx] != ref[mismatch_idx]) {
                    break;
                }
            }
            if (mismatch_idx == used && used > 0) {
                mismatch_idx = used - 1;
            }
            print_limb_window(a_row, b_row, c_row, shift, L_a, L_b, used, mismatch_idx);
            mpz_clears(a, b, shifted, sum, expected, actual, NULL);
            return false;
        }

        mpz_clears(a, b, shifted, sum, expected, actual, NULL);
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
    const uint32_t used = calc_len(L_a, L_b, L_c, shift);
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
    const size_t workspace_size = batch_shift_add_simple_workspace_size(N, L_a, L_b, L_c, shift);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_shift_add_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, size_B, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = true;
    std::vector<uint32_t> expected(L_c == 0 ? 1u : L_c);
    for (uint32_t i = 0; i < N && pass; ++i) {
        const uint32_t * a_row = h_A.data() + (size_t)i * stride_A;
        const uint32_t * b_row = h_B.data() + (size_t)i * stride_B;
        const uint32_t * c_row = h_C.data() + (size_t)i * stride_C;
        compute_reference_shift_add(a_row, b_row, expected.data(), shift, L_a, L_b, L_c);

        for (uint32_t j = 0; j < L_c; ++j) {
            if (c_row[j] != expected[j]) {
                pass = false;
                if (verbose) {
                    printf("  Mismatch at index %u\n", i);
                    print_limb_window(a_row, b_row, c_row, shift, L_a, L_b, L_c, j);
                }
                break;
            }
        }
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

bool test_long_carry_chain_case(bool verbose = false) {
    constexpr uint32_t shift = 17u;
    constexpr uint32_t N = 1u;
    constexpr uint32_t L_a = 4096u;
    constexpr uint32_t L_b = 4097u;
    constexpr uint32_t L_c = 4098u;
    constexpr uint32_t stride_A = L_a;
    constexpr uint32_t stride_B = L_b;
    constexpr uint32_t stride_C = L_c;

    if (verbose) {
        printf("Testing dedicated long carry chain case with shift=%u...\n", shift);
    }

    std::vector<uint32_t> h_A((size_t)N * stride_A, 0u);
    std::vector<uint32_t> h_B((size_t)N * stride_B, 0u);
    std::vector<uint32_t> h_C((size_t)N * stride_C, kOutputPadPattern);
    std::vector<uint32_t> ref(L_c);

    std::fill(h_A.begin(), h_A.end(), 0xffffffffu);
    h_B[17] = 1u;
    compute_reference_shift_add(h_A.data(), h_B.data(), ref.data(), shift, L_a, L_b, L_c);

    uint32_t * d_A = nullptr;
    uint32_t * d_B = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t size_B = h_B.size() * sizeof(uint32_t);
    const size_t size_C = h_C.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_shift_add_simple_workspace_size(N, L_a, L_b, L_c, shift);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_shift_add_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = true;
    for (uint32_t i = 0; i < L_c; ++i) {
        if (h_C[i] != ref[i]) {
            pass = false;
            if (verbose) {
                printf("  Long carry chain mismatch at word %u: expected=%08x actual=%08x\n", i, ref[i], h_C[i]);
            }
            break;
        }
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
    const size_t workspace_size = batch_shift_add_simple_workspace_size(N, L_a, L_b, L_c, shift);

    printf("Benchmarking shift=%u, L_a=%u, L_b=%u, L_c=%u, N=%u...\n", shift, L_a, L_b, L_c, N);

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

    CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, 0x4d595df4ULL + shift));
    CURAND_CHECK(curandGenerate(rand_gen, d_A, size_A_words));
    CURAND_CHECK(curandGenerate(rand_gen, d_B, size_B_words));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    if (run_spot_check) {
        sample_indices = make_sample_indices(N, 5u);
        copy_sampled_rows_to_host(h_A_verify, d_A, stride_A, sample_indices);
        copy_sampled_rows_to_host(h_B_verify, d_B, stride_B, sample_indices);
    }

    batch_shift_add_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    batch_shift_add_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    batch_shift_add_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_shift_add_simple(d_A, d_B, d_C, d_workspace, shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
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
    printf("ShiftAdd/s: %8.2f K ", ops_per_sec_k);
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
    }

    CURAND_CHECK(curandDestroyGenerator(rand_gen));
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
        {1, 19, 17, 23, 18, 20, 29, 24},
        {31, 11, 31, 30, 62, 35, 34, 66},
        {32, 9, 64, 47, 112, 69, 52, 120},
        {63, 8, 128, 130, 131, 128, 130, 136},
        {95, 4, 257, 260, 263, 260, 264, 272},
        {17, 4, 5000, 4097, 6000, 5000, 4097, 6008},
        {3, 3, 6000, 6000, 7000, 6000, 6000, 7008},
        {64, 2, 8192, 8191, 9000, 8192, 8191, 9016},
        {127, 5, 0, 23, 24, 3, 27, 29},
    };

    for (const FixedCase & cfg : fixed_cases) {
        if (!test_configuration(cfg.shift, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c,
                                cfg.stride_A, cfg.stride_B, cfg.stride_C, rng, true)) {
            all_passed = false;
            break;
        }
    }

    if (all_passed && !test_long_carry_chain_case(true)) {
        all_passed = false;
    }

    if (all_passed) {
        printf("\nRandomized tests:\n");
        std::uniform_int_distribution<uint32_t> n_dist(1u, 128u);
        std::uniform_int_distribution<uint32_t> len_small_dist(0u, 40u);
        std::uniform_int_distribution<uint32_t> len_large_dist(32u, 12000u);
        std::uniform_int_distribution<uint32_t> shift_small_dist(0u, 127u);
        std::uniform_int_distribution<uint32_t> shift_large_dist(0u, 1024u);
        std::uniform_int_distribution<uint32_t> extra_stride_dist(0u, 8u);

        for (int t = 0; t < 30; ++t) {
            const bool large_case = (t >= 15);
            const uint32_t N = large_case ? std::uniform_int_distribution<uint32_t>(1u, 8u)(rng) : n_dist(rng);
            const uint32_t L_a = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t L_b = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t shift = large_case ? shift_large_dist(rng) : shift_small_dist(rng);
            const uint32_t shifted_len = calc_shifted_len(L_a, shift);
            const uint32_t sum_len = std::max(shifted_len, L_b) + 1u;
            uint32_t L_c = 0;
            switch (t % 3) {
                case 0: L_c = sum_len == 0 ? 0 : std::max<uint32_t>(1u, sum_len - (sum_len > 0 ? (sum_len / 3u) : 0u)); break;
                case 1: L_c = sum_len; break;
                default: L_c = sum_len + extra_stride_dist(rng) + 7u; break;
            }
            const uint32_t stride_A = L_a + extra_stride_dist(rng);
            const uint32_t stride_B = L_b + extra_stride_dist(rng);
            const uint32_t stride_C = L_c + extra_stride_dist(rng);
            if (!test_configuration(shift, N, L_a, L_b, L_c, stride_A, stride_B, stride_C, rng, true)) {
                all_passed = false;
                break;
            }
        }
    }

    printf("\n=== Benchmark Tests ===\n\n");
    const bool run_benchmark_spot_check = !skip_benchmark_spot_check;
    benchmark_configuration(0, 1, 1, 1, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(1, 1, 2, 2, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(7, 2, 3, 3, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(13, 4, 5, 6, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(17, 8, 9, 10, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(23, 16, 17, 18, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(31, 24, 25, 26, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(32, 30, 31, 32, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(33, 31, 33, 33, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(47, 32, 34, 34, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(63, 63, 65, 65, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(64, 64, 66, 66, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(65, 127, 130, 130, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(95, 128, 131, 131, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(17, 8, 1024, 1025, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(113, 1024, 1028, 1029, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(31, 256, 257, 258, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(17, 64, 4096, 4097, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(81, 4096, 4099, 4100, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(19, 1024, 1025, 1026, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(29, 4096, 4097, 4098, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(61, 16384, 16386, 16386, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(91, 262144, 262147, 262147, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(123, 524288, 524292, 524292, 100000000ull, run_benchmark_spot_check);
    for (int i = 20; i <= 27; ++i) {
        benchmark_configuration(37u, 1u << i, (1u << i) + 2u, (1u << i) + 2u, 100000000ull, run_benchmark_spot_check);
    }

    printf("\nSummary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}
