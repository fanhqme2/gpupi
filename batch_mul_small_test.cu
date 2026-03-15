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

#include "batch_mul_small.h"

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
constexpr uint32_t kOutputPadPattern = 0xDEADBEEFu;

uint32_t calc_len(uint32_t L_a, uint32_t L_c) {
    return std::min<uint32_t>(L_c, L_a + 1u);
}

void words_to_mpz(mpz_t out, const uint32_t * words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

void compute_reference_mul_small(
    const uint32_t * a_row,
    uint32_t B,
    uint32_t * out,
    uint32_t L_a,
    uint32_t L_c
) {
    const uint32_t used = calc_len(L_a, L_c);
    uint64_t carry = 0;
    uint32_t prev_hi = 0u;
    for (uint32_t i = 0; i < used; ++i) {
        uint32_t lo = 0u;
        uint32_t hi = 0u;
        if (i < L_a) {
            const uint64_t product = (uint64_t)a_row[i] * (uint64_t)B;
            lo = (uint32_t)product;
            hi = (uint32_t)(product >> 32);
        }
        const uint64_t sum = (uint64_t)lo + (uint64_t)prev_hi + carry;
        out[i] = (uint32_t)sum;
        carry = sum >> 32;
        prev_hi = hi;
    }
    for (uint32_t i = used; i < L_c; ++i) {
        out[i] = 0u;
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
    const uint32_t * a_row,
    const uint32_t * c_row,
    uint32_t B,
    uint32_t L_a,
    uint32_t L_c,
    uint32_t mismatch_idx
) {
    const uint32_t window_lo = (mismatch_idx > 3u) ? mismatch_idx - 3u : 0u;
    const uint32_t window_hi = std::min<uint32_t>(L_c, mismatch_idx + 4u);
    std::vector<uint32_t> expected(L_c);
    compute_reference_mul_small(a_row, B, expected.data(), L_a, L_c);

    printf("  First mismatched limb: %u\n", mismatch_idx);
    printf("  Window [%u, %u):\n", window_lo, window_hi);
    for (uint32_t i = window_lo; i < window_hi; ++i) {
        const uint64_t prod = (i < L_a) ? ((uint64_t)a_row[i] * (uint64_t)B) : 0u;
        printf("    limb %u: A=%08x B=%08x lo=%08x hi(prev)=%08x Expected=%08x Actual=%08x\n",
               i,
               (i < L_a) ? a_row[i] : 0u,
               B,
               (uint32_t)prod,
               (i > 0u && i - 1u < L_a) ? (uint32_t)(((uint64_t)a_row[i - 1u] * (uint64_t)B) >> 32) : 0u,
               expected[i],
               c_row[i]);
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

bool verify_rows(
    const std::vector<uint32_t> & h_A_before,
    const std::vector<uint32_t> & h_C,
    uint32_t B,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_C,
    bool verbose
) {
    std::vector<uint32_t> expected(L_c);
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t * a_row = h_A_before.data() + (size_t)i * stride_A;
        const uint32_t * c_row = h_C.data() + (size_t)i * stride_C;
        compute_reference_mul_small(a_row, B, expected.data(), L_a, L_c);

        for (uint32_t j = 0; j < L_c; ++j) {
            if (c_row[j] != expected[j]) {
                if (verbose) {
                    printf("  Mismatch at index %u\n", i);
                    print_limb_window(a_row, c_row, B, L_a, L_c, j);
                }
                return false;
            }
        }
    }
    return true;
}

bool test_configuration(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_C,
    uint32_t B,
    std::mt19937_64 & rng,
    bool verbose = false
) {
    if (verbose) {
        printf("Testing N=%u, L_a=%u, L_c=%u, stride_A=%u, stride_C=%u, B=%08x...\n",
               N, L_a, L_c, stride_A, stride_C, B);
    }

    std::vector<uint32_t> h_A((size_t)N * stride_A);
    std::vector<uint32_t> h_A_before;
    std::vector<uint32_t> h_C((size_t)N * stride_C, kOutputPadPattern);

    fill_random_operand(h_A, N, L_a, stride_A, kInputPadPatternA, rng);
    h_A_before = h_A;

    uint32_t * d_A = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;

    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t size_C = h_C.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_mul_small_workspace_size(N, L_a, L_c);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_mul_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = verify_rows(h_A_before, h_C, B, N, L_a, L_c, stride_A, stride_C, verbose);
    if (pass && !verify_padding(h_A, N, L_a, stride_A, kInputPadPatternA)) {
        pass = false;
        if (verbose) printf("  Input A padding was modified\n");
    }
    if (pass && !verify_padding(h_C, N, L_c, stride_C, kOutputPadPattern)) {
        pass = false;
        if (verbose) printf("  Output padding beyond L_c was modified\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }
    return pass;
}

bool test_long_carry_chain_case(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_C,
    uint32_t B,
    bool verbose
) {
    if (verbose) {
        printf("Testing long carry chain case: N=%u, L_a=%u, L_c=%u, B=%08x...\n",
               N, L_a, L_c, B);
    }

    std::vector<uint32_t> h_A((size_t)N * stride_A, kInputPadPatternA);
    std::vector<uint32_t> h_A_before;
    std::vector<uint32_t> h_C((size_t)N * stride_C, kOutputPadPattern);

    for (uint32_t n = 0; n < N; ++n) {
        uint32_t * row = h_A.data() + (size_t)n * stride_A;
        if (L_a > 0) {
            row[0] = 0xffffffffu;
        }
        for (uint32_t i = 1; i < L_a; ++i) {
            row[i] = 0xffffffffu;
        }
    }
    h_A_before = h_A;

    uint32_t * d_A = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t size_C = h_C.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_mul_small_workspace_size(N, L_a, L_c);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_mul_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = verify_rows(h_A_before, h_C, B, N, L_a, L_c, stride_A, stride_C, verbose);
    if (pass && !verify_padding(h_A, N, L_a, stride_A, kInputPadPatternA)) {
        pass = false;
        if (verbose) printf("  Input A padding was modified\n");
    }
    if (pass && !verify_padding(h_C, N, L_c, stride_C, kOutputPadPattern)) {
        pass = false;
        if (verbose) printf("  Output padding beyond L_c was modified\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }
    return pass;
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
    const std::vector<uint32_t> & h_C,
    uint32_t B,
    uint32_t L_a,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_C
) {
    const uint32_t used = calc_len(L_a, L_c);

    for (size_t s = 0; s < h_A.size() / stride_A; ++s) {
        mpz_t a, b, product, expected, actual;
        mpz_inits(a, b, product, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + s * stride_A, L_a);
        mpz_set_ui(b, B);
        words_to_mpz(actual, h_C.data() + s * stride_C, used);
        mpz_mul(product, a, b);
        if (used == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, product, (mp_bitcnt_t)used * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            mpz_clears(a, b, product, expected, actual, NULL);
            return false;
        }
        mpz_clears(a, b, product, expected, actual, NULL);
    }
    return true;
}

void benchmark_configuration(
    uint32_t L_a,
    uint32_t L_c,
    uint32_t B,
    uint64_t target_words,
    bool run_spot_check
) {
    const uint32_t stride_A = L_a;
    const uint32_t stride_C = L_c;
    uint32_t N = (uint32_t)(target_words / std::max<uint64_t>(L_c, 1u));
    if (N == 0) N = 1;

    const size_t size_A_words = (size_t)N * stride_A;
    const size_t size_C_words = (size_t)N * stride_C;
    const size_t size_A = size_A_words * sizeof(uint32_t);
    const size_t size_C = size_C_words * sizeof(uint32_t);
    const size_t workspace_size = batch_mul_small_workspace_size(N, L_a, L_c);

    printf("Benchmarking L_a=%u, L_c=%u, N=%u...\n", L_a, L_c, N);

    uint32_t * d_A = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;
    curandGenerator_t rand_gen = nullptr;
    std::vector<uint32_t> sample_indices;
    std::vector<uint32_t> h_A_verify;
    std::vector<uint32_t> h_C_verify;

    cudaError_t err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_A failed: %s)\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_C, size_C);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_C failed: %s)\n", cudaGetErrorString(err));
        CUDA_CHECK(cudaFree(d_A));
        return;
    }
    if (workspace_size > 0) {
        err = cudaMalloc(&d_workspace, workspace_size);
        if (err != cudaSuccess) {
            printf("  SKIPPED (cudaMalloc workspace failed: %s)\n", cudaGetErrorString(err));
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_C));
            return;
        }
    }

    printf("  params size %4dM, workspace size %4dM ",
           int((size_A + size_C) / 1000000),
           int(workspace_size / 1000000));

    CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, 0x4d595df4ULL));
    CURAND_CHECK(curandGenerate(rand_gen, d_A, size_A_words));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    if (run_spot_check) {
        sample_indices = make_sample_indices(N, 5u);
        copy_sampled_rows_to_host(h_A_verify, d_A, stride_A, sample_indices);
    }

    batch_mul_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    batch_mul_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    batch_mul_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_mul_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;
    const double muls_per_sec_k = ((double)N / (avg_ms * 1e-3)) / 1e3;
    const double io_bytes = (double)N * (double)(L_a + L_c) * sizeof(uint32_t);
    const double bandwidth_gb_s = io_bytes / (avg_ms * 1e-3) / 1e9;

    printf("Average time: %6.3f ms ", avg_ms);
    printf("Mul/s: %8.2f K ", muls_per_sec_k);
    printf("Bandwidth (A+C): %6.2f GB/s\n", bandwidth_gb_s);

    if (run_spot_check) {
        copy_sampled_rows_to_host(h_C_verify, d_C, stride_C, sample_indices);
        if (!verify_sampled_results_with_gmp(h_A_verify, h_C_verify, B, L_a, L_c, stride_A, stride_C)) {
            fprintf(stderr, "  GMP spot-check failed after benchmark\n");
            exit(1);
        }
        printf("  GMP spot-check: PASSED (%zu samples)\n", sample_indices.size());
    }

    CURAND_CHECK(curandDestroyGenerator(rand_gen));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
}

}  // namespace

int main() {
    std::mt19937_64 rng(0x123456789abcdef0ULL);
    bool pass = true;

    const uint32_t b_values[] = {
        0u,
        1u,
        0xffffffffu,
        0x80000000u,
        0x7fffffffu,
        0x13579bdfu
    };

    const struct {
        uint32_t N;
        uint32_t L_a;
        uint32_t L_c;
        uint32_t stride_A;
        uint32_t stride_C;
    } cases[] = {
        {1u, 1u, 1u, 2u, 2u},
        {7u, 1u, 2u, 3u, 3u},
        {19u, 7u, 8u, 9u, 10u},
        {17u, 8u, 9u, 11u, 12u},
        {23u, 31u, 32u, 33u, 34u},
        {29u, 32u, 33u, 34u, 35u},
        {13u, 63u, 64u, 65u, 66u},
        {11u, 64u, 65u, 67u, 68u},
        {5u, 255u, 256u, 257u, 258u},
        {3u, 2048u, 2049u, 2049u, 2050u},
        {2u, 4096u, 4097u, 4097u, 4098u},
        {97u, 512u, 513u, 512u, 513u}
    };

    for (uint32_t b : b_values) {
        for (const auto & cfg : cases) {
            pass &= test_configuration(
                cfg.N, cfg.L_a, cfg.L_c, cfg.stride_A, cfg.stride_C, b, rng, !pass
            );
            if (!pass) break;
        }
        if (!pass) break;
    }

    if (pass) {
        pass &= test_long_carry_chain_case(1u, 4096u, 4097u, 4097u, 4098u, 0xffffffffu, true);
    }
    if (pass) {
        pass &= test_long_carry_chain_case(4u, 8192u, 8193u, 8193u, 8194u, 0xffffffffu, true);
    }

    if (!pass) {
        fprintf(stderr, "batch_mul_small tests FAILED\n");
        return 1;
    }

    printf("All batch_mul_small correctness tests PASSED\n");

    const uint32_t bench_b = 0xfedcba98u;
    const uint64_t target_words = 256ull * 1024ull * 1024ull;
    const uint32_t bench_lengths[] = {
        1u,
        2u,
        4u,
        8u,
        16u,
        32u,
        64u,
        256u,
        2048u,
        32768u,
        1u << 20,
        1u << 22,
        1u << 24,
        1u << 26
    };
    for (uint32_t L_a : bench_lengths) {
        benchmark_configuration(L_a, L_a + 1u, bench_b, target_words, true);
    }

    return 0;
}
