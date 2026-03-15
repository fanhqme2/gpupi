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

#include "batch_add_small.h"

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
    return std::min<uint32_t>(L_c, std::max<uint32_t>(L_a, 1u) + 1u);
}

void words_to_mpz(mpz_t out, const uint32_t * words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

void compute_reference_add_small(
    const uint32_t * a_row,
    uint32_t B,
    uint32_t * out,
    uint32_t L_a,
    uint32_t L_c
) {
    const uint32_t used = calc_len(L_a, L_c);
    uint64_t carry = 0;
    for (uint32_t i = 0; i < used; ++i) {
        const uint64_t a = (i < L_a) ? a_row[i] : 0u;
        const uint64_t addend = (i == 0u) ? B : 0u;
        const uint64_t sum = a + addend + carry;
        out[i] = (uint32_t)sum;
        carry = sum >> 32;
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
    compute_reference_add_small(a_row, B, expected.data(), L_a, L_c);

    printf("  First mismatched limb: %u\n", mismatch_idx);
    printf("  Window [%u, %u):\n", window_lo, window_hi);
    for (uint32_t i = window_lo; i < window_hi; ++i) {
        printf("    limb %u: A=%08x B=%08x Expected=%08x Actual=%08x\n",
               i,
               (i < L_a) ? a_row[i] : 0u,
               (i == 0u) ? B : 0u,
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
        compute_reference_add_small(a_row, B, expected.data(), L_a, L_c);

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
    const size_t workspace_size = batch_add_small_workspace_size(N, L_a, L_c);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_add_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
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

bool test_in_place_configuration(
    uint32_t N,
    uint32_t L,
    uint32_t B,
    std::mt19937_64 & rng,
    bool verbose = false
) {
    if (verbose) {
        printf("Testing in-place N=%u, L=%u, B=%08x...\n", N, L, B);
    }

    std::vector<uint32_t> h_A((size_t)N * L);
    std::vector<uint32_t> h_before;
    fill_random_operand(h_A, N, L, L, 0u, rng);
    h_before = h_A;

    uint32_t * d_A = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_add_small_workspace_size(N, L, L);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    batch_add_small(d_A, B, d_A, d_workspace, N, L, L, L, L);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));

    bool pass = verify_rows(h_before, h_A, B, N, L, L, L, L, verbose);

    CUDA_CHECK(cudaFree(d_A));
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
    bool in_place,
    bool verbose
) {
    if (verbose) {
        printf("Testing long carry chain %s case: N=%u, L_a=%u, L_c=%u, B=%08x...\n",
               in_place ? "in-place" : "out-of-place", N, L_a, L_c, B);
    }

    std::vector<uint32_t> h_A((size_t)N * stride_A, in_place ? 0u : kInputPadPatternA);
    std::vector<uint32_t> h_A_before;
    std::vector<uint32_t> h_C = in_place ? std::vector<uint32_t>() : std::vector<uint32_t>((size_t)N * stride_C, kOutputPadPattern);

    for (uint32_t n = 0; n < N; ++n) {
        uint32_t * row = h_A.data() + (size_t)n * stride_A;
        for (uint32_t i = 0; i < L_a; ++i) {
            row[i] = 0xffffffffu;
        }
    }
    h_A_before = h_A;

    uint32_t * d_A = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t size_C = in_place ? size_A : h_C.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_add_small_workspace_size(N, L_a, L_c);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    if (in_place) {
        d_C = d_A;
    } else {
        CUDA_CHECK(cudaMalloc(&d_C, size_C));
    }
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    if (!in_place) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));
    }

    batch_add_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (in_place) {
        CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    }

    bool pass = true;
    if (in_place) {
        pass = verify_rows(h_A_before, h_A, B, N, L_a, L_c, stride_A, stride_C, verbose);
    } else {
        pass = verify_rows(h_A_before, h_C, B, N, L_a, L_c, stride_A, stride_C, verbose);
        if (pass && !verify_padding(h_A, N, L_a, stride_A, kInputPadPatternA)) {
            pass = false;
            if (verbose) printf("  Input A padding was modified\n");
        }
        if (pass && !verify_padding(h_C, N, L_c, stride_C, kOutputPadPattern)) {
            pass = false;
            if (verbose) printf("  Output padding beyond L_c was modified\n");
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    if (!in_place) CUDA_CHECK(cudaFree(d_C));
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
        mpz_t a, b, sum, expected, actual;
        mpz_inits(a, b, sum, expected, actual, NULL);

        words_to_mpz(a, h_A.data() + s * stride_A, L_a);
        mpz_set_ui(b, B);
        words_to_mpz(actual, h_C.data() + s * stride_C, used);
        mpz_add(sum, a, b);
        if (used == 0) {
            mpz_set_ui(expected, 0u);
        } else {
            mpz_fdiv_r_2exp(expected, sum, (mp_bitcnt_t)used * 32u);
        }

        if (mpz_cmp(expected, actual) != 0) {
            mpz_clears(a, b, sum, expected, actual, NULL);
            return false;
        }
        mpz_clears(a, b, sum, expected, actual, NULL);
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
    const size_t workspace_size = batch_add_small_workspace_size(N, L_a, L_c);

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

    batch_add_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    batch_add_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    batch_add_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_add_small(d_A, B, d_C, d_workspace, N, L_a, L_c, stride_A, stride_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;
    const double adds_per_sec_k = ((double)N / (avg_ms * 1e-3)) / 1e3;
    const double io_bytes = (double)N * (double)(L_a + L_c) * sizeof(uint32_t);
    const double bandwidth_gb_s = io_bytes / (avg_ms * 1e-3) / 1e9;

    printf("Average time: %6.3f ms ", avg_ms);
    printf("Add/s: %8.2f K ", adds_per_sec_k);
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
        uint32_t L_c;
        uint32_t stride_A;
        uint32_t stride_C;
        uint32_t B;
    };

    const FixedCase fixed_cases[] = {
        {37, 0, 2, 3, 4, 0u},
        {37, 0, 2, 3, 4, 0xffffffffu},
        {19, 7, 9, 11, 13, 1u},
        {11, 17, 12, 20, 16, 0xdeadbeefu},
        {9, 31, 31, 35, 36, 0xffffffffu},
        {9, 31, 33, 35, 37, 1u},
        {8, 64, 65, 69, 71, 0xffffffffu},
        {128, 300, 301, 300, 301, 1u},
        {4, 4096, 4097, 4099, 4105, 1u},
        {3, 6000, 7000, 6004, 7008, 0xffffffffu},
        {2, 8192, 4096, 8197, 4101, 7u},
    };

    for (const FixedCase & cfg : fixed_cases) {
        if (!test_configuration(cfg.N, cfg.L_a, cfg.L_c, cfg.stride_A, cfg.stride_C, cfg.B, rng, true)) {
            all_passed = false;
            break;
        }
    }

    if (all_passed && !test_long_carry_chain_case(5, 23, 24, 27, 29, 1u, false, true)) {
        all_passed = false;
    }
    if (all_passed && !test_long_carry_chain_case(3, 1024, 1025, 1030, 1032, 1u, false, true)) {
        all_passed = false;
    }
    if (all_passed && !test_long_carry_chain_case(1, 5000, 5001, 5000, 5001, 1u, false, true)) {
        all_passed = false;
    }

    if (all_passed) {
        printf("\nRandomized tests:\n");
        std::uniform_int_distribution<uint32_t> n_dist(1u, 128u);
        std::uniform_int_distribution<uint32_t> len_small_dist(0u, 40u);
        std::uniform_int_distribution<uint32_t> len_large_dist(32u, 12000u);
        std::uniform_int_distribution<uint32_t> extra_stride_dist(0u, 8u);
        std::uniform_int_distribution<uint32_t> b_dist(0u, 0xffffffffu);

        for (int t = 0; t < 30; ++t) {
            const bool large_case = (t >= 15);
            const uint32_t N = large_case ? std::uniform_int_distribution<uint32_t>(1u, 8u)(rng) : n_dist(rng);
            const uint32_t L_a = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t sum_len = std::max<uint32_t>(L_a, 1u) + 1u;
            uint32_t L_c = 0;
            switch (t % 3) {
                case 0: L_c = std::max<uint32_t>(1u, sum_len - (sum_len / 3u)); break;
                case 1: L_c = sum_len; break;
                default: L_c = sum_len + extra_stride_dist(rng) + 7u; break;
            }
            const uint32_t stride_A = L_a + extra_stride_dist(rng);
            const uint32_t stride_C = L_c + extra_stride_dist(rng);
            if (!test_configuration(N, L_a, L_c, stride_A, stride_C, b_dist(rng), rng, true)) {
                all_passed = false;
                break;
            }
        }
    }

    if (all_passed) {
        printf("\nIn-place tests:\n");
        const FixedCase in_place_cases[] = {
            {37, 1, 1, 1, 1, 1u},
            {19, 23, 23, 23, 23, 0xffffffffu},
            {7, 256, 256, 256, 256, 1u},
            {3, 1024, 1024, 1024, 1024, 1u},
            {1, 5000, 5000, 5000, 5000, 1u},
        };

        for (const FixedCase & cfg : in_place_cases) {
            if (!test_in_place_configuration(cfg.N, cfg.L_a, cfg.B, rng, true)) {
                all_passed = false;
                break;
            }
        }
    }

    if (all_passed && !test_long_carry_chain_case(7, 31, 31, 31, 31, 1u, true, true)) {
        all_passed = false;
    }
    if (all_passed && !test_long_carry_chain_case(2, 1024, 1024, 1024, 1024, 1u, true, true)) {
        all_passed = false;
    }
    if (all_passed && !test_long_carry_chain_case(1, 5000, 5000, 5000, 5000, 1u, true, true)) {
        all_passed = false;
    }

    printf("\n=== Benchmark Tests ===\n\n");
    const bool run_benchmark_spot_check = !skip_benchmark_spot_check;
    benchmark_configuration(1, 2, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(8, 9, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(31, 32, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(32, 33, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(64, 65, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(256, 257, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(1024, 1025, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(4096, 4097, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(16384, 16385, 1u, 100000000ull, run_benchmark_spot_check);
    benchmark_configuration(262144, 262145, 1u, 100000000ull, run_benchmark_spot_check);

    printf("\nSummary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}
