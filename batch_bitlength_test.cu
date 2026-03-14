#include <cuda_runtime.h>
#include <gmp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "batch_bitlength.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

namespace {

constexpr uint32_t kInputPadPattern = 0xA5A5A5A5u;

enum class FillMode {
    RandomFull,
    VerySmall,
    HalfLength
};

void words_to_mpz(mpz_t out, const uint32_t * words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

uint32_t gmp_bitlength(const uint32_t * words, uint32_t L) {
    mpz_t value;
    mpz_init(value);
    words_to_mpz(value, words, L);
    const uint32_t bits = (mpz_sgn(value) == 0) ? 0u : (uint32_t)mpz_sizeinbase(value, 2);
    mpz_clear(value);
    return bits;
}

void fill_input(
    std::vector<uint32_t> & words,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    FillMode mode,
    std::mt19937_64 & rng
) {
    std::uniform_int_distribution<uint32_t> dist(0u, 0xffffffffu);
    std::fill(words.begin(), words.end(), kInputPadPattern);

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t * row = words.data() + (size_t)i * stride;
        for (uint32_t j = 0; j < L; ++j) {
            row[j] = 0u;
        }
        switch (mode) {
            case FillMode::RandomFull:
                for (uint32_t j = 0; j < L; ++j) {
                    row[j] = dist(rng);
                }
                break;
            case FillMode::VerySmall:
                if (L > 0) {
                    row[0] = (dist(rng) & 0xffffu) + 1u;
                }
                if (L > 1 && (i & 1u) != 0u) {
                    row[1] = dist(rng) & 0xffu;
                }
                break;
            case FillMode::HalfLength:
                if (L > 0) {
                    const uint32_t top = (L - 1u) / 2u;
                    row[top] = 1u << (dist(rng) % 31u);
                    for (uint32_t j = 0; j < top; ++j) {
                        row[j] = dist(rng);
                    }
                }
                break;
        }
    }
}

bool verify_padding(const std::vector<uint32_t> & words, uint32_t N, uint32_t L, uint32_t stride) {
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t * row = words.data() + (size_t)i * stride;
        for (uint32_t j = L; j < stride; ++j) {
            if (row[j] != kInputPadPattern) {
                return false;
            }
        }
    }
    return true;
}

const char * fill_mode_name(FillMode mode) {
    switch (mode) {
        case FillMode::RandomFull: return "random-full";
        case FillMode::VerySmall: return "very-small";
        case FillMode::HalfLength: return "half-length";
    }
    return "unknown";
}

bool test_configuration(
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    FillMode mode,
    std::mt19937_64 & rng,
    bool verbose
) {
    if (verbose) {
        printf("Testing N=%u, L=%u, stride=%u, mode=%s...\n", N, L, stride, fill_mode_name(mode));
    }

    std::vector<uint32_t> h_A((size_t)N * stride);
    fill_input(h_A, N, L, stride, mode, rng);

    uint32_t expected = 0u;
    for (uint32_t i = 0; i < N; ++i) {
        expected = std::max(expected, gmp_bitlength(h_A.data() + (size_t)i * stride, L));
    }

    uint32_t * d_A = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_bitlength_workspace_size(N, L);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));

    const uint32_t actual = batch_bitlength_max(d_A, d_workspace, N, L, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));

    bool pass = (actual == expected);
    if (!pass && verbose) {
        printf("  Bitlength mismatch: expected=%u actual=%u\n", expected, actual);
    }
    if (pass && !verify_padding(h_A, N, L, stride)) {
        pass = false;
        if (verbose) {
            printf("  Input padding was modified\n");
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }
    return pass;
}

void benchmark_configuration(
    uint32_t N,
    uint32_t L,
    FillMode mode
) {
    const uint32_t stride = L;
    std::vector<uint32_t> h_A((size_t)N * stride);
    std::mt19937_64 rng(0x1234abcdULL + (uint64_t)L * 17u + (uint64_t)N);
    fill_input(h_A, N, L, stride, mode, rng);

    uint32_t * d_A = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_bitlength_workspace_size(N, L);

    printf("Benchmarking N=%u, L=%u, mode=%s...\n", N, L, fill_mode_name(mode));

    cudaError_t err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_A failed: %s)\n", cudaGetErrorString(err));
        return;
    }
    if (workspace_size > 0) {
        err = cudaMalloc(&d_workspace, workspace_size);
        if (err != cudaSuccess) {
            printf("  SKIPPED (cudaMalloc workspace failed: %s)\n", cudaGetErrorString(err));
            CUDA_CHECK(cudaFree(d_A));
            return;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));

    batch_bitlength_max(d_A, d_workspace, N, L, stride);
    batch_bitlength_max(d_A, d_workspace, N, L, stride);
    batch_bitlength_max(d_A, d_workspace, N, L, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    uint32_t last_result = 0u;
    for (int i = 0; i < iterations; ++i) {
        last_result = batch_bitlength_max(d_A, d_workspace, N, L, stride);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;
    const double numbers_per_sec_m = ((double)N / (avg_ms * 1e-3)) / 1e6;
    const double read_bytes = (double)N * (double)L * sizeof(uint32_t);
    const double bandwidth_gb_s = read_bytes / (avg_ms * 1e-3) / 1e9;

    printf("  params size %4dM, workspace size %4dM Average time: %6.3f ms Throughput: %8.3f Mnum/s Bandwidth: %6.2f GB/s Result=%u\n",
           int(size_A / 1000000),
           int(workspace_size / 1000000),
           avg_ms,
           numbers_per_sec_m,
           bandwidth_gb_s,
           last_result);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
}

}  // namespace

int main() {
    std::mt19937_64 rng(123456789ull);
    bool all_passed = true;

    printf("=== Correctness Tests ===\n\n");

    struct FixedCase {
        uint32_t N;
        uint32_t L;
        uint32_t stride;
        FillMode mode;
    };

    const FixedCase fixed_cases[] = {
        {37, 7, 11, FillMode::RandomFull},
        {64, 31, 36, FillMode::VerySmall},
        {128, 300, 304, FillMode::RandomFull},
        {96, 1024, 1028, FillMode::VerySmall},
        {96, 1024, 1024, FillMode::HalfLength},
        {4, 5000, 5007, FillMode::RandomFull},
        {3, 6000, 6008, FillMode::HalfLength},
        {2, 8192, 8192, FillMode::VerySmall},
        {1, 0, 3, FillMode::RandomFull},
        {257, 1, 4, FillMode::VerySmall},
    };

    for (const FixedCase & cfg : fixed_cases) {
        if (!test_configuration(cfg.N, cfg.L, cfg.stride, cfg.mode, rng, true)) {
            all_passed = false;
            break;
        }
    }

    if (all_passed) {
        printf("\nRandomized tests:\n");
        std::uniform_int_distribution<uint32_t> n_dist(1u, 128u);
        std::uniform_int_distribution<uint32_t> n_small_dist(1u, 8u);
        std::uniform_int_distribution<uint32_t> len_small_dist(0u, 64u);
        std::uniform_int_distribution<uint32_t> len_large_dist(2048u, 12000u);
        std::uniform_int_distribution<uint32_t> extra_stride_dist(0u, 8u);
        std::uniform_int_distribution<int> mode_dist(0, 2);

        for (int t = 0; t < 30; ++t) {
            const bool large_case = (t >= 15);
            const uint32_t N = large_case ? n_small_dist(rng) : n_dist(rng);
            const uint32_t L = large_case ? len_large_dist(rng) : len_small_dist(rng);
            const uint32_t stride = L + extra_stride_dist(rng);
            const FillMode mode = static_cast<FillMode>(mode_dist(rng));
            if (!test_configuration(N, L, stride, mode, rng, true)) {
                all_passed = false;
                break;
            }
        }
    }

    printf("\n=== Benchmark Tests ===\n\n");
    benchmark_configuration(100000000u, 1u, FillMode::VerySmall);
    benchmark_configuration(50000000u, 8u, FillMode::VerySmall);
    benchmark_configuration(10000000u, 64u, FillMode::RandomFull);
    benchmark_configuration(1000000u, 1024u, FillMode::RandomFull);
    benchmark_configuration(1000000u, 1024u, FillMode::VerySmall);
    benchmark_configuration(1000000u, 1024u, FillMode::HalfLength);
    benchmark_configuration(4u, 262144u, FillMode::RandomFull);
    benchmark_configuration(4u, 262144u, FillMode::VerySmall);
    benchmark_configuration(4u, 262144u, FillMode::HalfLength);

    printf("\nSummary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}
