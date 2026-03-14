#include <cuda_runtime.h>
#include <curand.h>
#include <gmp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "batch_bitlength.h"

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

enum class FillMode {
    RandomFull,
    VerySmall,
    HalfLength
};

__global__ void shape_input_kernel(
    uint32_t * words,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    FillMode mode
) {
    const size_t total_words = (size_t)N * stride;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_words;
         idx += (size_t)gridDim.x * blockDim.x) {
        const uint32_t col = (uint32_t)(idx % stride);
        uint32_t value = words[idx];

        if (col >= L) {
            words[idx] = kInputPadPattern;
            continue;
        }

        switch (mode) {
            case FillMode::RandomFull:
                break;
            case FillMode::VerySmall: {
                const uint32_t row = (uint32_t)(idx / stride);
                if (col == 0u) {
                    words[idx] = (value & 0xffffu) + 1u;
                } else if (col == 1u && (row & 1u) != 0u) {
                    words[idx] = value & 0xffu;
                } else {
                    words[idx] = 0u;
                }
                break;
            }
            case FillMode::HalfLength: {
                const uint32_t top = (L == 0u) ? 0u : (L - 1u) / 2u;
                if (col < top) {
                    break;
                }
                if (col == top) {
                    words[idx] = 1u << (value % 31u);
                } else {
                    words[idx] = 0u;
                }
                break;
            }
        }
    }
}

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

void fill_input_device(
    uint32_t * d_words,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    FillMode mode,
    uint64_t seed
) {
    const size_t total_words = (size_t)N * stride;
    if (total_words == 0) {
        return;
    }

    curandGenerator_t rand_gen = nullptr;
    CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, seed));
    CURAND_CHECK(curandSetGeneratorOffset(rand_gen, 0ULL));
    CURAND_CHECK(curandGenerate(rand_gen, d_words, total_words));

    const uint32_t threads_per_block = 256u;
    const uint32_t num_blocks = std::min<uint32_t>(
        (uint32_t)((total_words + threads_per_block - 1u) / threads_per_block),
        65535u);
    shape_input_kernel<<<num_blocks, threads_per_block>>>(d_words, N, L, stride, mode);
    CUDA_CHECK(cudaGetLastError());

    CURAND_CHECK(curandDestroyGenerator(rand_gen));
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
    uint64_t seed,
    bool verbose
) {
    if (verbose) {
        printf("Testing N=%u, L=%u, stride=%u, mode=%s...\n", N, L, stride, fill_mode_name(mode));
    }

    std::vector<uint32_t> h_A((size_t)N * stride);

    uint32_t * d_A = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_bitlength_workspace_size(N, L);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }
    fill_input_device(d_A, N, L, stride, mode, seed);
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A, cudaMemcpyDeviceToHost));

    uint32_t expected = 0u;
    for (uint32_t i = 0; i < N; ++i) {
        expected = std::max(expected, gmp_bitlength(h_A.data() + (size_t)i * stride, L));
    }

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
    FillMode mode,
    uint64_t seed
) {
    const uint32_t stride = L;
    uint32_t * d_A = nullptr;
    uint32_t * d_workspace = nullptr;
    const size_t size_A = (size_t)N * stride * sizeof(uint32_t);
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
    fill_input_device(d_A, N, L, stride, mode, seed);

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
    bool all_passed = true;
    uint64_t seed = 123456789ull;

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
        if (!test_configuration(cfg.N, cfg.L, cfg.stride, cfg.mode, seed++, true)) {
            all_passed = false;
            break;
        }
    }

    if (all_passed) {
        printf("\nRandomized tests:\n");
        for (int t = 0; t < 30; ++t) {
            const bool large_case = (t >= 15);
            const uint32_t N = large_case ? (1u + (uint32_t)(seed % 8u)) : (1u + (uint32_t)(seed % 128u));
            seed = seed * 0x9e3779b97f4a7c15ULL + 0xbf58476d1ce4e5b9ULL;
            const uint32_t L = large_case ? (2048u + (uint32_t)(seed % 9953u)) : (uint32_t)(seed % 65u);
            seed = seed * 0x9e3779b97f4a7c15ULL + 0xbf58476d1ce4e5b9ULL;
            const uint32_t stride = L + (uint32_t)(seed % 9u);
            const FillMode mode = static_cast<FillMode>(seed % 3u);
            seed = seed * 0x9e3779b97f4a7c15ULL + 0xbf58476d1ce4e5b9ULL;
            if (!test_configuration(N, L, stride, mode, seed++, true)) {
                all_passed = false;
                break;
            }
        }
    }

    printf("\n=== Benchmark Tests ===\n\n");
    benchmark_configuration(100000000u, 1u, FillMode::VerySmall, seed++);
    benchmark_configuration(100000000u, 2u, FillMode::VerySmall, seed++);
    benchmark_configuration(100000000u, 4u, FillMode::VerySmall, seed++);
    benchmark_configuration(50000000u, 8u, FillMode::VerySmall, seed++);
    benchmark_configuration(10000000u, 64u, FillMode::RandomFull, seed++);
    benchmark_configuration(1000000u, 1024u, FillMode::RandomFull, seed++);
    benchmark_configuration(1000000u, 1024u, FillMode::VerySmall, seed++);
    benchmark_configuration(1000000u, 1024u, FillMode::HalfLength, seed++);
    benchmark_configuration(4u, 262144u, FillMode::RandomFull, seed++);
    benchmark_configuration(4u, 262144u, FillMode::VerySmall, seed++);
    benchmark_configuration(4u, 262144u, FillMode::HalfLength, seed++);

    printf("\nSummary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}

/*
Benchmarking N=100000000, L=1, mode=very-small...
  params size  400M, workspace size    0M Average time:  0.799 ms Throughput: 125084.657 Mnum/s Bandwidth: 500.34 GB/s Result=17
Benchmarking N=100000000, L=2, mode=very-small...
  params size  800M, workspace size    0M Average time:  1.028 ms Throughput: 97237.758 Mnum/s Bandwidth: 777.90 GB/s Result=40
Benchmarking N=100000000, L=4, mode=very-small...
  params size 1600M, workspace size    0M Average time:  1.505 ms Throughput: 66435.642 Mnum/s Bandwidth: 1062.97 GB/s Result=40
Benchmarking N=50000000, L=8, mode=very-small...
  params size 1600M, workspace size    0M Average time:  1.537 ms Throughput: 32530.790 Mnum/s Bandwidth: 1040.99 GB/s Result=40
Benchmarking N=10000000, L=64, mode=random-full...
  params size 2560M, workspace size    0M Average time:  1.251 ms Throughput: 7994.855 Mnum/s Bandwidth: 2046.68 GB/s Result=2048
Benchmarking N=1000000, L=1024, mode=random-full...
  params size 4096M, workspace size    0M Average time:  0.705 ms Throughput: 1418.825 Mnum/s Bandwidth: 5811.51 GB/s Result=32768
Benchmarking N=1000000, L=1024, mode=very-small...
  params size 4096M, workspace size    0M Average time:  2.915 ms Throughput:  343.014 Mnum/s Bandwidth: 1404.99 GB/s Result=40
Benchmarking N=1000000, L=1024, mode=half-length...
  params size 4096M, workspace size    0M Average time:  1.773 ms Throughput:  563.953 Mnum/s Bandwidth: 2309.95 GB/s Result=16383
Benchmarking N=4, L=262144, mode=random-full...
  params size    4M, workspace size    0M Average time:  0.497 ms Throughput:    0.008 Mnum/s Bandwidth:   8.45 GB/s Result=8388608
Benchmarking N=4, L=262144, mode=very-small...
  params size    4M, workspace size    0M Average time:  1.807 ms Throughput:    0.002 Mnum/s Bandwidth:   2.32 GB/s Result=40
Benchmarking N=4, L=262144, mode=half-length...
  params size    4M, workspace size    0M Average time:  1.159 ms Throughput:    0.003 Mnum/s Bandwidth:   3.62 GB/s Result=4194291
*/