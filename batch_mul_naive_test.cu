#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include "batch_mul_naive.h"

namespace {

constexpr uint32_t kInputPadPatternA = 0xA5A5A5A5u;
constexpr uint32_t kInputPadPatternB = 0x5A5A5A5Au;
constexpr uint32_t kOutputPadPattern = 0xDEADBEEFu;

void check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        exit(1);
    }
}

uint32_t random_word() {
    return ((uint32_t)rand() << 16) ^ (uint32_t)rand();
}

void fill_random_operand(uint32_t* words, int N, int L, int stride, uint32_t pad_pattern) {
    for (int i = 0; i < N; ++i) {
        uint32_t* row = words + (size_t)i * stride;
        for (int j = 0; j < L; ++j) {
            row[j] = random_word();
        }
        for (int j = L; j < stride; ++j) {
            row[j] = pad_pattern;
        }
    }
}

void words_to_mpz(mpz_t result, const uint32_t* words, int L) {
    mpz_init(result);
    mpz_import(result, L, -1, sizeof(uint32_t), 0, 0, words);
}

bool verify_padding(const uint32_t* words, int N, int used, int stride, uint32_t pad_pattern) {
    for (int i = 0; i < N; ++i) {
        const uint32_t* row = words + (size_t)i * stride;
        for (int j = used; j < stride; ++j) {
            if (row[j] != pad_pattern) {
                return false;
            }
        }
    }
    return true;
}

bool test_configuration(int N, int L_a, int L_b, int stride_A, int stride_B, int stride_ret, bool verbose = false) {
    const int L_ret = L_a + L_b;
    if (verbose) {
        printf("Testing N=%d, L_a=%d, L_b=%d, stride_A=%d, stride_B=%d, stride_ret=%d...\n",
               N, L_a, L_b, stride_A, stride_B, stride_ret);
    }

    const size_t size_A = (size_t)N * stride_A * sizeof(uint32_t);
    const size_t size_B = (size_t)N * stride_B * sizeof(uint32_t);
    const size_t size_ret = (size_t)N * stride_ret * sizeof(uint32_t);

    uint32_t* h_A = (uint32_t*)malloc(size_A);
    uint32_t* h_B = (uint32_t*)malloc(size_B);
    uint32_t* h_ret = (uint32_t*)malloc(size_ret);

    fill_random_operand(h_A, N, L_a, stride_A, kInputPadPatternA);
    fill_random_operand(h_B, N, L_b, stride_B, kInputPadPatternB);
    for (int i = 0; i < N * stride_ret; ++i) {
        h_ret[i] = kOutputPadPattern;
    }

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    uint32_t* d_ret = nullptr;
    check_cuda(cudaMalloc(&d_A, size_A), "cudaMalloc(d_A)");
    check_cuda(cudaMalloc(&d_B, size_B), "cudaMalloc(d_B)");
    check_cuda(cudaMalloc(&d_ret, size_ret), "cudaMalloc(d_ret)");

    check_cuda(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice), "cudaMemcpy A H2D");
    check_cuda(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), "cudaMemcpy B H2D");
    check_cuda(cudaMemcpy(d_ret, h_ret, size_ret, cudaMemcpyHostToDevice), "cudaMemcpy ret H2D");

    batch_mul_naive(d_A, d_B, d_ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
    check_cuda(cudaGetLastError(), "batch_mul_naive launch");
    check_cuda(cudaDeviceSynchronize(), "batch_mul_naive sync");

    check_cuda(cudaMemcpy(h_A, d_A, size_A, cudaMemcpyDeviceToHost), "cudaMemcpy A D2H");
    check_cuda(cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost), "cudaMemcpy B D2H");
    check_cuda(cudaMemcpy(h_ret, d_ret, size_ret, cudaMemcpyDeviceToHost), "cudaMemcpy ret D2H");

    bool pass = true;
    for (int i = 0; i < N && pass; ++i) {
        mpz_t a, b, expected, actual;
        mpz_inits(a, b, expected, actual, NULL);

        words_to_mpz(a, h_A + (size_t)i * stride_A, L_a);
        words_to_mpz(b, h_B + (size_t)i * stride_B, L_b);
        words_to_mpz(actual, h_ret + (size_t)i * stride_ret, L_ret);
        mpz_mul(expected, a, b);

        if (mpz_cmp(expected, actual) != 0) {
            pass = false;
            if (verbose) {
                printf("  Value mismatch at batch index %d\n", i);
                gmp_printf("  A        = %Zx\n", a);
                gmp_printf("  B        = %Zx\n", b);
                gmp_printf("  Expected = %Zx\n", expected);
                gmp_printf("  Actual   = %Zx\n", actual);
            }
            mpz_clears(a, b, expected, actual, NULL);
            fprintf(stderr, "Halting on first mismatch.\n");
            exit(1);
        }

        mpz_clears(a, b, expected, actual, NULL);
    }

    if (pass && !verify_padding(h_A, N, L_a, stride_A, kInputPadPatternA)) {
        pass = false;
        if (verbose) {
            printf("  Input padding after A payload was modified\n");
        }
        fprintf(stderr, "Halting on first mismatch.\n");
        exit(1);
    }
    if (pass && !verify_padding(h_B, N, L_b, stride_B, kInputPadPatternB)) {
        pass = false;
        if (verbose) {
            printf("  Input padding after B payload was modified\n");
        }
        fprintf(stderr, "Halting on first mismatch.\n");
        exit(1);
    }
    if (pass && !verify_padding(h_ret, N, L_ret, stride_ret, kOutputPadPattern)) {
        pass = false;
        if (verbose) {
            printf("  Output padding after result payload was modified\n");
        }
        fprintf(stderr, "Halting on first mismatch.\n");
        exit(1);
    }

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }

    free(h_A);
    free(h_B);
    free(h_ret);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ret);

    return pass;
}

void benchmark_configuration(int L_a, int L_b, int stride_A, int stride_B, int stride_ret, long long target_words) {
    const int L_ret = L_a + L_b;
    int N = (int)(target_words / L_ret);
    if (N < 1) {
        N = 1;
    }

    printf("Benchmarking L_a=%d, L_b=%d, stride_A=%d, stride_B=%d, stride_ret=%d, N=%d...\n",
           L_a, L_b, stride_A, stride_B, stride_ret, N);

    const size_t size_A = (size_t)N * stride_A * sizeof(uint32_t);
    const size_t size_B = (size_t)N * stride_B * sizeof(uint32_t);
    const size_t size_ret = (size_t)N * stride_ret * sizeof(uint32_t);

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    uint32_t* d_ret = nullptr;
    check_cuda(cudaMalloc(&d_A, size_A), "benchmark cudaMalloc(d_A)");
    check_cuda(cudaMalloc(&d_B, size_B), "benchmark cudaMalloc(d_B)");
    check_cuda(cudaMalloc(&d_ret, size_ret), "benchmark cudaMalloc(d_ret)");

    check_cuda(cudaMemset(d_A, 0, size_A), "benchmark memset A");
    check_cuda(cudaMemset(d_B, 0, size_B), "benchmark memset B");
    check_cuda(cudaMemset(d_ret, 0, size_ret), "benchmark memset ret");

    batch_mul_naive(d_A, d_B, d_ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
    check_cuda(cudaGetLastError(), "benchmark launch");
    check_cuda(cudaDeviceSynchronize(), "benchmark warmup sync");

    const int iterations = 100;
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        batch_mul_naive(d_A, d_B, d_ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float milliseconds = 0.0f;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");

    const double avg_time_ms = milliseconds / iterations;
    const double muls_per_sec_m = (double)N / (avg_time_ms * 1e-3) / 1e6;
    const double bytes_moved = (double)N * (stride_A + stride_B + stride_ret) * sizeof(uint32_t);
    const double bandwidth_gbs = bytes_moved / (avg_time_ms * 1e-3) / 1e9;

    printf("  Average time: %.3f ms\n", avg_time_ms);
    printf("  Multiplications per second: %.2f million\n", muls_per_sec_m);
    printf("  Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ret);
}

void benchmark_direct_equivalent(int L, long long target_words) {
    benchmark_configuration(L, L, L, L, L * 2, target_words);
}

void run_fixed_correctness_tests(bool* all_passed) {
    struct Config {
        int N;
        int L_a;
        int L_b;
        int stride_A;
        int stride_B;
        int stride_ret;
    };

    const Config configs[] = {
        {1, 1, 1, 1, 1, 2},
        {7, 1, 63, 3, 64, 66},
        {13, 63, 1, 63, 4, 65},
        {33, 17, 15, 19, 16, 35},
        {65, 31, 33, 36, 40, 70},
        {96, 32, 32, 32, 35, 70},
        {127, 48, 16, 51, 21, 70},
        {257, 5, 59, 8, 64, 70},
        {3, 513, 511, 518, 517, 1029}
    };

    printf("Fixed configuration tests:\n");
    for (const Config& cfg : configs) {
        bool pass = test_configuration(cfg.N, cfg.L_a, cfg.L_b, cfg.stride_A, cfg.stride_B, cfg.stride_ret, true);
        if (!pass) {
            *all_passed = false;
        }
    }
    printf("\n");
}

void run_random_correctness_tests(const char* label, int iterations, int max_N, bool* all_passed) {
    printf("%s:\n", label);
    for (int t = 0; t < iterations; ++t) {
        int test_L_max = 2 << (rand() % 10);
        int L_a = 1 + rand() % (test_L_max - 1);
        int max_L_b = test_L_max - L_a;
        int L_b = 1 + rand() % max_L_b;
        int stride_A = L_a + rand() % 5;
        int stride_B = L_b + rand() % 5;
        int stride_ret = L_a + L_b + rand() % 5;
        int N = 1 + rand() % max_N;
        bool pass = test_configuration(N, L_a, L_b, stride_A, stride_B, stride_ret, true);
        if (!pass) {
            *all_passed = false;
        }
    }
    printf("\n");
}

void run_scaled_correctness_tests(const char* label, int iterations, int target_total_words, bool* all_passed) {
    printf("%s:\n", label);
    for (int t = 0; t < iterations; ++t) {
        int test_L_max = 64 << (rand() % 5);
        int L_a = 1 + rand() % (test_L_max - 1);
        int max_L_b = test_L_max - L_a;
        int L_b = 1 + rand() % max_L_b;
        int L_ret = L_a + L_b;
        int stride_A = L_a + rand() % 7;
        int stride_B = L_b + rand() % 7;
        int stride_ret = L_ret + rand() % 7;
        int max_N = target_total_words / L_ret;
        if (max_N < 1) {
            max_N = 1;
        }
        int N = 1 + rand() % max_N;
        bool pass = test_configuration(N, L_a, L_b, stride_A, stride_B, stride_ret, true);
        if (!pass) {
            *all_passed = false;
        }
    }
    printf("\n");
}

}  // namespace

int main() {
    srand(100);

    bool all_passed = true;

    printf("=== Correctness Tests ===\n\n");

    run_fixed_correctness_tests(&all_passed);
    run_random_correctness_tests("Small range tests (1 <= N <= 1024)", 32, 1024, &all_passed);
    run_scaled_correctness_tests("Medium range tests (N * (L_a + L_b) <= 2^20)", 32, 1 << 20, &all_passed);
    run_scaled_correctness_tests("Large range tests (N * (L_a + L_b) <= 1e7)", 10, 10000000, &all_passed);

    printf("=== Benchmark Tests ===\n\n");
    const long long target_words = 100000000LL;
    printf("Direct-test-equivalent benchmarks:\n");
    benchmark_direct_equivalent(1, target_words);
    benchmark_direct_equivalent(2, target_words);
    for (int L = 4; L <= 64; L += 4) {
        benchmark_direct_equivalent(L, target_words);
    }
    benchmark_direct_equivalent(128, target_words);
    benchmark_direct_equivalent(256, target_words);
    benchmark_direct_equivalent(512, target_words);
    printf("\n");

    printf("Asymmetric/stride benchmarks:\n");
    benchmark_configuration(16, 16, 16, 16, 32, target_words);
    benchmark_configuration(8, 24, 11, 29, 35, target_words);
    benchmark_configuration(31, 17, 36, 20, 52, target_words);
    benchmark_configuration(7, 57, 12, 60, 68, target_words);
    benchmark_configuration(40, 24, 44, 29, 68, target_words);

    printf("\n=== Summary ===\n");
    printf("All correctness tests: %s\n", all_passed ? "PASSED" : "FAILED");

    return all_passed ? 0 : 1;
}

/*
Benchmarking L_a=1, L_b=1, stride_A=1, stride_B=1, stride_ret=2, N=50000000...
  Average time: 0.569 ms
  Multiplications per second: 87804.23 million
  Effective bandwidth: 1404.87 GB/s
Benchmarking L_a=2, L_b=2, stride_A=2, stride_B=2, stride_ret=4, N=25000000...
  Average time: 0.541 ms
  Multiplications per second: 46203.09 million
  Effective bandwidth: 1478.50 GB/s
Benchmarking L_a=4, L_b=4, stride_A=4, stride_B=4, stride_ret=8, N=12500000...
  Average time: 0.594 ms
  Multiplications per second: 21051.93 million
  Effective bandwidth: 1347.32 GB/s
Benchmarking L_a=8, L_b=8, stride_A=8, stride_B=8, stride_ret=16, N=6250000...
  Average time: 0.905 ms
  Multiplications per second: 6903.61 million
  Effective bandwidth: 883.66 GB/s
Benchmarking L_a=12, L_b=12, stride_A=12, stride_B=12, stride_ret=24, N=4166666...
  Average time: 1.125 ms
  Multiplications per second: 3704.20 million
  Effective bandwidth: 711.21 GB/s
Benchmarking L_a=16, L_b=16, stride_A=16, stride_B=16, stride_ret=32, N=3125000...
  Average time: 1.656 ms
  Multiplications per second: 1886.78 million
  Effective bandwidth: 483.02 GB/s
Benchmarking L_a=20, L_b=20, stride_A=20, stride_B=20, stride_ret=40, N=2500000...
  Average time: 1.516 ms
  Multiplications per second: 1648.55 million
  Effective bandwidth: 527.53 GB/s
Benchmarking L_a=24, L_b=24, stride_A=24, stride_B=24, stride_ret=48, N=2083333...
  Average time: 1.421 ms
  Multiplications per second: 1465.96 million
  Effective bandwidth: 562.93 GB/s
Benchmarking L_a=28, L_b=28, stride_A=28, stride_B=28, stride_ret=56, N=1785714...
  Average time: 1.351 ms
  Multiplications per second: 1321.47 million
  Effective bandwidth: 592.02 GB/s
Benchmarking L_a=32, L_b=32, stride_A=32, stride_B=32, stride_ret=64, N=1562500...
  Average time: 1.290 ms
  Multiplications per second: 1211.08 million
  Effective bandwidth: 620.07 GB/s
Benchmarking L_a=36, L_b=36, stride_A=36, stride_B=36, stride_ret=72, N=1388888...
  Average time: 1.303 ms
  Multiplications per second: 1066.00 million
  Effective bandwidth: 614.02 GB/s
Benchmarking L_a=40, L_b=40, stride_A=40, stride_B=40, stride_ret=80, N=1250000...
  Average time: 1.366 ms
  Multiplications per second: 915.36 million
  Effective bandwidth: 585.83 GB/s
Benchmarking L_a=44, L_b=44, stride_A=44, stride_B=44, stride_ret=88, N=1136363...
  Average time: 1.322 ms
  Multiplications per second: 859.42 million
  Effective bandwidth: 605.03 GB/s
Benchmarking L_a=48, L_b=48, stride_A=48, stride_B=48, stride_ret=96, N=1041666...
  Average time: 1.284 ms
  Multiplications per second: 811.35 million
  Effective bandwidth: 623.11 GB/s
Benchmarking L_a=52, L_b=52, stride_A=52, stride_B=52, stride_ret=104, N=961538...
  Average time: 1.304 ms
  Multiplications per second: 737.17 million
  Effective bandwidth: 613.32 GB/s
Benchmarking L_a=56, L_b=56, stride_A=56, stride_B=56, stride_ret=112, N=892857...
  Average time: 1.272 ms
  Multiplications per second: 701.92 million
  Effective bandwidth: 628.92 GB/s
Benchmarking L_a=60, L_b=60, stride_A=60, stride_B=60, stride_ret=120, N=833333...
  Average time: 1.258 ms
  Multiplications per second: 662.19 million
  Effective bandwidth: 635.70 GB/s
Benchmarking L_a=64, L_b=64, stride_A=64, stride_B=64, stride_ret=128, N=781250...
  Average time: 1.220 ms
  Multiplications per second: 640.33 million
  Effective bandwidth: 655.70 GB/s
Benchmarking L_a=128, L_b=128, stride_A=128, stride_B=128, stride_ret=256, N=390625...
  Average time: 2.122 ms
  Multiplications per second: 184.11 million
  Effective bandwidth: 377.05 GB/s
Benchmarking L_a=256, L_b=256, stride_A=256, stride_B=256, stride_ret=512, N=195312...
  Average time: 4.133 ms
  Multiplications per second: 47.25 million
  Effective bandwidth: 193.54 GB/s
Benchmarking L_a=512, L_b=512, stride_A=512, stride_B=512, stride_ret=1024, N=97656...
  Average time: 10.913 ms
  Multiplications per second: 8.95 million
  Effective bandwidth: 73.31 GB/s
*/
