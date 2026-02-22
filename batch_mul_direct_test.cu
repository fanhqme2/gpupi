#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include "batch_mul_direct.h"

// Helper to generate random L words integer (in host memory)
void generate_random_words(uint32_t* words, int L) {
    for (int i = 0; i < L; i++) {
        words[i] = ((uint32_t)rand() << 16) | (uint32_t)rand();
    }
}

// Convert L words to GMP mpz_t
void words_to_mpz(mpz_t result, uint32_t* words, int L) {
    mpz_init(result);
    // Words are stored in little-endian order (word 0 is least significant)
    mpz_import(result, L, -1, sizeof(uint32_t), 0, 0, words);
}

// Test a single configuration
bool test_configuration(int N, int L, bool verbose = false) {
    if (verbose) {
        printf("Testing N=%d, L=%d (N*L=%d)...\n", N, L, N*L);
    }

    size_t size_A = (size_t)N * L * sizeof(uint32_t);
    size_t size_B = (size_t)N * L * sizeof(uint32_t);
    size_t size_ret = (size_t)N * (L * 2) * sizeof(uint32_t);

    // Host memory
    uint32_t* h_A = (uint32_t*)malloc(size_A);
    uint32_t* h_B = (uint32_t*)malloc(size_B);
    uint32_t* h_ret = (uint32_t*)malloc(size_ret);

    // Initialize random data
    for (int i = 0; i < N * L; i++) {
        h_A[i] = ((uint32_t)rand() << 16) | (uint32_t)rand();
        h_B[i] = ((uint32_t)rand() << 16) | (uint32_t)rand();
    }

    // Device memory
    uint32_t* d_A;
    uint32_t* d_B;
    uint32_t* d_ret;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_ret, size_ret);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Run kernel
    batch_mul_direct(d_A, d_B, d_ret, N, L);

    // Copy back results
    cudaMemcpy(h_ret, d_ret, size_ret, cudaMemcpyDeviceToHost);

    // Verify with GMP
    bool pass = true;
    for (int i = 0; i < N && pass; i++) {
        mpz_t a, b, expected, result;
        mpz_inits(a, b, expected, result, NULL);

        // Load A and B
        words_to_mpz(a, &h_A[i * L], L);
        words_to_mpz(b, &h_B[i * L], L);

        // Compute expected
        mpz_mul(expected, a, b);

        // Get result from kernel and build mpz_t
        uint32_t* ret_words = &h_ret[i * L * 2];
        words_to_mpz(result, ret_words, L * 2);

        // Compare
        if (mpz_cmp(expected, result) != 0) {
            pass = false;
            if (verbose) {
                printf("  Mismatch at index %d\n", i);
                gmp_printf("  A = %Zx\n", a);
                gmp_printf("  B = %Zx\n", b);
                gmp_printf("  Expected = %Zx\n", expected);
                gmp_printf("  Got      = %Zx\n", result);
            }
        }

        mpz_clears(a, b, expected, result, NULL);
    }

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
        if (!pass){
            //show cuda last error
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("  CUDA error: %s\n", cudaGetErrorString(err));
            }
            exit(0);
        }
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_ret);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ret);

    return pass;
}

// Benchmark a configuration
void benchmark(int L, long long total_elements) {
    int N = (int)(total_elements / L);
    
    printf("Benchmarking L=%d, N=%d (N*L=%lld)...\n", L, N, (long long)N * L);

    size_t size_A = (size_t)N * L * sizeof(uint32_t);
    size_t size_B = (size_t)N * L * sizeof(uint32_t);
    size_t size_ret = (size_t)N * (L * 2) * sizeof(uint32_t);

    // Allocate and initialize
    uint32_t* d_A;
    uint32_t* d_B;
    uint32_t* d_ret;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_ret, size_ret);

    // Warmup
    batch_mul_direct(d_A, d_B, d_ret, N, L);
    cudaDeviceSynchronize();

    // Benchmark
    const int iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        batch_mul_direct(d_A, d_B, d_ret, N, L);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double avg_time = milliseconds / iterations;
    double total_words = (double)N * L * 2; // A + B words read
    double bandwidth = (total_words * sizeof(uint32_t)) / (avg_time * 1e-3) / 1e9; // GB/s

    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Multiplications per second: %.2f million\n", (double)N / (avg_time * 1e-3) / 1e6);
    printf("  Effective bandwidth: %.2f GB/s\n", bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ret);
}

int main() {
    srand(100);

    bool all_passed = true;

    printf("=== Correctness Tests ===\n\n");

    // Small range: 1 <= N <= 1024
    printf("Small range tests (1 <= N <= 1024):\n");
    for (int t = 0; t < 10; t++) {
        int L = 1 + rand() % BATCH_MUL_DIRECT_L_MAX;
        int N = 1 + rand() % 1024;
        bool pass = test_configuration(N, L, true);
        if (!pass) all_passed = false;
    }
    printf("\n");

    // Medium range: N * L <= 2^20
    printf("Medium range tests (N * L <= 2^20):\n");
    for (int t = 0; t < 10; t++) {
        int L = 1 + rand() % BATCH_MUL_DIRECT_L_MAX;
        int max_N = (1 << 20) / L;
        if (max_N < 1) max_N = 1;
        int N = 1 + rand() % max_N;
        bool pass = test_configuration(N, L, true);
        if (!pass) all_passed = false;
    }
    printf("\n");

    // Large range: N * L >= 1e7
    printf("Large range tests (N * L >= 1e7):\n");
    for (int t = 0; t < 3; t++) {
        int L = 1 + rand() % BATCH_MUL_DIRECT_L_MAX;
        int min_N = (int)(1e7 / L) + 1;
        int N = min_N + rand() % 1000;
        bool pass = test_configuration(N, L, true);
        if (!pass) all_passed = false;
    }
    printf("\n");

    printf("=== Benchmark Tests ===\n\n");
    long long target = 100000000LL; // 1e8
    for (int l = 16; l <= 64; l += 4){
        benchmark(l, target);
    }

    printf("\n=== Summary ===\n");
    printf("All correctness tests: %s\n", all_passed ? "PASSED" : "FAILED");

    return all_passed ? 0 : 1;
}

/*
Benchmarking L=16, N=6250000 (N*L=100000000)...
  Average time: 1.171 ms
  Multiplications per second: 5336.23 million
  Effective bandwidth: 683.04 GB/s
Benchmarking L=20, N=5000000 (N*L=100000000)...
  Average time: 2.027 ms
  Multiplications per second: 2466.11 million
  Effective bandwidth: 394.58 GB/s
Benchmarking L=24, N=4166666 (N*L=99999984)...
  Average time: 1.729 ms
  Multiplications per second: 2409.19 million
  Effective bandwidth: 462.56 GB/s
Benchmarking L=28, N=3571428 (N*L=99999984)...
  Average time: 1.579 ms
  Multiplications per second: 2262.10 million
  Effective bandwidth: 506.71 GB/s
Benchmarking L=32, N=3125000 (N*L=100000000)...
  Average time: 1.368 ms
  Multiplications per second: 2285.05 million
  Effective bandwidth: 584.97 GB/s
Benchmarking L=36, N=2777777 (N*L=99999972)...
  Average time: 1.970 ms
  Multiplications per second: 1410.10 million
  Effective bandwidth: 406.11 GB/s
Benchmarking L=40, N=2500000 (N*L=100000000)...
  Average time: 2.051 ms
  Multiplications per second: 1218.89 million
  Effective bandwidth: 390.05 GB/s
Benchmarking L=44, N=2272727 (N*L=99999988)...
  Average time: 1.951 ms
  Multiplications per second: 1164.66 million
  Effective bandwidth: 409.96 GB/s
Benchmarking L=48, N=2083333 (N*L=99999984)...
  Average time: 1.888 ms
  Multiplications per second: 1103.27 million
  Effective bandwidth: 423.65 GB/s
Benchmarking L=52, N=1923076 (N*L=99999952)...
  Average time: 1.914 ms
  Multiplications per second: 1004.86 million
  Effective bandwidth: 418.02 GB/s
Benchmarking L=56, N=1785714 (N*L=99999984)...
  Average time: 1.893 ms
  Multiplications per second: 943.48 million
  Effective bandwidth: 422.68 GB/s
Benchmarking L=60, N=1666666 (N*L=99999960)...
  Average time: 1.820 ms
  Multiplications per second: 915.76 million
  Effective bandwidth: 439.57 GB/s
Benchmarking L=64, N=1562500 (N*L=100000000)...
  Average time: 1.755 ms
  Multiplications per second: 890.19 million
  Effective bandwidth: 455.78 GB/s
*/