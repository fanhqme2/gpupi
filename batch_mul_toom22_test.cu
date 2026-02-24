#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include "batch_mul_toom22.h"
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

    // Allocate workspace
    size_t workspace_size = batch_mul_toom22_workspace_size(N, L);
    uint32_t* d_workspace = NULL;
    if (workspace_size > 0) {
        cudaMalloc(&d_workspace, workspace_size);
    }

    // Run kernel
    batch_mul_toom22(d_A, d_B, d_ret, d_workspace, N, L);

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
    if (d_workspace) cudaFree(d_workspace);

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

    // Allocate workspace
    size_t workspace_size = batch_mul_toom22_workspace_size(N, L);
    uint32_t* d_workspace = NULL;
    if (workspace_size > 0) {
        cudaMalloc(&d_workspace, workspace_size);
    }

    // Warmup
    batch_mul_toom22(d_A, d_B, d_ret, d_workspace, N, L);
    cudaDeviceSynchronize();

    // Benchmark
    const int iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        batch_mul_toom22(d_A, d_B, d_ret, d_workspace, N, L);
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
    if (d_workspace) cudaFree(d_workspace);
}

int main() {
    srand(100);

    bool all_passed = true;

    printf("=== Correctness Tests ===\n\n");

    // Small size tests: N*L <= 16384, L in [BATCH_MUL_DIRECT_L_MAX+1, BATCH_MUL_TOOM22_L_MAX]
    printf("Small size tests (N*L <= 16384, L > %d):\n", BATCH_MUL_DIRECT_L_MAX);
    for (int t = 0; t < 20; t++) {
        int L_min = BATCH_MUL_DIRECT_L_MAX + 1;
        int L = L_min + rand() % (BATCH_MUL_TOOM22_L_MAX - L_min + 1);
        int max_N = 16384 / L;
        if (max_N < 1) max_N = 1;
        int N = 1 + rand() % max_N;
        bool pass = test_configuration(N, L, true);
        if (!pass) all_passed = false;
    }
    printf("\n");

    // Medium range tests
    printf("Medium range tests (N*L <= 2^20, L > %d):\n", BATCH_MUL_DIRECT_L_MAX);
    for (int t = 0; t < 10; t++) {
        int L_min = BATCH_MUL_DIRECT_L_MAX + 1;
        int L = L_min + rand() % (BATCH_MUL_TOOM22_L_MAX - L_min + 1);
        int max_N = (1 << 20) / L;
        if (max_N < 1) max_N = 1;
        int N = 1 + rand() % max_N;
        bool pass = test_configuration(N, L, true);
        if (!pass) all_passed = false;
    }
    printf("\n");

    // Large range tests
    printf("Large range tests (N*L <= 2e7, L > %d):\n", BATCH_MUL_DIRECT_L_MAX);
    for (int t = 0; t < 3; t++) {
        int L_min = BATCH_MUL_DIRECT_L_MAX + 1;
        int L = L_min + rand() % (BATCH_MUL_TOOM22_L_MAX - L_min + 1);
        int max_N = 2e7 / L;
        if (max_N < 1) max_N = 1;
        int N = max_N;
        bool pass = test_configuration(N, L, true);
        if (!pass) all_passed = false;
    }
    printf("\n");

    printf("=== Benchmark Tests ===\n\n");
    long long target = 100000000LL; // 1e8
    
    // Benchmark with specified L values: 481, 241, 121, 65
    int benchmark_L[] = {65, 121, 241, 481};
    int num_benchmarks = sizeof(benchmark_L) / sizeof(benchmark_L[0]);
    
    for (int i = 0; i < num_benchmarks; i++) {
        int L = benchmark_L[i];
        if (L <= BATCH_MUL_TOOM22_L_MAX && L > BATCH_MUL_DIRECT_L_MAX) {
            benchmark(L, target);
        }
    }

    printf("\n=== Summary ===\n");
    printf("All correctness tests: %s\n", all_passed ? "PASSED" : "FAILED");

    return all_passed ? 0 : 1;
}

/*
Benchmarking L=65, N=1538461 (N*L=99999965)...
  Average time: 2.953 ms
  Multiplications per second: 520.98 million
  Effective bandwidth: 270.91 GB/s
Benchmarking L=121, N=826446 (N*L=99999966)...
  Average time: 2.511 ms
  Multiplications per second: 329.07 million
  Effective bandwidth: 318.54 GB/s
Benchmarking L=241, N=414937 (N*L=99999817)...
  Average time: 4.998 ms
  Multiplications per second: 83.02 million
  Effective bandwidth: 160.07 GB/s
Benchmarking L=481, N=207900 (N*L=99999900)...
  Average time: 9.764 ms
  Multiplications per second: 21.29 million
  Effective bandwidth: 81.93 GB/s
*/