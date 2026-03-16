#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <curand.h>
#include <gmp.h>

#include "batch_mul_ntt.h"

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

constexpr uint32_t QUICK_CHECK_MOD = 10007u;
constexpr uint32_t QUICK_CHECK_THREADS = 256u;

static void words_to_mpz(mpz_t out, const uint32_t* words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

struct QuickCheckFailure {
    uint32_t index;
    uint32_t expected;
    uint32_t got;
};

__global__ static void reduce_words_mod_chunks_kernel(
    const uint32_t* words,
    uint32_t stride_words,
    uint32_t words_per_number,
    uint32_t chunk_count,
    const uint32_t* chunk_base_pow_mod,
    const uint32_t* local_pow_mod,
    uint32_t* partials
) {
    __shared__ uint32_t shared[QUICK_CHECK_THREADS];

    const uint32_t block = blockIdx.x;
    const uint32_t number_idx = block / chunk_count;
    const uint32_t chunk_idx = block % chunk_count;
    const uint32_t chunk_start = chunk_idx * QUICK_CHECK_THREADS;
    const size_t base_offset = (size_t)number_idx * stride_words + chunk_start;

    uint32_t thread_sum = 0;
    for (uint32_t i = threadIdx.x; i < QUICK_CHECK_THREADS; i += blockDim.x) {
        const uint32_t word_idx = chunk_start + i;
        if (word_idx < words_per_number) {
            const uint32_t scaled =
                (uint32_t)(((uint64_t)(words[base_offset + i] % QUICK_CHECK_MOD) * local_pow_mod[i]) % QUICK_CHECK_MOD);
            const uint32_t weighted =
                (uint32_t)(((uint64_t)scaled * chunk_base_pow_mod[chunk_idx]) % QUICK_CHECK_MOD);
            thread_sum += weighted;
            if (thread_sum >= QUICK_CHECK_MOD) thread_sum -= QUICK_CHECK_MOD;
        }
    }

    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint32_t v = shared[threadIdx.x] + shared[threadIdx.x + stride];
            if (v >= QUICK_CHECK_MOD) v -= QUICK_CHECK_MOD;
            shared[threadIdx.x] = v;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partials[(size_t)number_idx * chunk_count + chunk_idx] = shared[0];
    }
}

__global__ static void reduce_partials_mod_chunks_kernel(
    const uint32_t* input,
    uint32_t values_per_number,
    uint32_t chunk_count,
    uint32_t* output
) {
    __shared__ uint32_t shared[QUICK_CHECK_THREADS];

    const uint32_t block = blockIdx.x;
    const uint32_t number_idx = block / chunk_count;
    const uint32_t chunk_idx = block % chunk_count;
    const uint32_t chunk_start = chunk_idx * QUICK_CHECK_THREADS;
    const size_t base_offset = (size_t)number_idx * values_per_number + chunk_start;

    uint32_t thread_sum = 0;
    for (uint32_t i = threadIdx.x; i < QUICK_CHECK_THREADS; i += blockDim.x) {
        const uint32_t value_idx = chunk_start + i;
        if (value_idx < values_per_number) {
            thread_sum += input[base_offset + i];
            if (thread_sum >= QUICK_CHECK_MOD) thread_sum -= QUICK_CHECK_MOD;
        }
    }

    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            uint32_t v = shared[threadIdx.x] + shared[threadIdx.x + stride];
            if (v >= QUICK_CHECK_MOD) v -= QUICK_CHECK_MOD;
            shared[threadIdx.x] = v;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[(size_t)number_idx * chunk_count + chunk_idx] = shared[0];
    }
}

__global__ static void compare_products_mod_kernel(
    const uint32_t* a_mod,
    const uint32_t* b_mod,
    const uint32_t* ret_mod,
    uint32_t N,
    QuickCheckFailure* failure
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const uint32_t expected = (uint32_t)(((uint64_t)a_mod[i] * b_mod[i]) % QUICK_CHECK_MOD);
        if (expected != ret_mod[i]) {
            if (atomicCAS(&failure->index, 0xffffffffu, i) == 0xffffffffu) {
                failure->expected = expected;
                failure->got = ret_mod[i];
            }
        }
    }
}

static void reduce_numbers_mod_10007_cuda(
    const uint32_t* d_words,
    uint32_t N,
    uint32_t words_per_number,
    uint32_t stride_words,
    const uint32_t* d_chunk_base_pow_mod,
    const uint32_t* d_local_pow_mod,
    uint32_t* d_stage0,
    uint32_t* d_stage1,
    uint32_t* d_out
) {
    uint32_t current_count = (words_per_number + QUICK_CHECK_THREADS - 1) / QUICK_CHECK_THREADS;
    const uint32_t initial_blocks = N * current_count;
    reduce_words_mod_chunks_kernel<<<initial_blocks, QUICK_CHECK_THREADS>>>(
        d_words, stride_words, words_per_number, current_count, d_chunk_base_pow_mod, d_local_pow_mod, d_stage0);
    CUDA_CHECK(cudaGetLastError());

    uint32_t* current = d_stage0;
    uint32_t* next = d_stage1;
    while (current_count > 1) {
        const uint32_t next_count = (current_count + QUICK_CHECK_THREADS - 1) / QUICK_CHECK_THREADS;
        const uint32_t blocks = N * next_count;
        reduce_partials_mod_chunks_kernel<<<blocks, QUICK_CHECK_THREADS>>>(
            current, current_count, next_count, next);
        CUDA_CHECK(cudaGetLastError());
        current = next;
        next = (next == d_stage1) ? d_stage0 : d_stage1;
        current_count = next_count;
    }

    CUDA_CHECK(cudaMemcpy(d_out, current, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
}

static void quick_check_products_mod_10007_cuda(
    const uint32_t* d_A,
    const uint32_t* d_B,
    const uint32_t* d_ret,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_ret
) {
    const uint32_t L = L_a + L_b;
    const uint32_t max_words = std::max(L, std::max(L_a, L_b));
    const uint32_t chunk_capacity = (max_words + QUICK_CHECK_THREADS - 1) / QUICK_CHECK_THREADS;
    const size_t stage_size = (size_t)N * chunk_capacity * sizeof(uint32_t);

    std::vector<uint32_t> h_chunk_base_pow_mod(chunk_capacity);
    std::vector<uint32_t> h_local_pow_mod(QUICK_CHECK_THREADS);
    const uint32_t base_mod = (uint32_t)((1ull << 32) % QUICK_CHECK_MOD);
    h_local_pow_mod[0] = 1;
    for (uint32_t i = 1; i < QUICK_CHECK_THREADS; ++i) {
        h_local_pow_mod[i] = (uint32_t)(((uint64_t)h_local_pow_mod[i - 1] * base_mod) % QUICK_CHECK_MOD);
    }
    const uint32_t chunk_stride_pow =
        (uint32_t)(((uint64_t)h_local_pow_mod[QUICK_CHECK_THREADS - 1] * base_mod) % QUICK_CHECK_MOD);
    h_chunk_base_pow_mod[0] = 1;
    for (uint32_t i = 1; i < chunk_capacity; ++i) {
        h_chunk_base_pow_mod[i] =
            (uint32_t)(((uint64_t)h_chunk_base_pow_mod[i - 1] * chunk_stride_pow) % QUICK_CHECK_MOD);
    }

    uint32_t* d_chunk_base_pow_mod = nullptr;
    uint32_t* d_local_pow_mod = nullptr;
    uint32_t* d_stage0 = nullptr;
    uint32_t* d_stage1 = nullptr;
    uint32_t* d_a_mod = nullptr;
    uint32_t* d_b_mod = nullptr;
    uint32_t* d_ret_mod = nullptr;
    QuickCheckFailure* d_failure = nullptr;

    CUDA_CHECK(cudaMalloc(&d_chunk_base_pow_mod, (size_t)chunk_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_local_pow_mod, (size_t)QUICK_CHECK_THREADS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_stage0, stage_size));
    CUDA_CHECK(cudaMalloc(&d_stage1, stage_size));
    CUDA_CHECK(cudaMalloc(&d_a_mod, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b_mod, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ret_mod, (size_t)N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_failure, sizeof(QuickCheckFailure)));

    CUDA_CHECK(cudaMemcpy(d_chunk_base_pow_mod, h_chunk_base_pow_mod.data(),
                          (size_t)chunk_capacity * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_pow_mod, h_local_pow_mod.data(),
                          (size_t)QUICK_CHECK_THREADS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_failure, 0xff, sizeof(QuickCheckFailure)));

    reduce_numbers_mod_10007_cuda(d_A, N, L_a, stride_A, d_chunk_base_pow_mod, d_local_pow_mod, d_stage0, d_stage1, d_a_mod);
    reduce_numbers_mod_10007_cuda(d_B, N, L_b, stride_B, d_chunk_base_pow_mod, d_local_pow_mod, d_stage0, d_stage1, d_b_mod);
    reduce_numbers_mod_10007_cuda(d_ret, N, L, stride_ret, d_chunk_base_pow_mod, d_local_pow_mod, d_stage0, d_stage1, d_ret_mod);

    const uint32_t compare_blocks = (N + QUICK_CHECK_THREADS - 1) / QUICK_CHECK_THREADS;
    compare_products_mod_kernel<<<compare_blocks, QUICK_CHECK_THREADS>>>(d_a_mod, d_b_mod, d_ret_mod, N, d_failure);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    QuickCheckFailure h_failure;
    CUDA_CHECK(cudaMemcpy(&h_failure, d_failure, sizeof(QuickCheckFailure), cudaMemcpyDeviceToHost));
    if (h_failure.index != 0xffffffffu) {
        printf("  WARNING: quick correctness check failed at index %u (mod 10007: expected %u, got %u)\n",
               h_failure.index, h_failure.expected, h_failure.got);
    }

    CUDA_CHECK(cudaFree(d_chunk_base_pow_mod));
    CUDA_CHECK(cudaFree(d_local_pow_mod));
    CUDA_CHECK(cudaFree(d_stage0));
    CUDA_CHECK(cudaFree(d_stage1));
    CUDA_CHECK(cudaFree(d_a_mod));
    CUDA_CHECK(cudaFree(d_b_mod));
    CUDA_CHECK(cudaFree(d_ret_mod));
    CUDA_CHECK(cudaFree(d_failure));
}

static uint32_t floor_log2_u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << (r + 1)) <= x && r < 30) ++r;
    return r;
}

static uint32_t sample_log_uniform_u32(uint32_t lo, uint32_t hi, std::mt19937_64& rng) {
    uint32_t min_bin = floor_log2_u32(lo);
    uint32_t max_bin = floor_log2_u32(hi);
    std::uniform_int_distribution<uint32_t> bin_dist(min_bin, max_bin);
    uint32_t b = bin_dist(rng);
    uint32_t b_lo = std::max(lo, 1u << b);
    uint32_t b_hi = std::min(hi, (b == 31) ? 0xffffffffu : ((1u << (b + 1)) - 1u));
    std::uniform_int_distribution<uint32_t> val_dist(b_lo, b_hi);
    return val_dist(rng);
}

static bool test_configuration(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    NTTPrecomputedTables tables,
    std::mt19937_64& rng,
    bool verbose = false
) {
    const uint32_t L = L_a + L_b;
    std::bernoulli_distribution packed_dist(0.5);
    std::uniform_int_distribution<uint32_t> extra_stride_dist(1u, 8u);
    const uint32_t stride_A = packed_dist(rng) ? L_a : (L_a + extra_stride_dist(rng));
    const uint32_t stride_B = packed_dist(rng) ? L_b : (L_b + extra_stride_dist(rng));
    const uint32_t stride_ret = packed_dist(rng) ? L : (L + extra_stride_dist(rng));
    const size_t size_A_words = (size_t)N * stride_A;
    const size_t size_B_words = (size_t)N * stride_B;
    const size_t size_ret_words = (size_t)N * stride_ret;

    if (verbose) {
        printf("Testing N=%u, L_a=%u, L_b=%u, stride_A=%u, stride_B=%u, stride_ret=%u...\n",
               N, L_a, L_b, stride_A, stride_B, stride_ret);
    }

    std::vector<uint32_t> h_A(size_A_words);
    std::vector<uint32_t> h_B(size_B_words);
    std::vector<uint32_t> h_ret(size_ret_words, 0);

    std::uniform_int_distribution<uint32_t> word_dist(0u, 0xffffffffu);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < L_a; ++j) h_A[(size_t)i * stride_A + j] = word_dist(rng);
        for (uint32_t j = 0; j < L_b; ++j) h_B[(size_t)i * stride_B + j] = word_dist(rng);
    }

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    uint32_t* d_ret = nullptr;
    uint32_t* d_workspace = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, size_A_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ret, size_ret_words * sizeof(uint32_t)));

    size_t workspace_size = batch_mul_ntt_workspace_size(N, L_a, L_b);
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B_words * sizeof(uint32_t), cudaMemcpyHostToDevice));

    batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b, stride_A, stride_B, stride_ret);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_ret.data(), d_ret, size_ret_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (uint32_t i = 0; i < N; ++i) {
        mpz_t a, b, expected, got;
        mpz_inits(a, b, expected, got, NULL);

        words_to_mpz(a, &h_A[(size_t)i * stride_A], L_a);
        words_to_mpz(b, &h_B[(size_t)i * stride_B], L_b);
        mpz_mul(expected, a, b);
        words_to_mpz(got, &h_ret[(size_t)i * stride_ret], L);

        if (mpz_cmp(expected, got) != 0) {
            pass = false;
            if (verbose) {
                printf("  Mismatch at index %u\n", i);
                gmp_printf("  A = %Zx\n", a);
                gmp_printf("  B = %Zx\n", b);
                gmp_printf("  Expected = %Zx\n", expected);
                gmp_printf("  Got      = %Zx\n", got);
            }
            mpz_clears(a, b, expected, got, NULL);
            break;
        }
        mpz_clears(a, b, expected, got, NULL);
    }

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_ret));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));

    return pass;
}

static bool run_suite(
    const char* name,
    uint32_t total_limit,
    int rounds,
    bool fixed_total,
    NTTPrecomputedTables tables,
    std::mt19937_64& rng
) {
    constexpr uint32_t L_MIN = 512;
    constexpr uint32_t L_MAX = 16384;

    printf("%s\n", name);
    std::uniform_int_distribution<uint32_t> total_dist(L_MIN, total_limit);

    for (int t = 0; t < rounds; ++t) {
        uint32_t target_total = fixed_total ? total_limit : total_dist(rng);
        uint32_t max_L = std::min(L_MAX, target_total);
        uint32_t L = sample_log_uniform_u32(L_MIN, max_L, rng);
        std::uniform_int_distribution<uint32_t> la_dist(1u, L - 1);
        uint32_t L_a = la_dist(rng);
        uint32_t L_b = L - L_a;
        uint32_t N = target_total / L;
        if (N == 0) N = 1;

        if (!test_configuration(N, L_a, L_b, tables, rng, true)) {
            return false;
        }
    }
    return true;
}

static void benchmark_configuration(uint32_t L_a, uint32_t L_b, uint64_t target_NL, NTTPrecomputedTables tables) {
    uint32_t N = (uint32_t)(target_NL / L_a);
    if (N == 0) N = 1;
    const uint32_t L = L_a + L_b;
    const uint32_t stride_A = L_a;
    const uint32_t stride_B = L_b;
    const uint32_t stride_ret = L;
    const size_t size_A_words = (size_t)N * stride_A;
    const size_t size_B_words = (size_t)N * stride_B;
    const size_t size_ret_words = (size_t)N * stride_ret;

    printf("Benchmarking L_a=%u, L_b=%u, N=%u (N*L_a=%llu)...\n",
           L_a, L_b, N, (unsigned long long)((uint64_t)N * L_a));

    const size_t size_A = size_A_words * sizeof(uint32_t);
    const size_t size_B = size_B_words * sizeof(uint32_t);
    const size_t size_ret = size_ret_words * sizeof(uint32_t);
    const size_t workspace_size = batch_mul_ntt_workspace_size(N, L_a, L_b);

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    uint32_t* d_ret = nullptr;
    uint32_t* d_workspace = nullptr;
    curandGenerator_t rand_gen = nullptr;

    cudaError_t err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_A failed: %s)\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_B, size_B);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_B failed: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc(&d_ret, size_ret);
    if (err != cudaSuccess) {
        printf("  SKIPPED (cudaMalloc d_ret failed: %s)\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }
    if (workspace_size > 0) {
        err = cudaMalloc(&d_workspace, workspace_size);
        if (err != cudaSuccess) {
            printf("  SKIPPED (cudaMalloc workspace failed: %s)\n", cudaGetErrorString(err));
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_ret);
            return;
        }
    }
    printf("  params size %4dM, workspace size %4dM ",
        int((size_A + size_B + size_ret) / 1000000),
        int(workspace_size / 1000000));

    CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, 0x4d595df4ULL));
    CURAND_CHECK(curandGenerate(rand_gen, d_A, size_A_words));
    CURAND_CHECK(curandGenerate(rand_gen, d_B, size_B_words));
    CURAND_CHECK(curandGenerate(rand_gen, d_ret, size_ret_words));

    // Warmup
    batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b, stride_A, stride_B, stride_ret);
    batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b, stride_A, stride_B, stride_ret);
    batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b, stride_A, stride_B, stride_ret);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_mul_ntt(d_A, d_B, d_ret, d_workspace, tables, N, L_a, L_b, stride_A, stride_B, stride_ret);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    const double avg_ms = milliseconds / iterations;

    const double mul_per_sec_thousand = ((double)N / (avg_ms * 1e-3)) / 1e3;
    const double io_words = (double)N * (double)(L_a + L_b + L);
    const double bandwidth_gb_s = (io_words * sizeof(uint32_t)) / (avg_ms * 1e-3) / 1e9;

    printf("  Average time: %6.3f ms ", avg_ms);
    printf("Mul/s: %8.2f K ", mul_per_sec_thousand);
    printf("Bandwidth (A+B+ret): %6.2f GB/s\n", bandwidth_gb_s);

    quick_check_products_mod_10007_cuda(d_A, d_B, d_ret, N, L_a, L_b, stride_A, stride_B, stride_ret);

    CURAND_CHECK(curandDestroyGenerator(rand_gen));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_ret));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
}

int main() {
    std::mt19937_64 rng(100ull);
    bool all_passed = true;
    const uint64_t target_NL = 100000000ull; // keep N * L_a around 1e8

    uint3* d_lv1 = nullptr;
    uint3* d_lv2 = nullptr;
    uint3* d_inv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lv1, 65536 * sizeof(uint3)));
    CUDA_CHECK(cudaMalloc(&d_lv2, 65536 * sizeof(uint3)));
    CUDA_CHECK(cudaMalloc(&d_inv, 33 * sizeof(uint3)));

    NTTPrecomputedTables tables;
    tables.roots_table_lv1 = d_lv1;
    tables.roots_table_lv2 = d_lv2;
    tables.inv2n_table = d_inv;
    init_ntt_precomputed_tables(&tables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("=== Correctness Tests ===\n\n");

    all_passed = run_suite("Small size tests (N*(L_a+L_b) <= 16384):", 16384, 12, false, tables, rng) && all_passed;
    printf("\n");
    if (!all_passed) goto done;

    all_passed = run_suite("Medium size tests (N*(L_a+L_b) <= 2^20):", (1u << 20), 8, false, tables, rng) && all_passed;
    printf("\n");
    if (!all_passed) goto done;

    all_passed = run_suite("Large size tests (N*(L_a+L_b) ~ 1e7):", 10000000u, 3, true, tables, rng) && all_passed;
    printf("\n");
    if (!all_passed) goto done;

    printf("Large L tests (N=1,2,3,4,8 with N*(L_a+L_b) ~ 1e7):\n");
    {
        constexpr uint32_t target_total = 10000000u;
        const uint32_t large_l_ns[] = {1u, 2u, 3u, 4u, 8u};
        for (uint32_t N_case : large_l_ns) {
            uint32_t L = target_total / N_case;
            uint32_t L_a = L / 2;
            uint32_t L_b = L - L_a;
            if (!test_configuration(N_case, L_a, L_b, tables, rng, true)) {
                all_passed = false;
                break;
            }
        }
    }
    printf("\n");
    if (!all_passed) goto done;

    printf("=== Benchmark Tests ===\n\n");
    for (uint32_t e = 8; e <= 27; ++e) {
        
        uint32_t L_a = 1u << e;
        uint32_t L_b = L_a;
        benchmark_configuration(L_a, L_b, target_NL, tables);
    }
    printf("\n");

done:
    CUDA_CHECK(cudaFree(d_lv1));
    CUDA_CHECK(cudaFree(d_lv2));
    CUDA_CHECK(cudaFree(d_inv));

    printf("=== Summary ===\n");
    printf("All correctness tests: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}

/*
Benchmarking L_a=256, L_b=256, N=390625 (N*L_a=100000000)...
  params size 1600M, workspace size    0M   Average time:  8.424 ms Mul/s: 46368.46 K Bandwidth (A+B+ret): 189.93 GB/s
Benchmarking L_a=512, L_b=512, N=195312 (N*L_a=99999744)...
  params size 1599M, workspace size    0M   Average time:  9.131 ms Mul/s: 21390.29 K Bandwidth (A+B+ret): 175.23 GB/s
Benchmarking L_a=1024, L_b=1024, N=97656 (N*L_a=99999744)...
  params size 1599M, workspace size    0M   Average time: 10.406 ms Mul/s:  9384.55 K Bandwidth (A+B+ret): 153.76 GB/s
Benchmarking L_a=2048, L_b=2048, N=48828 (N*L_a=99999744)...
  params size 1599M, workspace size   95M   Average time: 12.580 ms Mul/s:  3881.32 K Bandwidth (A+B+ret): 127.18 GB/s
Benchmarking L_a=4096, L_b=4096, N=24414 (N*L_a=99999744)...
  params size 1599M, workspace size   95M   Average time: 17.326 ms Mul/s:  1409.07 K Bandwidth (A+B+ret):  92.35 GB/s
Benchmarking L_a=8192, L_b=8192, N=12207 (N*L_a=99999744)...
  params size 1599M, workspace size   95M   Average time: 18.164 ms Mul/s:   672.05 K Bandwidth (A+B+ret):  88.09 GB/s
Benchmarking L_a=16384, L_b=16384, N=6103 (N*L_a=99991552)...
  params size 1599M, workspace size   95M   Average time: 19.647 ms Mul/s:   310.64 K Bandwidth (A+B+ret):  81.43 GB/s
Benchmarking L_a=32768, L_b=32768, N=3051 (N*L_a=99975168)...
  params size 1599M, workspace size   95M   Average time: 22.000 ms Mul/s:   138.68 K Bandwidth (A+B+ret):  72.71 GB/s
Benchmarking L_a=65536, L_b=65536, N=1525 (N*L_a=99942400)...
  params size 1599M, workspace size   94M   Average time: 24.383 ms Mul/s:    62.54 K Bandwidth (A+B+ret):  65.58 GB/s
Benchmarking L_a=131072, L_b=131072, N=762 (N*L_a=99876864)...
  params size 1598M, workspace size   94M   Average time: 25.534 ms Mul/s:    29.84 K Bandwidth (A+B+ret):  62.58 GB/s
Benchmarking L_a=262144, L_b=262144, N=381 (N*L_a=99876864)...
  params size 1598M, workspace size   88M   Average time: 26.268 ms Mul/s:    14.50 K Bandwidth (A+B+ret):  60.83 GB/s
Benchmarking L_a=524288, L_b=524288, N=190 (N*L_a=99614720)...
  params size 1593M, workspace size   75M   Average time: 28.176 ms Mul/s:     6.74 K Bandwidth (A+B+ret):  56.57 GB/s
Benchmarking L_a=1048576, L_b=1048576, N=95 (N*L_a=99614720)...
  params size 1593M, workspace size   50M   Average time: 32.381 ms Mul/s:     2.93 K Bandwidth (A+B+ret):  49.22 GB/s
Benchmarking L_a=2097152, L_b=2097152, N=47 (N*L_a=98566144)...
  params size 1577M, workspace size  100M   Average time: 32.641 ms Mul/s:     1.44 K Bandwidth (A+B+ret):  48.31 GB/s
Benchmarking L_a=4194304, L_b=4194304, N=23 (N*L_a=96468992)...
  params size 1543M, workspace size  201M   Average time: 39.238 ms Mul/s:     0.59 K Bandwidth (A+B+ret):  39.34 GB/s
Benchmarking L_a=8388608, L_b=8388608, N=11 (N*L_a=92274688)...
  params size 1476M, workspace size  402M   Average time: 38.002 ms Mul/s:     0.29 K Bandwidth (A+B+ret):  38.85 GB/s
Benchmarking L_a=16777216, L_b=16777216, N=5 (N*L_a=83886080)...
  params size 1342M, workspace size  805M   Average time: 35.801 ms Mul/s:     0.14 K Bandwidth (A+B+ret):  37.49 GB/s
Benchmarking L_a=33554432, L_b=33554432, N=2 (N*L_a=67108864)...
  params size 1073M, workspace size 1610M   Average time: 29.147 ms Mul/s:     0.07 K Bandwidth (A+B+ret):  36.84 GB/s
Benchmarking L_a=67108864, L_b=67108864, N=1 (N*L_a=67108864)...
  params size 1073M, workspace size 3221M   Average time: 30.552 ms Mul/s:     0.03 K Bandwidth (A+B+ret):  35.15 GB/s
Benchmarking L_a=134217728, L_b=134217728, N=1 (N*L_a=134217728)...
  params size 2147M, workspace size 6442M   Average time: 64.758 ms Mul/s:     0.02 K Bandwidth (A+B+ret):  33.16 GB/s
*/
