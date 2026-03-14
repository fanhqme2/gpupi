#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <gmp.h>

#include "batch_arith.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

namespace {

struct CaseConfig {
    const char *name;
    uint32_t N;
    uint32_t L_a;
    uint32_t L_b;
};

uint32_t make_word(uint32_t seed, uint32_t i, uint32_t j) {
    uint32_t x = seed ^ (0x9e3779b9u * (i + 1u)) ^ (0x7f4a7c15u * (j + 1u));
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

void fill_operands(std::vector<uint32_t> &dst, uint32_t N, uint32_t L, uint32_t stride, uint32_t seed) {
    std::fill(dst.begin(), dst.end(), 0u);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
            dst[(size_t)i * stride + j] = make_word(seed, i, j);
        }
    }
}

void words_to_mpz(mpz_t out, const uint32_t *words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

bool run_case(BatchMPContext *ctx, const CaseConfig &cfg, size_t *workspace_after_call) {
    const uint32_t stride_A = cfg.L_a;
    const uint32_t stride_B = cfg.L_b;
    const uint32_t stride_ret = cfg.L_a + cfg.L_b;

    std::vector<uint32_t> h_A((size_t)cfg.N * stride_A);
    std::vector<uint32_t> h_B((size_t)cfg.N * stride_B);
    std::vector<uint32_t> h_ret((size_t)cfg.N * stride_ret, 0u);

    fill_operands(h_A, cfg.N, cfg.L_a, stride_A, 0x12345678u);
    fill_operands(h_B, cfg.N, cfg.L_b, stride_B, 0x87654321u);

    uint32_t *d_A = nullptr;
    uint32_t *d_B = nullptr;
    uint32_t *d_ret = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ret, h_ret.size() * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ret, 0, h_ret.size() * sizeof(uint32_t)));

    CUDA_CHECK(batch_mp_mul(
        ctx, d_A, d_B, d_ret,
        cfg.N, cfg.L_a, cfg.L_b,
        stride_A, stride_B, stride_ret
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_ret.data(), d_ret, h_ret.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (uint32_t i = 0; i < cfg.N; ++i) {
        mpz_t a, b, expected, got;
        mpz_inits(a, b, expected, got, NULL);
        words_to_mpz(a, h_A.data() + (size_t)i * stride_A, cfg.L_a);
        words_to_mpz(b, h_B.data() + (size_t)i * stride_B, cfg.L_b);
        words_to_mpz(got, h_ret.data() + (size_t)i * stride_ret, stride_ret);
        mpz_mul(expected, a, b);

        if (mpz_cmp(expected, got) != 0) {
            fprintf(stderr, "Mismatch in case %s at batch index %u\n", cfg.name, i);
            gmp_printf("A        = %Zx\n", a);
            gmp_printf("B        = %Zx\n", b);
            gmp_printf("Expected = %Zx\n", expected);
            gmp_printf("Got      = %Zx\n", got);
            pass = false;
            mpz_clears(a, b, expected, got, NULL);
            break;
        }

        mpz_clears(a, b, expected, got, NULL);
    }

    *workspace_after_call = ctx->workspace_size_bytes;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_ret));
    return pass;
}

}  // namespace

int main() {
    BatchMPContext *ctx = batch_mp_init();
    if (ctx == nullptr) {
        fprintf(stderr, "batch_mp_init failed\n");
        return 1;
    }

    const CaseConfig cases[] = {
        {"naive_boundary", 4, 512, 512},
        {"ntt_boundary", 3, 513, 512},
        {"ntt_workspace_growth", 2, 2200, 2200},
    };

    size_t previous_workspace_size = ctx->workspace_size_bytes;
    bool all_passed = true;

    for (const CaseConfig &cfg : cases) {
        size_t workspace_after_call = 0;
        printf("Running %-22s N=%u L_a=%u L_b=%u\n", cfg.name, cfg.N, cfg.L_a, cfg.L_b);
        if (!run_case(ctx, cfg, &workspace_after_call)) {
            all_passed = false;
            break;
        }

        printf("  workspace: %zu -> %zu bytes\n", previous_workspace_size, workspace_after_call);

        if (cfg.L_a + cfg.L_b <= 1024 && workspace_after_call != previous_workspace_size) {
            fprintf(stderr, "Naive dispatch unexpectedly changed workspace size\n");
            all_passed = false;
            break;
        }
        if (cfg.L_a + cfg.L_b > 1024 && workspace_after_call < previous_workspace_size) {
            fprintf(stderr, "Workspace size shrank after an NTT call\n");
            all_passed = false;
            break;
        }
        if (cfg.L_a + cfg.L_b > 2048 && workspace_after_call == 0) {
            fprintf(stderr, "Large NTT case did not allocate workspace\n");
            all_passed = false;
            break;
        }

        previous_workspace_size = workspace_after_call;
    }

    batch_mp_destroy(ctx);

    printf("Summary: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed ? 0 : 1;
}
