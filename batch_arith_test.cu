#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

uint32_t make_word(uint32_t seed, uint32_t row, uint32_t col) {
    uint32_t x = seed ^ (0x9e3779b9u * (row + 1u)) ^ (0x7f4a7c15u * (col + 1u));
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

void fill_words(std::vector<uint32_t> &dst, uint32_t N, uint32_t L, uint32_t stride, uint32_t seed) {
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

struct DeviceBuffer {
    uint32_t *ptr = nullptr;

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

bool test_mul(BatchMPContext *ctx) {
    struct CaseConfig {
        const char *name;
        uint32_t N;
        uint32_t L_a;
        uint32_t L_b;
    };

    const CaseConfig cases[] = {
        {"naive_boundary", 4, 512, 512},
        {"ntt_boundary", 3, 513, 512},
        {"ntt_workspace_growth", 2, 2200, 2200},
    };

    size_t previous_workspace = batch_mp_workspace_size(ctx);

    for (const CaseConfig &cfg : cases) {
        const uint32_t stride_A = cfg.L_a;
        const uint32_t stride_B = cfg.L_b;
        const uint32_t stride_ret = cfg.L_a + cfg.L_b;

        std::vector<uint32_t> h_A((size_t)cfg.N * stride_A);
        std::vector<uint32_t> h_B((size_t)cfg.N * stride_B);
        std::vector<uint32_t> h_ret((size_t)cfg.N * stride_ret, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, stride_A, 0x12345678u);
        fill_words(h_B, cfg.N, cfg.L_b, stride_B, 0x87654321u);

        DeviceBuffer d_A;
        DeviceBuffer d_B;
        DeviceBuffer d_ret;
        CUDA_CHECK(cudaMalloc(&d_A.ptr, h_A.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_B.ptr, h_B.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ret.ptr, h_ret.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.ptr, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_ret.ptr, 0, h_ret.size() * sizeof(uint32_t)));

        printf("Running mul %-18s N=%u L_a=%u L_b=%u\n", cfg.name, cfg.N, cfg.L_a, cfg.L_b);
        CUDA_CHECK(batch_mp_mul(ctx, d_A.ptr, d_B.ptr, d_ret.ptr, cfg.N, cfg.L_a, cfg.L_b, stride_A, stride_B, stride_ret));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ret.data(), d_ret.ptr, h_ret.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, b, expected, got;
            mpz_inits(a, b, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * stride_A, cfg.L_a);
            words_to_mpz(b, h_B.data() + (size_t)i * stride_B, cfg.L_b);
            words_to_mpz(got, h_ret.data() + (size_t)i * stride_ret, stride_ret);
            mpz_mul(expected, a, b);
            if (mpz_cmp(expected, got) != 0) {
                fprintf(stderr, "Multiply mismatch in case %s at batch index %u\n", cfg.name, i);
                mpz_clears(a, b, expected, got, NULL);
                return false;
            }
            mpz_clears(a, b, expected, got, NULL);
        }

        const size_t workspace_after = batch_mp_workspace_size(ctx);
        printf("  workspace: %zu -> %zu bytes\n", previous_workspace, workspace_after);
        if (cfg.L_a + cfg.L_b <= 1024 && workspace_after != previous_workspace) {
            fprintf(stderr, "Naive multiply unexpectedly changed workspace size\n");
            return false;
        }
        if (cfg.L_a + cfg.L_b > 1024 && workspace_after < previous_workspace) {
            fprintf(stderr, "Workspace size shrank after NTT multiply\n");
            return false;
        }
        previous_workspace = workspace_after;
    }

    return true;
}

bool test_add(BatchMPContext *ctx) {
    struct CaseConfig {
        const char *name;
        uint32_t N;
        uint32_t L_a;
        uint32_t L_b;
        uint32_t L_c;
    };

    const CaseConfig cases[] = {
        {"small", 7, 13, 11, 15},
        {"workspace", 2, 5000, 4097, 5002},
    };

    size_t previous_workspace = batch_mp_workspace_size(ctx);

    for (const CaseConfig &cfg : cases) {
        const uint32_t stride_A = cfg.L_a;
        const uint32_t stride_B = cfg.L_b;
        const uint32_t stride_C = cfg.L_c;

        std::vector<uint32_t> h_A((size_t)cfg.N * stride_A);
        std::vector<uint32_t> h_B((size_t)cfg.N * stride_B);
        std::vector<uint32_t> h_C((size_t)cfg.N * stride_C, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, stride_A, 0xABCDEF01u);
        fill_words(h_B, cfg.N, cfg.L_b, stride_B, 0x10FEDCBAu);

        DeviceBuffer d_A;
        DeviceBuffer d_B;
        DeviceBuffer d_C;
        CUDA_CHECK(cudaMalloc(&d_A.ptr, h_A.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_B.ptr, h_B.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_C.ptr, h_C.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.ptr, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C.ptr, 0, h_C.size() * sizeof(uint32_t)));

        printf("Running add %-18s N=%u L_a=%u L_b=%u L_c=%u\n", cfg.name, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c);
        CUDA_CHECK(batch_mp_add(ctx, d_A.ptr, d_B.ptr, d_C.ptr, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c, stride_A, stride_B, stride_C));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.ptr, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, b, expected, got;
            mpz_inits(a, b, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * stride_A, cfg.L_a);
            words_to_mpz(b, h_B.data() + (size_t)i * stride_B, cfg.L_b);
            words_to_mpz(got, h_C.data() + (size_t)i * stride_C, cfg.L_c);
            mpz_add(expected, a, b);
            mpz_fdiv_r_2exp(expected, expected, (mp_bitcnt_t)cfg.L_c * 32u);
            if (mpz_cmp(expected, got) != 0) {
                fprintf(stderr, "Addition mismatch in case %s at batch index %u\n", cfg.name, i);
                mpz_clears(a, b, expected, got, NULL);
                return false;
            }
            mpz_clears(a, b, expected, got, NULL);
        }

        const size_t workspace_after = batch_mp_workspace_size(ctx);
        printf("  workspace: %zu -> %zu bytes\n", previous_workspace, workspace_after);
        if (cfg.L_c <= 4096 && workspace_after != previous_workspace) {
            fprintf(stderr, "Small addition unexpectedly changed workspace size\n");
            return false;
        }
        if (cfg.L_c > 4096 && workspace_after < previous_workspace) {
            fprintf(stderr, "Workspace size shrank after large addition\n");
            return false;
        }
        previous_workspace = workspace_after;
    }

    return true;
}

bool test_sub(BatchMPContext *ctx) {
    struct CaseConfig {
        const char *name;
        uint32_t N;
        uint32_t L_a;
        uint32_t L_b;
        uint32_t L_c;
    };

    const CaseConfig cases[] = {
        {"small", 6, 9, 12, 12},
        {"workspace", 2, 5001, 4098, 5003},
    };

    size_t previous_workspace = batch_mp_workspace_size(ctx);

    for (const CaseConfig &cfg : cases) {
        const uint32_t stride_A = cfg.L_a;
        const uint32_t stride_B = cfg.L_b;
        const uint32_t stride_C = cfg.L_c;

        std::vector<uint32_t> h_A((size_t)cfg.N * stride_A);
        std::vector<uint32_t> h_B((size_t)cfg.N * stride_B);
        std::vector<uint32_t> h_C((size_t)cfg.N * stride_C, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, stride_A, 0xCAFEBABEu);
        fill_words(h_B, cfg.N, cfg.L_b, stride_B, 0x13572468u);

        DeviceBuffer d_A;
        DeviceBuffer d_B;
        DeviceBuffer d_C;
        CUDA_CHECK(cudaMalloc(&d_A.ptr, h_A.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_B.ptr, h_B.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_C.ptr, h_C.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.ptr, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C.ptr, 0, h_C.size() * sizeof(uint32_t)));

        printf("Running sub %-18s N=%u L_a=%u L_b=%u L_c=%u\n", cfg.name, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c);
        CUDA_CHECK(batch_mp_sub(ctx, d_A.ptr, d_B.ptr, d_C.ptr, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c, stride_A, stride_B, stride_C));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.ptr, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, b, expected, got;
            mpz_inits(a, b, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * stride_A, cfg.L_a);
            words_to_mpz(b, h_B.data() + (size_t)i * stride_B, cfg.L_b);
            words_to_mpz(got, h_C.data() + (size_t)i * stride_C, cfg.L_c);
            mpz_sub(expected, a, b);
            mpz_fdiv_r_2exp(expected, expected, (mp_bitcnt_t)cfg.L_c * 32u);
            if (mpz_cmp(expected, got) != 0) {
                fprintf(stderr, "Subtraction mismatch in case %s at batch index %u\n", cfg.name, i);
                mpz_clears(a, b, expected, got, NULL);
                return false;
            }
            mpz_clears(a, b, expected, got, NULL);
        }

        const size_t workspace_after = batch_mp_workspace_size(ctx);
        printf("  workspace: %zu -> %zu bytes\n", previous_workspace, workspace_after);
        if (cfg.L_c <= 4096 && workspace_after != previous_workspace) {
            fprintf(stderr, "Small subtraction unexpectedly changed workspace size\n");
            return false;
        }
        if (cfg.L_c > 4096 && workspace_after < previous_workspace) {
            fprintf(stderr, "Workspace size shrank after large subtraction\n");
            return false;
        }
        previous_workspace = workspace_after;
    }

    return true;
}

bool test_shift(BatchMPContext *ctx) {
    struct CaseConfig {
        const char *name;
        uint32_t N;
        uint32_t L_in;
        uint32_t L_out;
        int32_t shift_bits;
    };

    const CaseConfig cases[] = {
        {"left", 5, 10, 12, 37},
        {"right", 5, 12, 9, -43},
    };

    for (const CaseConfig &cfg : cases) {
        const uint32_t stride_in = cfg.L_in;
        const uint32_t stride_out = cfg.L_out;
        std::vector<uint32_t> h_A((size_t)cfg.N * stride_in);
        std::vector<uint32_t> h_B((size_t)cfg.N * stride_out, 0u);
        fill_words(h_A, cfg.N, cfg.L_in, stride_in, 0x2468ACE1u);

        DeviceBuffer d_A;
        DeviceBuffer d_B;
        CUDA_CHECK(cudaMalloc(&d_A.ptr, h_A.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_B.ptr, h_B.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_B.ptr, 0, h_B.size() * sizeof(uint32_t)));

        printf("Running shift %-16s N=%u L_in=%u L_out=%u shift=%d\n", cfg.name, cfg.N, cfg.L_in, cfg.L_out, cfg.shift_bits);
        CUDA_CHECK(batch_mp_shift_bits(ctx, d_A.ptr, d_B.ptr, cfg.N, cfg.L_in, cfg.L_out, stride_in, stride_out, cfg.shift_bits));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_B.data(), d_B.ptr, h_B.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t input, expected, got;
            mpz_inits(input, expected, got, NULL);
            words_to_mpz(input, h_A.data() + (size_t)i * stride_in, cfg.L_in);
            if (cfg.shift_bits >= 0) {
                mpz_mul_2exp(expected, input, (mp_bitcnt_t)cfg.shift_bits);
            } else {
                mpz_fdiv_q_2exp(expected, input, (mp_bitcnt_t)(-cfg.shift_bits));
            }
            mpz_fdiv_r_2exp(expected, expected, (mp_bitcnt_t)cfg.L_out * 32u);
            words_to_mpz(got, h_B.data() + (size_t)i * stride_out, cfg.L_out);
            if (mpz_cmp(expected, got) != 0) {
                fprintf(stderr, "Shift mismatch in case %s at batch index %u\n", cfg.name, i);
                mpz_clears(input, expected, got, NULL);
                return false;
            }
            mpz_clears(input, expected, got, NULL);
        }
    }

    return true;
}

bool test_bitlength(BatchMPContext *ctx) {
    const uint32_t N = 4;
    const uint32_t L = 20;
    const uint32_t stride_A = L;
    std::vector<uint32_t> h_A((size_t)N * stride_A, 0u);

    h_A[0] = 0u;
    h_A[(size_t)1 * stride_A + 0] = 1u;
    h_A[(size_t)2 * stride_A + 7] = 0x00008000u;
    h_A[(size_t)3 * stride_A + 19] = 0x80000000u;

    DeviceBuffer d_A;
    CUDA_CHECK(cudaMalloc(&d_A.ptr, h_A.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t got = 0u;
    printf("Running bitlength N=%u L=%u\n", N, L);
    CUDA_CHECK(batch_mp_bitlength_max(ctx, d_A.ptr, N, L, stride_A, &got));
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t expected = 0u;
    for (uint32_t i = 0; i < N; ++i) {
        mpz_t value;
        mpz_init(value);
        words_to_mpz(value, h_A.data() + (size_t)i * stride_A, L);
        const uint32_t bits = (mpz_sgn(value) == 0) ? 0u : (uint32_t)mpz_sizeinbase(value, 2);
        expected = std::max(expected, bits);
        mpz_clear(value);
    }

    if (got != expected) {
        fprintf(stderr, "Bitlength mismatch: expected=%u got=%u\n", expected, got);
        return false;
    }

    return true;
}

}  // namespace

int main() {
    bool ok = true;

    auto run_with_context = [&](bool (*fn)(BatchMPContext *)) {
        BatchMPContext *ctx = batch_mp_init();
        if (ctx == nullptr) {
            fprintf(stderr, "batch_mp_init failed\n");
            ok = false;
            return;
        }
        ok = ok && fn(ctx);
        batch_mp_destroy(ctx);
    };

    run_with_context(test_mul);
    run_with_context(test_add);
    run_with_context(test_sub);
    run_with_context(test_shift);
    run_with_context(test_bitlength);

    printf("Summary: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
