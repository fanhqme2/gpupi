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

struct ArrayOwner {
    BatchMPArray array{};

    ~ArrayOwner() {
        batch_mp_array_release(array);
    }
};

BatchMPArray make_array(uint32_t batch_size, uint32_t length, uint32_t stride = 0u) {
    BatchMPArray array = batch_mp_array_create(batch_size, length, stride);
    if (batch_size != 0u && array.data == nullptr) {
        fprintf(stderr, "batch_mp_array_create failed\n");
        exit(1);
    }
    return array;
}

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
        ArrayOwner A{make_array(cfg.N, cfg.L_a)};
        ArrayOwner B{make_array(cfg.N, cfg.L_b)};
        ArrayOwner C{make_array(cfg.N, cfg.L_a + cfg.L_b)};

        std::vector<uint32_t> h_A((size_t)cfg.N * A.array.stride);
        std::vector<uint32_t> h_B((size_t)cfg.N * B.array.stride);
        std::vector<uint32_t> h_C((size_t)cfg.N * C.array.stride, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, A.array.stride, 0x12345678u);
        fill_words(h_B, cfg.N, cfg.L_b, B.array.stride, 0x87654321u);

        CUDA_CHECK(cudaMemcpy(A.array.data, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.array.data, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(C.array.data, 0, h_C.size() * sizeof(uint32_t)));

        printf("Running mul %-18s N=%u L_a=%u L_b=%u\n", cfg.name, cfg.N, cfg.L_a, cfg.L_b);
        CUDA_CHECK(batch_mp_mul(ctx, A.array, B.array, C.array));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), C.array.data, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, b, expected, got;
            mpz_inits(a, b, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * A.array.stride, cfg.L_a);
            words_to_mpz(b, h_B.data() + (size_t)i * B.array.stride, cfg.L_b);
            words_to_mpz(got, h_C.data() + (size_t)i * C.array.stride, C.array.length);
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

    ArrayOwner A{make_array(2, 10)};
    ArrayOwner B{make_array(3, 8)};
    ArrayOwner C{make_array(2, 18)};
    if (batch_mp_mul(ctx, A.array, B.array, C.array) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected batch mismatch to fail in multiply\n");
        return false;
    }

    ArrayOwner WrongC{make_array(2, 17)};
    ArrayOwner MatchB{make_array(2, 8)};
    if (batch_mp_mul(ctx, A.array, MatchB.array, WrongC.array) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected output length mismatch to fail in multiply\n");
        return false;
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
        ArrayOwner A{make_array(cfg.N, cfg.L_a)};
        ArrayOwner B{make_array(cfg.N, cfg.L_b)};
        ArrayOwner C{make_array(cfg.N, cfg.L_c)};

        std::vector<uint32_t> h_A((size_t)cfg.N * A.array.stride);
        std::vector<uint32_t> h_B((size_t)cfg.N * B.array.stride);
        std::vector<uint32_t> h_C((size_t)cfg.N * C.array.stride, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, A.array.stride, 0xABCDEF01u);
        fill_words(h_B, cfg.N, cfg.L_b, B.array.stride, 0x10FEDCBAu);

        CUDA_CHECK(cudaMemcpy(A.array.data, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.array.data, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(C.array.data, 0, h_C.size() * sizeof(uint32_t)));

        printf("Running add %-18s N=%u L_a=%u L_b=%u L_c=%u\n", cfg.name, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c);
        CUDA_CHECK(batch_mp_add(ctx, A.array, B.array, C.array));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), C.array.data, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, b, expected, got;
            mpz_inits(a, b, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * A.array.stride, cfg.L_a);
            words_to_mpz(b, h_B.data() + (size_t)i * B.array.stride, cfg.L_b);
            words_to_mpz(got, h_C.data() + (size_t)i * C.array.stride, cfg.L_c);
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

    ArrayOwner BadAddA{make_array(2, 4)};
    ArrayOwner BadAddB{make_array(3, 4)};
    ArrayOwner BadAddC{make_array(2, 5)};
    if (batch_mp_add(ctx, BadAddA.array, BadAddB.array, BadAddC.array) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected batch mismatch to fail in addition\n");
        return false;
    }

    return true;
}

bool test_add_small(BatchMPContext *ctx) {
    struct CaseConfig {
        const char *name;
        uint32_t N;
        uint32_t L_a;
        uint32_t L_c;
        uint32_t B;
    };

    const CaseConfig cases[] = {
        {"small", 7, 13, 15, 0x12345678u},
        {"workspace", 2, 5000, 5002, 0xFEDCBA98u},
    };

    size_t previous_workspace = batch_mp_workspace_size(ctx);

    for (const CaseConfig &cfg : cases) {
        ArrayOwner A{make_array(cfg.N, cfg.L_a)};
        ArrayOwner C{make_array(cfg.N, cfg.L_c)};

        std::vector<uint32_t> h_A((size_t)cfg.N * A.array.stride);
        std::vector<uint32_t> h_C((size_t)cfg.N * C.array.stride, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, A.array.stride, 0x31415926u);

        CUDA_CHECK(cudaMemcpy(A.array.data, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(C.array.data, 0, h_C.size() * sizeof(uint32_t)));

        printf("Running add_small %-12s N=%u L_a=%u L_c=%u B=%08x\n",
               cfg.name, cfg.N, cfg.L_a, cfg.L_c, cfg.B);
        CUDA_CHECK(batch_mp_add_small(ctx, A.array, cfg.B, C.array));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), C.array.data, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, expected, got;
            mpz_inits(a, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * A.array.stride, cfg.L_a);
            words_to_mpz(got, h_C.data() + (size_t)i * C.array.stride, cfg.L_c);
            mpz_set(expected, a);
            mpz_add_ui(expected, expected, (unsigned long)cfg.B);
            mpz_fdiv_r_2exp(expected, expected, (mp_bitcnt_t)cfg.L_c * 32u);
            if (mpz_cmp(expected, got) != 0) {
                fprintf(stderr, "Add-small mismatch in case %s at batch index %u\n", cfg.name, i);
                mpz_clears(a, expected, got, NULL);
                return false;
            }
            mpz_clears(a, expected, got, NULL);
        }

        const size_t workspace_after = batch_mp_workspace_size(ctx);
        printf("  workspace: %zu -> %zu bytes\n", previous_workspace, workspace_after);
        if (cfg.L_c <= 4096 && workspace_after != previous_workspace) {
            fprintf(stderr, "Small add-small unexpectedly changed workspace size\n");
            return false;
        }
        if (cfg.L_c > 4096 && workspace_after < previous_workspace) {
            fprintf(stderr, "Workspace size shrank after large add-small\n");
            return false;
        }
        previous_workspace = workspace_after;
    }

    ArrayOwner BadAddSmallA{make_array(2, 4)};
    ArrayOwner BadAddSmallC{make_array(3, 5)};
    if (batch_mp_add_small(ctx, BadAddSmallA.array, 7u, BadAddSmallC.array) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected batch mismatch to fail in add-small\n");
        return false;
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
        ArrayOwner A{make_array(cfg.N, cfg.L_a)};
        ArrayOwner B{make_array(cfg.N, cfg.L_b)};
        ArrayOwner C{make_array(cfg.N, cfg.L_c)};

        std::vector<uint32_t> h_A((size_t)cfg.N * A.array.stride);
        std::vector<uint32_t> h_B((size_t)cfg.N * B.array.stride);
        std::vector<uint32_t> h_C((size_t)cfg.N * C.array.stride, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, A.array.stride, 0xCAFEBABEu);
        fill_words(h_B, cfg.N, cfg.L_b, B.array.stride, 0x13572468u);

        CUDA_CHECK(cudaMemcpy(A.array.data, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.array.data, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(C.array.data, 0, h_C.size() * sizeof(uint32_t)));

        printf("Running sub %-18s N=%u L_a=%u L_b=%u L_c=%u\n", cfg.name, cfg.N, cfg.L_a, cfg.L_b, cfg.L_c);
        CUDA_CHECK(batch_mp_sub(ctx, A.array, B.array, C.array));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), C.array.data, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, b, expected, got;
            mpz_inits(a, b, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * A.array.stride, cfg.L_a);
            words_to_mpz(b, h_B.data() + (size_t)i * B.array.stride, cfg.L_b);
            words_to_mpz(got, h_C.data() + (size_t)i * C.array.stride, cfg.L_c);
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

    ArrayOwner BadSubA{make_array(2, 4)};
    ArrayOwner BadSubB{make_array(3, 4)};
    ArrayOwner BadSubC{make_array(2, 5)};
    if (batch_mp_sub(ctx, BadSubA.array, BadSubB.array, BadSubC.array) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected batch mismatch to fail in subtraction\n");
        return false;
    }

    return true;
}

bool test_sub_small(BatchMPContext *ctx) {
    struct CaseConfig {
        const char *name;
        uint32_t N;
        uint32_t L_a;
        uint32_t L_c;
        uint32_t B;
    };

    const CaseConfig cases[] = {
        {"small", 6, 12, 12, 0x02468ACEu},
        {"workspace", 2, 5001, 5003, 0x89ABCDEFu},
    };

    size_t previous_workspace = batch_mp_workspace_size(ctx);

    for (const CaseConfig &cfg : cases) {
        ArrayOwner A{make_array(cfg.N, cfg.L_a)};
        ArrayOwner C{make_array(cfg.N, cfg.L_c)};

        std::vector<uint32_t> h_A((size_t)cfg.N * A.array.stride);
        std::vector<uint32_t> h_C((size_t)cfg.N * C.array.stride, 0u);
        fill_words(h_A, cfg.N, cfg.L_a, A.array.stride, 0x27182818u);

        CUDA_CHECK(cudaMemcpy(A.array.data, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(C.array.data, 0, h_C.size() * sizeof(uint32_t)));

        printf("Running sub_small %-12s N=%u L_a=%u L_c=%u B=%08x\n",
               cfg.name, cfg.N, cfg.L_a, cfg.L_c, cfg.B);
        CUDA_CHECK(batch_mp_sub_small(ctx, A.array, cfg.B, C.array));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), C.array.data, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t a, expected, got;
            mpz_inits(a, expected, got, NULL);
            words_to_mpz(a, h_A.data() + (size_t)i * A.array.stride, cfg.L_a);
            words_to_mpz(got, h_C.data() + (size_t)i * C.array.stride, cfg.L_c);
            mpz_set(expected, a);
            mpz_sub_ui(expected, expected, (unsigned long)cfg.B);
            mpz_fdiv_r_2exp(expected, expected, (mp_bitcnt_t)cfg.L_c * 32u);
            if (mpz_cmp(expected, got) != 0) {
                fprintf(stderr, "Sub-small mismatch in case %s at batch index %u\n", cfg.name, i);
                mpz_clears(a, expected, got, NULL);
                return false;
            }
            mpz_clears(a, expected, got, NULL);
        }

        const size_t workspace_after = batch_mp_workspace_size(ctx);
        printf("  workspace: %zu -> %zu bytes\n", previous_workspace, workspace_after);
        if (cfg.L_c <= 4096 && workspace_after != previous_workspace) {
            fprintf(stderr, "Small sub-small unexpectedly changed workspace size\n");
            return false;
        }
        if (cfg.L_c > 4096 && workspace_after < previous_workspace) {
            fprintf(stderr, "Workspace size shrank after large sub-small\n");
            return false;
        }
        previous_workspace = workspace_after;
    }

    ArrayOwner BadSubSmallA{make_array(2, 4)};
    ArrayOwner BadSubSmallC{make_array(3, 5)};
    if (batch_mp_sub_small(ctx, BadSubSmallA.array, 7u, BadSubSmallC.array) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected batch mismatch to fail in sub-small\n");
        return false;
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
        ArrayOwner A{make_array(cfg.N, cfg.L_in)};
        ArrayOwner B{make_array(cfg.N, cfg.L_out)};
        std::vector<uint32_t> h_A((size_t)cfg.N * A.array.stride);
        std::vector<uint32_t> h_B((size_t)cfg.N * B.array.stride, 0u);
        fill_words(h_A, cfg.N, cfg.L_in, A.array.stride, 0x2468ACE1u);

        CUDA_CHECK(cudaMemcpy(A.array.data, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(B.array.data, 0, h_B.size() * sizeof(uint32_t)));

        printf("Running shift %-16s N=%u L_in=%u L_out=%u shift=%d\n", cfg.name, cfg.N, cfg.L_in, cfg.L_out, cfg.shift_bits);
        CUDA_CHECK(batch_mp_shift_bits(ctx, A.array, B.array, cfg.shift_bits));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_B.data(), B.array.data, h_B.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < cfg.N; ++i) {
            mpz_t input, expected, got;
            mpz_inits(input, expected, got, NULL);
            words_to_mpz(input, h_A.data() + (size_t)i * A.array.stride, cfg.L_in);
            if (cfg.shift_bits >= 0) {
                mpz_mul_2exp(expected, input, (mp_bitcnt_t)cfg.shift_bits);
            } else {
                mpz_fdiv_q_2exp(expected, input, (mp_bitcnt_t)(-cfg.shift_bits));
            }
            mpz_fdiv_r_2exp(expected, expected, (mp_bitcnt_t)cfg.L_out * 32u);
            words_to_mpz(got, h_B.data() + (size_t)i * B.array.stride, cfg.L_out);
            if (mpz_cmp(expected, got) != 0) {
                fprintf(stderr, "Shift mismatch in case %s at batch index %u\n", cfg.name, i);
                mpz_clears(input, expected, got, NULL);
                return false;
            }
            mpz_clears(input, expected, got, NULL);
        }
    }

    ArrayOwner BadShiftA{make_array(2, 4)};
    ArrayOwner BadShiftB{make_array(3, 4)};
    if (batch_mp_shift_bits(ctx, BadShiftA.array, BadShiftB.array, 7) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected batch mismatch to fail in shift\n");
        return false;
    }

    return true;
}

bool test_bitlength_and_compact(BatchMPContext *ctx) {
    ArrayOwner A{make_array(4, 20, 24)};
    std::vector<uint32_t> h_A((size_t)A.array.batch_size * A.array.stride, 0u);

    h_A[0] = 0u;
    h_A[(size_t)1 * A.array.stride + 0] = 1u;
    h_A[(size_t)2 * A.array.stride + 7] = 0x00008000u;
    h_A[(size_t)3 * A.array.stride + 19] = 0x80000000u;

    CUDA_CHECK(cudaMemcpy(A.array.data, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t got = 0u;
    printf("Running bitlength N=%u L=%u\n", A.array.batch_size, A.array.length);
    CUDA_CHECK(batch_mp_bitlength_max(ctx, A.array, &got));
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t expected = 0u;
    for (uint32_t i = 0; i < A.array.batch_size; ++i) {
        mpz_t value;
        mpz_init(value);
        words_to_mpz(value, h_A.data() + (size_t)i * A.array.stride, A.array.length);
        const uint32_t bits = (mpz_sgn(value) == 0) ? 0u : (uint32_t)mpz_sizeinbase(value, 2);
        expected = std::max(expected, bits);
        mpz_clear(value);
    }

    if (got != expected) {
        fprintf(stderr, "Bitlength mismatch: expected=%u got=%u\n", expected, got);
        return false;
    }

    BatchMPArray compacted = A.array;
    CUDA_CHECK(compacted.compact(ctx));
    if (compacted.length != 20u) {
        fprintf(stderr, "Compact changed length unexpectedly for full-width data: got=%u\n", compacted.length);
        return false;
    }

    ArrayOwner B{make_array(3, 12, 16)};
    std::vector<uint32_t> h_B((size_t)B.array.batch_size * B.array.stride, 0u);
    h_B[(size_t)0 * B.array.stride + 1] = 0x00000001u;
    h_B[(size_t)1 * B.array.stride + 4] = 0x7fffffffu;
    h_B[(size_t)2 * B.array.stride + 5] = 0x00000001u;
    CUDA_CHECK(cudaMemcpy(B.array.data, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    BatchMPArray compacted_small = B.array;
    CUDA_CHECK(compacted_small.compact(ctx));
    if (compacted_small.length != 6u) {
        fprintf(stderr, "Compact computed wrong length: expected=6 got=%u\n", compacted_small.length);
        return false;
    }

    BatchMPArray invalid_array{};
    invalid_array.length = 4;
    invalid_array.batch_size = 2;
    invalid_array.stride = 3;
    if (batch_mp_bitlength_max(ctx, invalid_array, &got) != cudaErrorInvalidValue) {
        fprintf(stderr, "Expected invalid array metadata to fail in bitlength\n");
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
    run_with_context(test_add_small);
    run_with_context(test_sub);
    run_with_context(test_sub_small);
    run_with_context(test_shift);
    run_with_context(test_bitlength_and_compact);

    printf("Summary: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
