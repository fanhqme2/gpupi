#include "batch_arith.h"

#include <new>

#include "batch_add.h"
#include "batch_bitlength.h"
#include "batch_mul_naive.h"
#include "batch_shift.h"
#include "batch_sub.h"
#include "batch_mul_ntt.h"

struct BatchMPContext {
    NTTPrecomputedTables ntt_tables;
    uint32_t *workspace;
    size_t workspace_size_bytes;
};

namespace {

cudaError_t allocate_ntt_tables(BatchMPContext *ctx) {
    cudaError_t err = cudaMalloc(&ctx->ntt_tables.roots_table_lv1, 65536 * sizeof(uint3));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&ctx->ntt_tables.roots_table_lv2, 65536 * sizeof(uint3));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&ctx->ntt_tables.inv2n_table, 33 * sizeof(uint3));
    if (err != cudaSuccess) return err;

    init_ntt_precomputed_tables(&ctx->ntt_tables);

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    return cudaDeviceSynchronize();
}

cudaError_t ensure_workspace(BatchMPContext *ctx, size_t required_bytes) {
    if (required_bytes <= ctx->workspace_size_bytes) {
        return cudaSuccess;
    }

    uint32_t *new_workspace = nullptr;
    cudaError_t err = cudaMalloc(&new_workspace, required_bytes);
    if (err != cudaSuccess) {
        return err;
    }

    if (ctx->workspace != nullptr) {
        cudaError_t free_err = cudaFree(ctx->workspace);
        if (free_err != cudaSuccess) {
            cudaFree(new_workspace);
            return free_err;
        }
    }

    ctx->workspace = new_workspace;
    ctx->workspace_size_bytes = required_bytes;
    return cudaSuccess;
}

void release_context_buffers(BatchMPContext *ctx) {
    if (ctx->workspace != nullptr) {
        cudaFree(ctx->workspace);
        ctx->workspace = nullptr;
    }
    ctx->workspace_size_bytes = 0;

    if (ctx->ntt_tables.roots_table_lv1 != nullptr) {
        cudaFree(ctx->ntt_tables.roots_table_lv1);
        ctx->ntt_tables.roots_table_lv1 = nullptr;
    }
    if (ctx->ntt_tables.roots_table_lv2 != nullptr) {
        cudaFree(ctx->ntt_tables.roots_table_lv2);
        ctx->ntt_tables.roots_table_lv2 = nullptr;
    }
    if (ctx->ntt_tables.inv2n_table != nullptr) {
        cudaFree(ctx->ntt_tables.inv2n_table);
        ctx->ntt_tables.inv2n_table = nullptr;
    }
}

}  // namespace

BatchMPContext * batch_mp_init() {
    BatchMPContext *ctx = new (std::nothrow) BatchMPContext{};
    if (ctx == nullptr) {
        return nullptr;
    }

    const cudaError_t err = allocate_ntt_tables(ctx);
    if (err != cudaSuccess) {
        release_context_buffers(ctx);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void batch_mp_destroy(BatchMPContext *ctx) {
    if (ctx == nullptr) {
        return;
    }

    release_context_buffers(ctx);
    delete ctx;
}

size_t batch_mp_workspace_size(const BatchMPContext *ctx) {
    return (ctx == nullptr) ? 0u : ctx->workspace_size_bytes;
}

cudaError_t batch_mp_mul(
    BatchMPContext * ctx,
    uint32_t * A,
    uint32_t * B,
    uint32_t * ret,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_ret
) {
    if (ctx == nullptr || A == nullptr || B == nullptr || ret == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t L = (size_t)L_a + (size_t)L_b;
    if (L <= BATCH_MUL_NAIVE_L_MAX) {
        batch_mul_naive(
            A, B, ret,
            (int)N, (int)L_a, (int)L_b,
            (int)stride_A, (int)stride_B, (int)stride_ret
        );
        return cudaGetLastError();
    }

    const size_t workspace_size = batch_mul_ntt_workspace_size(N, L_a, L_b);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_mul_ntt(
        A, B, ret, ctx->workspace, ctx->ntt_tables,
        N, L_a, L_b, stride_A, stride_B, stride_ret
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_add(
    BatchMPContext * ctx,
    uint32_t * A,
    uint32_t * B,
    uint32_t * C,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    if (ctx == nullptr || A == nullptr || B == nullptr || C == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_add_simple_workspace_size(N, L_a, L_b, L_c);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_add_simple(A, B, C, ctx->workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    return cudaGetLastError();
}

cudaError_t batch_mp_sub(
    BatchMPContext * ctx,
    uint32_t * A,
    uint32_t * B,
    uint32_t * C,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    if (ctx == nullptr || A == nullptr || B == nullptr || C == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_sub_simple_workspace_size(N, L_a, L_b, L_c);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_sub_simple(A, B, C, ctx->workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    return cudaGetLastError();
}

cudaError_t batch_mp_shift_bits(
    BatchMPContext * ctx,
    const uint32_t * A,
    uint32_t * B,
    uint32_t N,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    int32_t shift_bits
) {
    if (ctx == nullptr || A == nullptr || B == nullptr) {
        return cudaErrorInvalidValue;
    }

    batch_shift_bits(A, B, N, L_in, L_out, stride_in, stride_out, shift_bits);
    return cudaGetLastError();
}

cudaError_t batch_mp_bitlength_max(
    BatchMPContext * ctx,
    const uint32_t * A,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A,
    uint32_t * result
) {
    if (ctx == nullptr || A == nullptr || result == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_bitlength_workspace_size(N, L);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    *result = batch_bitlength_max(A, ctx->workspace, N, L, stride_A);
    return cudaGetLastError();
}
