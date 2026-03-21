#include "batch_arith.h"

#include <new>

#include "batch_add.h"
#include "batch_add_small.h"
#include "batch_bitlength.h"
#include "batch_exactdiv_small.h"
#include "batch_mul_naive.h"
#include "batch_mul_ntt.h"
#include "batch_mul_small.h"
#include "batch_shift.h"
#include "batch_shift_add.h"
#include "batch_shift_sub.h"
#include "batch_sub.h"
#include "batch_sub_small.h"

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

bool valid_array(BatchMPArray array) {
    if (array.batch_size == 0) {
        return array.length <= array.stride;
    }
    if (array.data == nullptr) {
        return false;
    }
    if (array.stride == 0u) {
        return true;
    }
    return array.length <= array.stride;
}

bool same_batch(BatchMPArray a, BatchMPArray b) {
    return a.batch_size == b.batch_size;
}

}  // namespace

cudaError_t batch_mp_ensure_workspace(BatchMPContext *ctx, size_t required_bytes){
    return ensure_workspace(ctx, required_bytes);
}

BatchMPArray batch_mp_array_create(uint32_t batch_size, uint32_t length, uint32_t stride) {
    BatchMPArray array{};
    array.length = length;
    array.batch_size = batch_size;
    array.stride = (stride == 0u) ? length : stride;
    if (array.length > array.stride) {
        array.stride = 0u;
        array.length = 0u;
        array.batch_size = 0u;
        return array;
    }

    const size_t bytes = (size_t)array.batch_size * array.stride * sizeof(uint32_t);
    if (bytes == 0u) {
        return array;
    }

    if (cudaMalloc(&array.data, bytes) != cudaSuccess) {
        array = {};
    }
    return array;
}

void batch_mp_array_release(BatchMPArray array) {
    if (array.data != nullptr) {
        cudaFree(array.data);
    }
}

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

cudaError_t batch_mp_mul(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B) || !same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }
    if (C.length != A.length + B.length) {
        return cudaErrorInvalidValue;
    }

    const size_t L = (size_t)A.length + (size_t)B.length;
    if (L <= BATCH_MUL_NAIVE_L_MAX) {
        batch_mul_naive(
            A.data, B.data, C.data,
            (int)A.batch_size, (int)A.length, (int)B.length,
            (int)A.stride, (int)B.stride, (int)C.stride
        );
        return cudaGetLastError();
    }

    const size_t workspace_size = batch_mul_ntt_workspace_size(A.batch_size, A.length, B.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_mul_ntt(
        A.data, B.data, C.data, ctx->workspace, ctx->ntt_tables,
        A.batch_size, A.length, B.length, A.stride, B.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_mul_small(BatchMPContext *ctx, BatchMPArray A, uint32_t B, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_mul_small_workspace_size(A.batch_size, A.length, C.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_mul_small(
        A.data, B, C.data, ctx->workspace,
        A.batch_size, A.length, C.length, A.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_exactdiv_small(BatchMPContext *ctx, BatchMPArray A, uint32_t B) {
    if (ctx == nullptr || !valid_array(A)) {
        return cudaErrorInvalidValue;
    }
    if (B == 0u || (B & 1u) == 0u || A.length > 128u) {
        return cudaErrorInvalidValue;
    }

    batch_exactdiv_small(A.data, B, A.batch_size, A.length, A.stride);
    return cudaGetLastError();
}

cudaError_t batch_mp_add(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B) || !same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_add_simple_workspace_size(A.batch_size, A.length, B.length, C.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_add_simple(
        A.data, B.data, C.data, ctx->workspace,
        A.batch_size, A.length, B.length, C.length, A.stride, B.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_add_small(BatchMPContext *ctx, BatchMPArray A, uint32_t B, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_add_small_workspace_size(A.batch_size, A.length, C.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_add_small(
        A.data, B, C.data, ctx->workspace,
        A.batch_size, A.length, C.length, A.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_shift_add(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, uint32_t shift, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B) || !same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }
    if (C.data == A.data || C.data == B.data) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_shift_add_simple_workspace_size(
        A.batch_size, A.length, B.length, C.length, shift
    );
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_shift_add_simple(
        A.data, B.data, C.data, ctx->workspace, shift,
        A.batch_size, A.length, B.length, C.length, A.stride, B.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_shift_sub(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, uint32_t shift, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B) || !same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }
    if (C.data == A.data || C.data == B.data) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_shift_sub_simple_workspace_size(
        A.batch_size, A.length, B.length, C.length, shift
    );
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_shift_sub_simple(
        A.data, B.data, C.data, ctx->workspace, shift,
        A.batch_size, A.length, B.length, C.length, A.stride, B.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_sub(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B) || !same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_sub_simple_workspace_size(A.batch_size, A.length, B.length, C.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_sub_simple(
        A.data, B.data, C.data, ctx->workspace,
        A.batch_size, A.length, B.length, C.length, A.stride, B.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_sub_small(BatchMPContext *ctx, BatchMPArray A, uint32_t B, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_sub_small_workspace_size(A.batch_size, A.length, C.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    batch_sub_small(
        A.data, B, C.data, ctx->workspace,
        A.batch_size, A.length, C.length, A.stride, C.stride
    );
    return cudaGetLastError();
}

cudaError_t batch_mp_shift_bits(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, int32_t shift_bits) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B)) {
        return cudaErrorInvalidValue;
    }

    batch_shift_bits(A.data, B.data, A.batch_size, A.length, B.length, A.stride, B.stride, shift_bits);
    return cudaGetLastError();
}

cudaError_t batch_mp_bitlength_max(BatchMPContext *ctx, BatchMPArray A, uint32_t *result) {
    if (ctx == nullptr || !valid_array(A) || result == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_bitlength_workspace_size(A.batch_size, A.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    *result = batch_bitlength_max(A.data, ctx->workspace, A.batch_size, A.length, A.stride);
    return cudaGetLastError();
}

cudaError_t batch_mp_limblength_max(BatchMPContext *ctx, BatchMPArray A, uint32_t *result) {
    if (ctx == nullptr || !valid_array(A) || result == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t workspace_size = batch_bitlength_workspace_size(A.batch_size, A.length);
    cudaError_t err = ensure_workspace(ctx, workspace_size);
    if (err != cudaSuccess) {
        return err;
    }

    *result = batch_limblength_max(A.data, ctx->workspace, A.batch_size, A.length, A.stride);
    return cudaGetLastError();
}

cudaError_t BatchMPArray::compact(BatchMPContext *ctx) {
    uint32_t limblength = 0u;
    cudaError_t err = batch_mp_limblength_max(ctx, *this, &limblength);
    if (err != cudaSuccess) {
        return err;
    }

    length = limblength;
    return cudaSuccess;
}
