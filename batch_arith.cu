#include "pi_device_memory.h"
#include "batch_arith.h"

#include <new>
#include <algorithm>

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
    size_t multiplication_budget;
};

namespace {

cudaError_t allocate_ntt_tables(BatchMPContext *ctx) {
    cudaError_t err = pi_memory::allocate(&ctx->ntt_tables.roots_table_lv1, 65536 * sizeof(uint3));
    if (err != cudaSuccess) return err;

    err = pi_memory::allocate(&ctx->ntt_tables.roots_table_lv2, 65536 * sizeof(uint3));
    if (err != cudaSuccess) return err;

    err = pi_memory::allocate(&ctx->ntt_tables.inv2n_table, 33 * sizeof(uint3));
    if (err != cudaSuccess) return err;

    err = pi_memory::allocate(&ctx->ntt_tables.expanded, (1u << kRootTableBits) * sizeof(uint3));
    if (err == cudaErrorMemoryAllocation) {
        ctx->ntt_tables.expanded = nullptr;
        cudaGetLastError();
    } else if (err != cudaSuccess) {
        return err;
    }

    init_ntt_precomputed_tables(&ctx->ntt_tables);

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    return cudaDeviceSynchronize();
}

cudaError_t ensure_workspace(BatchMPContext *ctx, size_t required_bytes) {
    if (required_bytes <= ctx->workspace_size_bytes) {
        return cudaSuccess;
    }

    // Scratch has no live contents between arithmetic calls. Release it before
    // growth so the old and new multi-GiB NTT arenas never coexist.
    if (ctx->workspace != nullptr) {
        cudaError_t err = pi_memory::release(ctx->workspace);
        if (err != cudaSuccess) return err;
        ctx->workspace = nullptr;
        ctx->workspace_size_bytes = 0;
    }
    uint32_t *new_workspace = nullptr;
    cudaError_t err = pi_memory::allocate(&new_workspace, required_bytes);
    if (err != cudaSuccess) return err;

    ctx->workspace = new_workspace;
    ctx->workspace_size_bytes = required_bytes;
    return cudaSuccess;
}

void release_context_buffers(BatchMPContext *ctx) {
    if (ctx->workspace != nullptr) {
        pi_memory::release(ctx->workspace);
        ctx->workspace = nullptr;
    }
    ctx->workspace_size_bytes = 0;

    if (ctx->ntt_tables.roots_table_lv1 != nullptr) {
        pi_memory::release(ctx->ntt_tables.roots_table_lv1);
        ctx->ntt_tables.roots_table_lv1 = nullptr;
    }
    if (ctx->ntt_tables.roots_table_lv2 != nullptr) {
        pi_memory::release(ctx->ntt_tables.roots_table_lv2);
        ctx->ntt_tables.roots_table_lv2 = nullptr;
    }
    if (ctx->ntt_tables.expanded) pi_memory::release(ctx->ntt_tables.expanded);
    if (ctx->ntt_tables.inv2n_table != nullptr) {
        pi_memory::release(ctx->ntt_tables.inv2n_table);
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

    if (pi_memory::allocate(&array.data, bytes) != cudaSuccess) {
        array = {};
    }
    return array;
}

void batch_mp_array_release(BatchMPArray array) {
    if (array.data != nullptr) {
        pi_memory::release(array.data);
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

void batch_mp_limit_large_products(BatchMPContext *ctx, size_t budget) {
    ctx->multiplication_budget = budget;
}

// Overlap-add whole-limb products. The suffix includes the remaining high
// limbs so that carries propagate across tile boundaries.
static cudaError_t tiled_mul(BatchMPContext *ctx, BatchMPArray a,
    BatchMPArray b, BatchMPArray out) {
    uint32_t tile = 1;
    while (tile < (1u << 29) &&
           batch_mul_ntt_workspace_size(1, 2 * tile, 2 * tile) <= ctx->multiplication_budget)
        tile *= 2;
    auto temporary = batch_mp_array_create(1, 2 * tile);
    if (!temporary.data) return cudaErrorMemoryAllocation;
    cudaError_t err = cudaSuccess;
    for (uint32_t row = 0; row < out.batch_size && err == cudaSuccess; ++row) {
        auto *dest = out.data + (size_t)row * out.stride;
        err = cudaMemsetAsync(dest, 0, (size_t)out.length * sizeof(uint32_t));
        for (uint32_t i = 0; i < a.length && err == cudaSuccess; i += tile) {
            const uint32_t na = std::min(tile, a.length - i);
            BatchMPArray x{a.data + (size_t)row * a.stride + i, na, 1, na};
            for (uint32_t j = 0; j < b.length && err == cudaSuccess; j += tile) {
                const uint32_t nb = std::min(tile, b.length - j);
                BatchMPArray y{b.data + (size_t)row * b.stride + j, nb, 1, nb};
                temporary.length = na + nb;
                // The linear transform for a tile pair fits the budget, even
                // when the parent selected tiling because of its batch size.
                err = batch_mp_mul(ctx, x, y, temporary);
                if (err == cudaSuccess) {
                    uint32_t length = out.length - i - j;
                    BatchMPArray sum{dest + i + j, length, 1, length};
                    err = batch_mp_add(ctx, sum, temporary, sum);
                }
            }
        }
    }
    batch_mp_array_release(temporary);
    return err;
}

cudaError_t batch_mp_mul(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, BatchMPArray C) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B) || !valid_array(C)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B) || !same_batch(A, C)) {
        return cudaErrorInvalidValue;
    }
    if ((uint64_t)C.length != (uint64_t)A.length + B.length ||
        (uint64_t)A.length + B.length > (1ull << 31)) {
        return cudaErrorInvalidValue;
    }

    const size_t L = (size_t)A.length + (size_t)B.length;
    if (L <= BATCH_MP_MUL_NAIVE_CUTOFF) {
        batch_mul_naive(
            A.data, B.data, C.data,
            (int)A.batch_size, (int)A.length, (int)B.length,
            (int)A.stride, (int)B.stride, (int)C.stride
        );
        return cudaGetLastError();
    }

    const size_t workspace_size = batch_mul_ntt_workspace_size(A.batch_size, A.length, B.length);
    if (ctx->multiplication_budget && workspace_size > ctx->multiplication_budget)
        return tiled_mul(ctx, A, B, C);
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
    if (B == 0u || (B & 1u) == 0u || A.length > 2048u) {
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

cudaError_t batch_mp_shift_bits(BatchMPContext *ctx, BatchMPArray A, BatchMPArray B, int64_t shift_bits) {
    if (ctx == nullptr || !valid_array(A) || !valid_array(B)) {
        return cudaErrorInvalidValue;
    }
    if (!same_batch(A, B)) {
        return cudaErrorInvalidValue;
    }

    batch_shift_bits(A.data, B.data, A.batch_size, A.length, B.length, A.stride, B.stride, shift_bits);
    return cudaGetLastError();
}

cudaError_t batch_mp_bitlength_max(BatchMPContext *ctx, BatchMPArray A, uint64_t *result) {
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

cudaError_t batch_mp_mul_shared_b(BatchMPContext *ctx, BatchMPArray A,
    BatchMPArray C, BatchMPArray B, BatchMPArray out_a, BatchMPArray out_c) {
    if (!ctx || !valid_array(A) || !valid_array(B) || !valid_array(C) ||
        !valid_array(out_a) || !valid_array(out_c) || !same_batch(A, B) ||
        !same_batch(A, C) || !same_batch(A, out_a) || !same_batch(A, out_c) ||
        (uint64_t)A.length + B.length != out_a.length ||
        (uint64_t)C.length + B.length != out_c.length ||
        out_a.length > (1ull << 31) || out_c.length > (1ull << 31))
        return cudaErrorInvalidValue;
    size_t bytes = batch_mul_ntt_shared_b_workspace_size(A.batch_size, A.length, C.length, B.length);
    if (!bytes || (ctx->multiplication_budget && bytes > ctx->multiplication_budget)) {
        cudaError_t err = batch_mp_mul(ctx, A, B, out_a);
        return err == cudaSuccess ? batch_mp_mul(ctx, C, B, out_c) : err;
    }
    cudaError_t err = ensure_workspace(ctx, bytes);
    if (err != cudaSuccess) return err;
    batch_mul_ntt_shared_b({A.data, out_a.data, A.length, A.stride, out_a.stride},
        {C.data, out_c.data, C.length, C.stride, out_c.stride}, B.data, B.length,
        B.stride, A.batch_size, ctx->workspace, ctx->ntt_tables);
    return cudaGetLastError();
}

cudaError_t batch_mp_exactdiv_small_many(BatchMPContext *ctx, BatchMPArray A,
    const uint32_t *divisors, uint32_t count) {
    if (!ctx || !valid_array(A) || A.length > 2048u || count > 32u ||
        (count && !divisors)) return cudaErrorInvalidValue;
    for (uint32_t i = 0; i < count; ++i)
        if (!divisors[i] || !(divisors[i] & 1u)) return cudaErrorInvalidValue;
    batch_exactdiv_small_many(A.data, divisors, count, A.batch_size, A.length, A.stride);
    return cudaGetLastError();
}

cudaError_t batch_mp_split_merge(BatchMPContext *ctx, BatchMPArray p, BatchMPArray q,
    BatchMPArray r, BatchMPArray shifted, BatchMPArray po, BatchMPArray qo, BatchMPArray ro) {
    const uint32_t pairs = qo.batch_size;
    if (!ctx || !pairs || !valid_array(p) || !valid_array(q) || !valid_array(r) ||
        !valid_array(shifted) || !valid_array(po) || !valid_array(qo) || !valid_array(ro) ||
        q.batch_size != 2ull * pairs || r.batch_size != q.batch_size ||
        (p.batch_size != q.batch_size && p.batch_size + 1ull != q.batch_size) ||
        po.batch_size != p.batch_size / 2 || ro.batch_size != pairs || shifted.batch_size != pairs ||
        !p.stride || !q.stride || !r.stride || !shifted.stride || !po.stride || !qo.stride || !ro.stride ||
        po.length != 2ull * p.length || qo.length != 2ull * q.length ||
        ro.length <= (uint64_t)shifted.length + q.length ||
        ro.length <= (uint64_t)p.length + r.length)
        return cudaErrorInvalidValue;
    const size_t bytes = batch_ntt_merge_workspace(po, qo, ro);
    if (!bytes) return cudaErrorNotSupported;
    cudaError_t err = ensure_workspace(ctx, bytes);
    if (err != cudaSuccess) return err;
    batch_ntt_merge(p, q, r, shifted, po, qo, ro, ctx->workspace, bytes, ctx->ntt_tables);
    return cudaGetLastError();
}
