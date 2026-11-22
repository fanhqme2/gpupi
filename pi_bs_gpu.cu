#include "pi_device_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <cerrno>
#include <climits>

#include <gmp.h>

#include "batch_arith.h"
#include "convert_decimal.h"
#include "batch_mul_naive.h"
#include "batch_mul_ntt.h"

__device__ __forceinline__ void reduce_gcd_quotient(uint32_t &a, uint32_t &b){
    if (a % 3 == 0 && b % 3 == 0){
        a /= 3;
        b /= 3;
    }
    if (a % 5 == 0 && b % 5 == 0){
        a /= 5;
        b /= 5;
    }
    if (a % 23 == 0 && b % 23 == 0){
        a /= 23;
        b /= 23;
    }
    if (a % 29 == 0 && b % 29 == 0){
        a /= 29;
        b /= 29;
    }
}

// leaf case of binary splitting.
template<bool is_head>
__global__ void construct_leaf_17(uint32_t * arr_P, uint32_t * arr_Q, uint32_t * arr_R, int N, uint32_t block_offset = 0){
    for (int i2 = blockIdx.x * blockDim.x + threadIdx.x; i2 < N; i2 += blockDim.x * gridDim.x){
        uint32_t i;
        if (is_head){
            i = i2 * 17 + 16;
        }else{
            i = i2 + (i2 >> 4);
        }
        i += block_offset * 17u;
        uint32_t p1 = 6u * i + 1;
        uint32_t p2 = 2u * i + 1;
        uint32_t p3 = 6u * i + 5;
        uint32_t q1 = i + 1;
        uint32_t q2 = i + 1;
        uint32_t q3 = i + 1;
        uint32_t q4 = 640320 >> 6;
        uint32_t q5 = 40020 >> 2;
        uint32_t q6 = 426880 >> 7;
        uint64_t r1 = (int64_t)545140134 * i + 13591409;
        if (p1 % 5 == 0 && q1 % 5 == 0) {
            p1 /= 5;
            q1 /= 5;
        }
        if (p1 % 5 == 0 && q2 % 5 == 0) {
            p1 /= 5;
            q2 /= 5;
        }
        if (p1 % 5 == 0 && q3 % 5 == 0) {
            p1 /= 5;
            q3 /= 5;
        }
        reduce_gcd_quotient(p1, q4);
        reduce_gcd_quotient(p1, q5);
        reduce_gcd_quotient(p2, q4);
        reduce_gcd_quotient(p2, q5);
        reduce_gcd_quotient(p3, q4);
        reduce_gcd_quotient(p3, q5);

        // p = mpz(p1) * p2 * p3
        // q = mpz(q1) * q2 * q3 * q4 * q5 * q6
        // r = mpz(r1) * q1 * q2 * q3 * q4 * q5

        uint32_t P[3] = {p1, 0, 0};
        uint32_t p_components[2] = {p2, p3};

        for (int k = 0; k < 2; k ++){
            uint32_t carry = 0;
            for (int j = 0; j < 3; j++){
                uint64_t mul = (uint64_t)P[j] * p_components[k] + carry;
                P[j] = mul & 0xffffffff;
                carry = mul >> 32;
            }
        }

        uint32_t Q[4] = {q1, 0, 0, 0};
        uint32_t q_components[5] = {q2, q3, q4, q5, q6};

        for (int k = 0; k < 5; k ++){
            uint32_t carry = 0;
            for (int j = 0; j < 4; j++){
                uint64_t mul = (uint64_t)Q[j] * q_components[k] + carry;
                Q[j] = mul & 0xffffffff;
                carry = mul >> 32;
            }
        }

        uint32_t R[6] = {(uint32_t)r1, (uint32_t)(r1 >> 32), 0, 0, 0, 0};
        uint32_t r_components[5] = {q1, q2, q3, q4, q5};

        for (int k = 0; k < 5; k ++){
            uint32_t carry = 0;
            for (int j = 0; j < 6; j++){
                uint64_t mul = (uint64_t)R[j] * r_components[k] + carry;
                R[j] = mul & 0xffffffff;
                carry = mul >> 32;
            }
        }

        arr_P[i2 * 3 + 0] = P[0];
        arr_P[i2 * 3 + 1] = P[1];
        arr_P[i2 * 3 + 2] = P[2];

        arr_Q[i2 * 4 + 0] = Q[0];
        arr_Q[i2 * 4 + 1] = Q[1];
        arr_Q[i2 * 4 + 2] = Q[2];
        arr_Q[i2 * 4 + 3] = Q[3];

        arr_R[i2 * 6 + 0] = R[0];
        arr_R[i2 * 6 + 1] = R[1];
        arr_R[i2 * 6 + 2] = R[2];
        arr_R[i2 * 6 + 3] = R[3];
        arr_R[i2 * 6 + 4] = R[4];
        arr_R[i2 * 6 + 5] = R[5];
    }
}

struct StageProfileEntry {
    const char * kind;
    const char * label;
    uint64_t n1;
    uint32_t batch_size;
    uint32_t lhs_length;
    uint32_t rhs_length;
    float elapsed_ms;
    bool uses_ntt;
};

struct StageProfiler {
    StageProfileEntry entries[2048];
    int count;
};

namespace {

constexpr uint32_t kLargeBlockTerms = 8704;
constexpr uint32_t kBlockCommonDivisorLo = 830297u;
constexpr uint32_t kBlockCommonDivisorHi = 156590819u;
// Guaranteed common factors beyond the original 17-term block cancellation.
// Derivation and residue-exhaustive proof: generate_common_factors.py.
static const uint32_t kCommon34[] = {980837u};
static const uint32_t kCommon68[] = {1732459729u, 649235263u, 4255083361u, 146969u};
static const uint32_t kCommon136[] = {247494247u, 474934849u, 127434929u, 209826283u,
    518240837u, 927671347u, 1606662479u, 2427855097u, 54567769u, 107972737u, 12317u};


constexpr uint32_t kValidationMod = 1000000007u;
constexpr uint64_t kLimbBaseMod = (1ull << 32) % kValidationMod;
uint32_t kDecimalLeafDigits = 477u;

cudaError_t append_stage_profile(
    StageProfiler * profiler,
    const char * kind,
    const char * label,
    uint64_t n1,
    uint32_t batch_size,
    uint32_t lhs_length,
    uint32_t rhs_length,
    float elapsed_ms,
    bool uses_ntt
) {
    if (profiler != nullptr && profiler->count < (int)(sizeof(profiler->entries) / sizeof(profiler->entries[0]))) {
        profiler->entries[profiler->count++] = {
            .kind = kind,
            .label = label,
            .n1 = n1,
            .batch_size = batch_size,
            .lhs_length = lhs_length,
            .rhs_length = rhs_length,
            .elapsed_ms = elapsed_ms,
            .uses_ntt = uses_ntt
        };
    }
    return cudaSuccess;
}

template <typename Fn>
cudaError_t profile_stage(
    const char * kind,
    const char * label,
    uint64_t n1,
    uint32_t batch_size,
    uint32_t lhs_length,
    uint32_t rhs_length,
    bool uses_ntt,
    StageProfiler * profiler,
    Fn && fn
) {
    if (getenv("GPUPI_TRACE")) {
        fprintf(stderr, "stage kind=%s label=%s n=%" PRIu64 " batch=%u a=%u b=%u\n",
            kind, label, n1, batch_size, lhs_length, rhs_length);
    }
    if (profiler == nullptr) {
        return fn();
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        return err;
    }
    err = cudaEventRecord(start);
    if (err == cudaSuccess) {
        err = fn();
    }
    if (err == cudaSuccess) {
        err = cudaEventRecord(stop);
    }
    if (err == cudaSuccess) {
        err = cudaEventSynchronize(stop);
    }

    float elapsed_ms = 0.0f;
    if (err == cudaSuccess) {
        err = cudaEventElapsedTime(&elapsed_ms, start, stop);
    }
    if (err == cudaSuccess) {
        err = append_stage_profile(
            profiler,
            kind,
            label,
            n1,
            batch_size,
            lhs_length,
            rhs_length,
            elapsed_ms,
            uses_ntt
        );
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return err;
}

cudaError_t profile_mul(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C,
    const char * label,
    uint64_t n1,
    StageProfiler * profiler
) {
    return profile_stage(
        "mul",
        label,
        n1,
        A.batch_size,
        A.length,
        B.length,
        ((size_t)A.length + (size_t)B.length) > BATCH_MP_MUL_NAIVE_CUTOFF,
        profiler,
        [&]() { return batch_mp_mul(context, A, B, C); }
    );
}

cudaError_t profile_shift_addsub(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C,
    uint32_t shift_amount,
    bool subtract,
    uint64_t n1,
    StageProfiler * profiler
) {
    return profile_stage(
        subtract ? "sub" : "add",
        subtract ? "R*Q<<15n-P*R" : "R*Q<<15n+P*R",
        n1,
        A.batch_size,
        A.length,
        B.length,
        false,
        profiler,
        [&]() {
            return subtract
                ? batch_mp_shift_sub(context, A, B, shift_amount, C)
                : batch_mp_shift_add(context, A, B, shift_amount, C);
        }
    );
}

cudaError_t profile_compact(
    BatchMPContext * context,
    BatchMPArray & array,
    const char * label,
    uint64_t n1,
    StageProfiler * profiler
) {
    return profile_stage(
        "compact",
        label,
        n1,
        array.batch_size,
        array.length,
        0,
        false,
        profiler,
        [&]() { return array.compact(context); }
    );
}

cudaError_t profile_exactdiv(
    BatchMPContext * context,
    BatchMPArray array,
    const char * label,
    uint32_t divisor,
    uint64_t n1,
    StageProfiler * profiler
) {
    return profile_stage(
        "exactdiv",
        label,
        n1,
        array.batch_size,
        array.length,
        0,
        false,
        profiler,
        [&]() { return batch_mp_exactdiv_small(context, array, divisor); }
    );
}

cudaError_t profile_mul_small(
    BatchMPContext * context,
    BatchMPArray A,
    uint32_t multiplier,
    BatchMPArray C,
    const char * label,
    uint64_t digits,
    StageProfiler * profiler
) {
    return profile_stage(
        "mul_small",
        label,
        digits,
        A.batch_size,
        A.length,
        1,
        false,
        profiler,
        [&]() { return batch_mp_mul_small(context, A, multiplier, C); }
    );
}

cudaError_t profile_add(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C,
    const char * label,
    uint64_t digits,
    StageProfiler * profiler
) {
    return profile_stage(
        "add",
        label,
        digits,
        A.batch_size,
        A.length,
        B.length,
        false,
        profiler,
        [&]() { return batch_mp_add(context, A, B, C); }
    );
}

cudaError_t profile_sub(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C,
    const char * label,
    uint64_t digits,
    StageProfiler * profiler
) {
    return profile_stage(
        "sub",
        label,
        digits,
        A.batch_size,
        A.length,
        B.length,
        false,
        profiler,
        [&]() { return batch_mp_sub(context, A, B, C); }
    );
}

cudaError_t profile_add_small(
    BatchMPContext * context,
    BatchMPArray A,
    uint32_t value,
    BatchMPArray C,
    const char * label,
    uint64_t digits,
    StageProfiler * profiler
) {
    return profile_stage(
        "add",
        label,
        digits,
        A.batch_size,
        A.length,
        1,
        false,
        profiler,
        [&]() { return batch_mp_add_small(context, A, value, C); }
    );
}

cudaError_t profile_shift_bits(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    int64_t shift_bits,
    const char * label,
    uint64_t digits,
    StageProfiler * profiler
) {
    return profile_stage(
        "shift",
        label,
        digits,
        A.batch_size,
        A.length,
        0,
        false,
        profiler,
        [&]() { return batch_mp_shift_bits(context, A, B, shift_bits); }
    );
}

cudaError_t profile_decimal_precompute_powers(
    BatchMPContext * context,
    ConvertDecimalPowerTable * table,
    uint32_t leaf_digits,
    uint32_t levels,
    StageProfiler * profiler
) {
    return profile_stage(
        "decimal_powers",
        "precompute_5pow",
        levels == 0u ? leaf_digits : ((uint64_t)leaf_digits << levels),
        1u,
        levels,
        leaf_digits,
        false,
        profiler,
        [&]() { return convert_decimal_precompute_powers(context, table, leaf_digits, levels); }
    );
}

cudaError_t profile_decimal_convert(
    BatchMPContext * context,
    BatchMPArray input,
    uint64_t input_bits,
    uint64_t total_digits,
    uint32_t leaf_digits,
    const ConvertDecimalPowerTable * table,
    char * output_digits,
    StageProfiler * profiler
) {
    return profile_stage(
        "decimal_convert",
        "equal_split",
        total_digits,
        input.batch_size,
        input.length,
        leaf_digits,
        false,
        profiler,
        [&]() {
            return convert_decimal_equal_split(
                context, input, input_bits, total_digits, leaf_digits, table, output_digits
            );
        }
    );
}

}  // namespace

#define CHECK_AND_RETURN(err, cleanup) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            cleanup; \
            return _err; \
        } \
    } while (0)

// Multiply prime powers in a balanced tree so GMP can use fast multiplication.
static void factor_product(mpz_t out, const std::vector<uint2> &factors,
    size_t first, size_t last) {
    if (last - first == 1) {
        mpz_ui_pow_ui(out, factors[first].x, factors[first].y);
        return;
    }
    const size_t middle = first + (last - first) / 2;
    mpz_t right;
    mpz_init(right);
    factor_product(out, factors, first, middle);
    factor_product(right, factors, middle, last);
    mpz_mul(out, out, right);
    mpz_clear(right);
}

// For each odd prime power m coprime to the leaf constants, a term of R
// uses a prefix of P factors followed by a suffix of Q factors. In each
// period the three numerator factors occur at floor(m/6), floor(m/2),
// floor(5m/6), and all three denominator factors at m-1. Completed
// periods therefore contribute equally on either side of the split.
// In [a,b), any split at s contributes 3*(floor(b/m)-floor(a/m))
// minus the numerator prefix at a mod m, plus the prefix at s mod m.
// The latter is nonnegative. Crossing an extra period gains three factors,
// enough to cover even the largest starting prefix.
static void block_common_divisor(mpz_t divisor, uint32_t terms) {
    std::vector<bool> composite(terms + 1);
    std::vector<uint2> powers;
    auto valuation = [](uint32_t value, uint32_t p) {
        uint32_t e = 0;
        while (value % p == 0) { ++e; value /= p; }
        return e;
    };
    for (uint32_t p = 2; p <= terms; ++p) {
        if (composite[p]) continue;
        for (uint32_t j = p + p; j <= terms; j += p) composite[j] = true;
        if (p <= 5 || p == 23 || p == 29) continue;
        uint32_t exponent = 0;
        for (uint64_t m = p; m <= terms; m *= p) {
            // With no extra denominator period, the largest possible
            // starting residue is m-1-(terms % m). Discount its prefix.
            const uint32_t start = m - 1 - terms % m;
            exponent += 3 * (terms / m) - (m / 6 < start) -
                (m / 2 < start) - (5 * m / 6 < start);
        }
        uint32_t removed = 8 * (valuation(kBlockCommonDivisorLo, p) + valuation(kBlockCommonDivisorHi, p));
        for (auto f : kCommon34) removed += 4 * valuation(f, p);
        for (auto f : kCommon68) removed += 2 * valuation(f, p);
        for (auto f : kCommon136) removed += valuation(f, p);
        removed *= terms / 136;
        if (exponent > removed) powers.push_back(make_uint2(p, exponent - removed));
    }
    factor_product(divisor, powers, 0, powers.size());
}

// Use the low limbs of X * D^-1 mod 2^k for the exact quotient X/D.
// The same odd divisor applies to every row and all three columns.
static cudaError_t cancel_block_factor(BatchMPContext *ctx, uint32_t terms,
    BatchMPArray &P, BatchMPArray &Q, BatchMPArray &R, uint32_t *scratch, size_t capacity) {
    mpz_t divisor, residual, inverse;
    mpz_inits(divisor, residual, inverse, nullptr);
    block_common_divisor(divisor, terms);
    const uint32_t saved = (mpz_sizeinbase(divisor, 2) - 1) / 32;
    const uint32_t length = std::max({P.length, Q.length, R.length});
    // A different precision may have different row lengths. Skip this
    // optional reduction if the existing binary-split scratch cannot hold it.
    for (const auto &column : {P, Q, R}) {
        if (column.batch_size && (column.length <= saved ||
            (size_t)(length - saved) + 2ull * (column.length - saved) * column.batch_size > capacity)) {
            mpz_clears(divisor, residual, inverse, nullptr);
            return cudaSuccess;
        }
    }
    const uint32_t words = length - saved;
    // Newton lifting doubles the correct low bits at each iteration:
    // 1 - D*x' = (1 - D*x)^2 for x' = x*(2 - D*x).
    const mp_bitcnt_t bits = (mp_bitcnt_t)32 * words;
    mpz_set_ui(inverse, 1);
    for (mp_bitcnt_t n = 1; n < bits; n *= 2) {
        const mp_bitcnt_t next = std::min(2 * n, bits);
        mpz_fdiv_r_2exp(residual, divisor, next);
        mpz_mul(residual, residual, inverse);
        mpz_fdiv_r_2exp(residual, residual, next);
        mpz_ui_sub(residual, 2, residual);
        mpz_mul(inverse, inverse, residual);
        mpz_fdiv_r_2exp(inverse, inverse, next);
    }
    std::vector<uint32_t> host(words);
    mpz_export(host.data(), nullptr, -1, 4, 0, 0, inverse);
    mpz_clears(divisor, residual, inverse, nullptr);

    cudaError_t err = cudaMemcpy(scratch, host.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    for (auto *column : {&P, &Q, &R}) {
        if (!column->batch_size) continue;
        auto low = *column;
        low.length -= saved;
        BatchMPArray factor{scratch, low.length, low.batch_size, 0};
        BatchMPArray product{scratch + words, 2 * low.length, low.batch_size, 2 * low.length};
        err = batch_mp_mul(ctx, low, factor, product);
        if (err != cudaSuccess) return err;
        err = cudaMemcpy2DAsync(low.data, low.stride * sizeof(uint32_t),
            product.data, product.stride * sizeof(uint32_t), low.length * sizeof(uint32_t),
            low.batch_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return err;
        column->length = low.length;
    }
    return cudaSuccess;
}

// A subtree starting at term a contributes through the preceding product
// of Chudnovsky ratios. Each ratio, including its implicit power of two,
// is at most 1728 / 640320^3 < 10^-14.18. A joint limb cut introduces
// relative error below 2^(32*(1-keep)) in Q, and the same absolute
// bound in P/Q from rounding P. The 16 guard limbs cover the linear
// term, the partially occupied top limb, and rounding across the batch and forest merges.
struct SplitRow { BatchMPArray p, q, r; };

static cudaError_t trim_split_row(BatchMPContext *ctx, SplitRow &node,
    uint32_t term_offset, uint64_t digits) {
    for (auto *col : {&node.p, &node.q, &node.r}) {
        if (!col->data) continue;
        cudaError_t err = col->compact(ctx);
        if (err != cudaSuccess) return err;
    }
    if (!digits) return cudaSuccess;
    const int64_t remaining = (int64_t)digits - (int64_t)1418 * term_offset / 100;
    const uint32_t keep = 16 + (uint32_t)ceil(std::max<int64_t>(0, remaining) * (3.322 / 32.0));
    uint32_t drop = node.q.length > keep ? node.q.length - keep : 0;
    if (node.p.data) drop = std::min(drop, node.p.length ? node.p.length - 1 : 0);
    drop = std::min(drop, node.r.length);
    for (auto *col : {&node.p, &node.q, &node.r}) {
        if (col->data) {
            col->data += drop;
            col->length -= drop;
            col->stride = col->length;
        }
    }
    return cudaSuccess;
}

static cudaError_t finish_split(BatchMPContext *ctx, uint32_t rows,
    uint32_t terms, uint32_t term_offset, bool need_p, uint64_t digits, BatchMPArray &P, BatchMPArray &Q,
    BatchMPArray &R, BatchMPArray &p_next, BatchMPArray &q_next,
    BatchMPArray &r_next, uint32_t *rq_data, uint32_t *pr_data,
    StageProfiler *profiler) {
    auto view = [](uint32_t *ptr, uint32_t len) -> BatchMPArray {
        return {ptr, len, 1, len};
    };
    std::vector<SplitRow> current(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        if (need_p || i + 1 < rows) current[i].p = view(P.data + (size_t)i * P.stride, P.length);
        current[i].q = view(Q.data + (size_t)i * Q.stride, Q.length);
        current[i].r = view(R.data + (size_t)i * R.stride, R.length);
    }
    while (rows > 1) {
        auto *pd = p_next.data, *qd = q_next.data, *rd = r_next.data;
        const uint32_t span = terms / rows;
        for (uint32_t i = 0; i < rows; ++i) {
            cudaError_t err = trim_split_row(ctx, current[i], term_offset + i * span, digits);
            if (err != cudaSuccess) return err;
        }
        for (uint32_t i = 0; i < rows / 2; ++i) {
            auto left = current[2 * i], right = current[2 * i + 1];
            auto q = view(qd, left.q.length + right.q.length);
            auto rq = view(rq_data, left.r.length + right.q.length);
            auto pr = view(pr_data, left.p.length + right.r.length);
            BatchMPArray p{};
            cudaError_t err = profile_stage("mul", "trim_Q_RQ", rows, 1,
                left.q.length, right.q.length, true, profiler, [&]() {
                    return batch_mp_mul_shared_b(ctx, left.q, left.r, right.q, q, rq);
                });
            if (err != cudaSuccess) return err;
            if (right.p.data) {
                p = view(pd, left.p.length + right.p.length);
                pd += p.length;
                err = profile_stage("mul", "trim_P_PR", rows, 1, left.p.length,
                    right.p.length, true, profiler, [&]() {
                        return batch_mp_mul_shared_b(ctx, right.p, right.r, left.p, p, pr);
                    });
            } else {
                err = profile_mul(ctx, left.p, right.r, pr, "trim_PR", rows, profiler);
            }
            if (err != cudaSuccess) return err;
            const uint32_t shift = 15u * span;
            auto r = view(rd, std::max(rq.length + (shift + 31) / 32, pr.length) + 1);
            err = profile_shift_addsub(ctx, rq, pr, r, shift, false, rows, profiler);
            if (err != cudaSuccess) return err;
            current[i] = {p, q, r};
            qd += q.length;
            rd += r.length;
        }
        std::swap(P, p_next);
        std::swap(Q, q_next);
        std::swap(R, r_next);
        rows /= 2;
    }
    if (need_p) P = current.front().p;
    Q = current.front().q;
    R = current.front().r;
    cudaError_t err = Q.compact(ctx);
    return err == cudaSuccess ? R.compact(ctx) : err;
}

// do binary splitting for N * 17 elements, where N is a power of 2
cudaError_t binary_split_batched(BatchMPContext * context, int N, BatchMPArray &Q, BatchMPArray &R, StageProfiler * profiler = nullptr, uint32_t block_offset = 0, BatchMPArray *P_out = nullptr, uint64_t digits = 0){
    if (context == nullptr || N <= 0 || (N & (N - 1)) != 0) {
        return cudaErrorInvalidValue;
    }
    const uint32_t extra_common_limit = getenv("GPUPI_COMMON_FACTORS")
        ? (uint32_t)strtoul(getenv("GPUPI_COMMON_FACTORS"), nullptr, 10) : kLargeBlockTerms;
    // Reserve both root transforms and their carry scratch before building the
    // tree. Growing a multi-GiB allocation at the final merge is expensive.
    const uint32_t root_operand = std::min((uint32_t)N, 1u << 22) * 32u;
    const size_t reserve = batch_mul_ntt_shared_b_workspace_size(
        1, root_operand, root_operand, root_operand);
    cudaError_t workspace_err = batch_mp_ensure_workspace(context, reserve);
    if (workspace_err != cudaSuccess) return workspace_err;

    BatchMPArray P0 = {};
    BatchMPArray Q0 = {};
    BatchMPArray R0 = {};
    BatchMPArray P = {};
    Q = {};
    R = {};
    BatchMPArray P_next{};
    BatchMPArray Q_next{};
    BatchMPArray R_next{};
    BatchMPArray R_prod_1{};
    BatchMPArray R_prod_2{};

    uint32_t * temp_prod_workspace = nullptr;
    const size_t temp_words = (size_t)N * (3 + 4 + 6 + 88 + 72);
    cudaError_t temp_workspace_err = pi_memory::allocate(&temp_prod_workspace,
        temp_words * sizeof(uint32_t));
    if (temp_workspace_err != cudaSuccess || temp_prod_workspace == nullptr) {
        return temp_workspace_err != cudaSuccess ? temp_workspace_err : cudaErrorMemoryAllocation;
    }
    uint32_t * const temp_prod_workspace_base = temp_prod_workspace;
    
    auto release_array = [](BatchMPArray &array) {
        if (array.data != nullptr) {
            batch_mp_array_release(array);
            array = {};
        }
    };
    auto release_all = [&]() {
        release_array(P);
        release_array(Q);
        release_array(R);
        release_array(P_next);
        release_array(Q_next);
        release_array(R_next);
        pi_memory::release(temp_prod_workspace_base);
    };

    P0 = {.data = temp_prod_workspace, .length = 3, .batch_size = (uint32_t)N,  .stride = 3};
    temp_prod_workspace += N * 3;
    Q0 = {.data = temp_prod_workspace, .length = 4, .batch_size = (uint32_t)N,  .stride = 4};
    temp_prod_workspace += N * 4;
    R0 = {.data = temp_prod_workspace, .length = 6, .batch_size = (uint32_t)N,  .stride = 6};
    temp_prod_workspace += N * 6;

    P = batch_mp_array_create(N * 16, 3);
    Q = batch_mp_array_create(N * 16, 5); // we ask for 5 size, but will only use 4, for safety
    R = batch_mp_array_create(N * 16, 6);
    if (P0.data == nullptr || Q0.data == nullptr || R0.data == nullptr || P.data == nullptr || Q.data == nullptr || R.data == nullptr) {
        release_all();
        return cudaErrorMemoryAllocation;
    }
    Q.length = 4;
    Q.stride = 4;

    const int threads_per_block = 128;
    int num_blocks = min((N + threads_per_block - 1) / threads_per_block, 65535);
    cudaError_t err = profile_stage(
        "leaf",
        "construct_leaf",
        (uint32_t)N * 17,
        (uint32_t)N,
        P0.length,
        0,
        false,
        profiler,
        [&]() {
            construct_leaf_17<true><<<num_blocks, threads_per_block>>>(P0.data, Q0.data, R0.data, N, block_offset);
            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                return kernel_err;
            }
            return cudaDeviceSynchronize();
        }
    );
    CHECK_AND_RETURN(err, release_all());
    
    err = profile_stage(
        "leaf",
        "construct_leaf",
        (uint32_t)N * 17,
        (uint32_t)N * 16,
        P.length,
        0,
        false,
        profiler,
        [&]() {
            construct_leaf_17<false><<<num_blocks, threads_per_block>>>(P.data, Q.data, R.data, N * 16, block_offset);
            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                return kernel_err;
            }
            return cudaDeviceSynchronize();
        }
    );
    CHECK_AND_RETURN(err, release_all());
    int n = N * 16;
    P_next = batch_mp_array_create(n / 2, 6);
    Q_next = batch_mp_array_create(n / 2, 10);
    R_next = batch_mp_array_create(n / 2, 12);
    R_prod_1 = {.data = temp_prod_workspace, .length = 11, .batch_size = (uint32_t)(n / 2),  .stride = 11};
    temp_prod_workspace += (n / 2) * 11;
    R_prod_2 = {.data = temp_prod_workspace, .length = 9, .batch_size = (uint32_t)(n / 2),  .stride = 9};
    temp_prod_workspace += (n / 2) * 9;
    
    if (P_next.data == nullptr || Q_next.data == nullptr || R_next.data == nullptr ||
        R_prod_1.data == nullptr || R_prod_2.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }

    for (uint32_t n1 = (uint32_t)n; n1 > 1; n1 >>= 1){
        constexpr uint32_t trim_rows = 4;
        if (digits && N >= (1 << 20) && n1 == trim_rows) {
            err = finish_split(context, n1, (uint32_t)N * 17, block_offset * 17, P_out != nullptr, digits, P, Q, R,
                P_next, Q_next, R_next, R_prod_1.data, R_prod_2.data, profiler);
            CHECK_AND_RETURN(err, release_all());
            break;
        }
        /*
        p = p1 * p2
        q = q1 * q2
        r1 = r1 * q2
        r2 = p1 * r2
        if (k - i) & 1:
            r = r1 - r2
        else:
            r = r1 + r2*/
        const bool subtract = (n1 == (uint32_t)n) || (n1 == (n >> 4));

        int ignore_last_P = P_out == nullptr && n1 <= N;

        P_next.batch_size = n1 / 2 - ignore_last_P;
        P_next.length = P.length * 2;
        P_next.stride = P.length * 2;

        BatchMPArray P_even = {
            .data = P.data,
            .length = P.length,
            .batch_size = n1 / 2 - ignore_last_P,
            .stride = P.stride * 2
        };
        BatchMPArray P_odd = {
            .data = P.data + P.stride,
            .length = P.length,
            .batch_size = n1 / 2 - ignore_last_P,
            .stride = P.stride * 2
        };


        Q_next.batch_size = n1 / 2;
        Q_next.length = Q.length * 2;
        Q_next.stride = Q.length * 2;

        BatchMPArray Q_even = {
            .data = Q.data,
            .length = Q.length,
            .batch_size = n1 / 2,
            .stride = Q.stride * 2
        };
        BatchMPArray Q_odd = {
            .data = Q.data + Q.stride,
            .length = Q.length,
            .batch_size = n1 / 2,
            .stride = Q.stride * 2
        };
        BatchMPArray R_even = {
            .data = R.data,
            .length = R.length,
            .batch_size = n1 / 2,
            .stride = R.stride * 2
        };
        BatchMPArray R_odd = {
            .data = R.data + R.stride,
            .length = R.length,
            .batch_size = n1 / 2,
            .stride = R.stride * 2
        };
        R_prod_1.batch_size = n1 / 2;
        R_prod_1.length = R.length + Q.length;
        R_prod_1.stride = R.length + Q.length;

        const uint32_t shift_amount = 15u * ((n / n1) + ((n / n1) >> 4));
        if (N >= (1 << 22) && n1 <= (uint32_t)N / 512 && n1 >= 4) {
            auto shifted = R_prod_1;
            shifted.length = shifted.stride = R.length + shift_amount / 32;
            R_next.batch_size = Q_next.batch_size;
            R_next.length = R_next.stride = std::max(shifted.length + Q.length, P.length + R.length) + 1;
            err = profile_stage("mul", "merged_PQR", n1, Q_next.batch_size,
                P.length, Q.length, true, profiler, [&]() {
                    cudaError_t status = batch_mp_shift_bits(context, R_even, shifted, shift_amount);
                    if (status != cudaSuccess) return status;
                    return batch_mp_split_merge(context, P, Q, R, shifted, P_next, Q_next, R_next);
                });
            if (err != cudaErrorNotSupported) {
                CHECK_AND_RETURN(err, release_all());
                // These are upper bounds. The final rows are compacted before
                // selecting precision, so intermediate scans are unnecessary.
                std::swap(P, P_next);
                std::swap(Q, Q_next);
                std::swap(R, R_next);
                continue;
            }
        }
        err = profile_stage("mul", "Q_and_RQ", n1, Q_even.batch_size,
            Q_even.length, Q_odd.length, true, profiler, [&]() {
                return batch_mp_mul_shared_b(context, Q_even, R_even, Q_odd, Q_next, R_prod_1);
            });
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, Q_next, "Q", n1, profiler);
        CHECK_AND_RETURN(err, release_all());



        R_prod_2.batch_size = n1 / 2;
        R_prod_2.length = R.length + P.length;
        R_prod_2.stride = R.length + P.length;

        const uint32_t p_rows = P_odd.batch_size;
        P_even.batch_size = n1 / 2;
        if (p_rows > 0 && !getenv("GPUPI_OLD_P_PAIR")) {
            auto common = P_even, other = R_odd, product = R_prod_2;
            common.batch_size = other.batch_size = product.batch_size = p_rows;
            err = profile_stage("mul", "P_and_PR", n1, p_rows, P_odd.length,
                common.length, true, profiler, [&]() {
                    return batch_mp_mul_shared_b(context, P_odd, other, common, P_next, product);
                });
            CHECK_AND_RETURN(err, release_all());
            if (p_rows < R_prod_2.batch_size) {
                auto tail = [p_rows](BatchMPArray v) {
                    v.data += (size_t)p_rows * v.stride; v.batch_size = 1; return v;
                };
                err = profile_mul(context, tail(P_even), tail(R_odd), tail(R_prod_2),
                    "PR_tail", n1, profiler);
                CHECK_AND_RETURN(err, release_all());
            }
        } else {
            if (p_rows > 0) {
                auto even = P_even; even.batch_size = p_rows;
                err = profile_mul(context, even, P_odd, P_next, "P", n1, profiler);
                CHECK_AND_RETURN(err, release_all());
            }
            err = profile_mul(context, P_even, R_odd, R_prod_2, "P*R", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
        }
        if (p_rows > 0) {
            err = profile_compact(context, P_next, "P", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
        }
        const uint32_t shifted_r_prod_1_length = R_prod_1.length + (((uint32_t)shift_amount) >> 5) + ((shift_amount & 31) != 0);
        R_next.batch_size = n1 / 2;
        R_next.length = std::max(shifted_r_prod_1_length, R_prod_2.length) + (subtract ? 0 : 1);
        R_next.stride = R_next.length;

        err = profile_shift_addsub(context, R_prod_1, R_prod_2, R_next, (uint32_t)shift_amount, subtract, n1, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, R_next, "R", n1, profiler);
        CHECK_AND_RETURN(err, release_all());
        std::swap(P, P_next);
        std::swap(Q, Q_next);
        std::swap(R, R_next);

        if (n1 == (n >> 3)){
            // we will add P0 here
            P_next.batch_size = n1 / 2;
            P_next.length = P.length + P0.length;
            P_next.stride = P.length + P0.length;
            err = profile_mul(context, P, P0, P_next, "P", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_compact(context, P_next, "P", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            
            Q_next.batch_size = n1 / 2;
            Q_next.length = Q.length + Q0.length;
            Q_next.stride = Q.length + Q0.length;
            err = profile_mul(context, Q, Q0, Q_next, "Q", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_compact(context, Q_next, "Q", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            R_prod_1.batch_size = n1 / 2;
            R_prod_1.length = R.length + Q0.length;
            R_prod_1.stride = R.length + Q0.length;
            err = profile_mul(context, R, Q0, R_prod_1, "R*Q", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            R_prod_2.batch_size = n1 / 2;
            R_prod_2.length = P.length + R0.length;
            R_prod_2.stride = P.length + R0.length;
            err = profile_mul(context, P, R0, R_prod_2, "P*R", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            R_next.batch_size = n1 / 2;
            R_next.length = std::max(R_prod_1.length + 1, R_prod_2.length) + 1;
            R_next.stride = R_next.length;
            err = profile_shift_addsub(context, R_prod_1, R_prod_2, R_next, 15u, false, n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_compact(context, R_next, "R", n1, profiler);
            CHECK_AND_RETURN(err, release_all());

            err = profile_exactdiv(context, P_next, "P", kBlockCommonDivisorLo, n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_exactdiv(context, P_next, "P", kBlockCommonDivisorHi, n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_compact(context, P_next, "P", n1, profiler);
            CHECK_AND_RETURN(err, release_all());

            err = profile_exactdiv(context, Q_next, "Q", kBlockCommonDivisorLo, n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_exactdiv(context, Q_next, "Q", kBlockCommonDivisorHi, n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_compact(context, Q_next, "Q", n1, profiler);
            CHECK_AND_RETURN(err, release_all());

            err = profile_exactdiv(context, R_next, "R", kBlockCommonDivisorLo, n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_exactdiv(context, R_next, "R", kBlockCommonDivisorHi, n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_compact(context, R_next, "R", n1, profiler);
            CHECK_AND_RETURN(err, release_all());

            std::swap(P, P_next);
            std::swap(Q, Q_next);
            std::swap(R, R_next);
        }
        // At this tree depth a shorter P product can reuse the first half
        // of the transform for its larger sibling. Amortize the division
        // over the upper levels of large trees.
        if (N >= (1 << 22) && extra_common_limit >= kLargeBlockTerms &&
            n1 == (uint32_t)N / (kLargeBlockTerms / 34)) {
            err = profile_stage("exactdiv", "block_factor", n1, R.batch_size,
                R.length, kLargeBlockTerms, false, profiler, [&]() {
                    return cancel_block_factor(context, kLargeBlockTerms, P, Q, R,
                        temp_prod_workspace_base, temp_words);
                });
            CHECK_AND_RETURN(err, release_all());
        }
        if (n1 <= (uint32_t)N && extra_common_limit >= 34) {
            const uint64_t terms = (uint64_t)N * 34 / n1;
            const uint32_t *factors = nullptr; size_t count = 0;
            if (terms <= extra_common_limit) {
                if (terms == 34) { factors = kCommon34; count = sizeof(kCommon34)/4; }
                if (terms == 68) { factors = kCommon68; count = sizeof(kCommon68)/4; }
                if (terms == 136) { factors = kCommon136; count = sizeof(kCommon136)/4; }
            }
            if (count) {
                BatchMPArray arrays[] = {P, Q, R};
                const char *labels[] = {"common_P", "common_Q", "common_R"};
                for (int which = 0; which < 3; ++which) {
                    if (!arrays[which].batch_size) continue;
                    err = profile_stage("exactdiv", labels[which], n1,
                        arrays[which].batch_size, arrays[which].length, (uint32_t)count,
                        false, profiler, [&]() {
                            return batch_mp_exactdiv_small_many(context, arrays[which], factors, (uint32_t)count);
                        });
                    CHECK_AND_RETURN(err, release_all());
                }
            }
            if (count) {
                if (P.batch_size) {
                    err = profile_compact(context, P, "extra_P", n1, profiler);
                    CHECK_AND_RETURN(err, release_all());
                }
                err = profile_compact(context, Q, "extra_Q", n1, profiler);
                CHECK_AND_RETURN(err, release_all());
                err = profile_compact(context, R, "extra_R", n1, profiler);
                CHECK_AND_RETURN(err, release_all());
            }
        }
    }
    if (P_out) { *P_out = P; P = {}; }
    release_array(P);
    release_array(P_next);
    release_array(Q_next);
    release_array(R_next);
    pi_memory::release(temp_prod_workspace_base);
    return cudaSuccess;
}

namespace {

void release_array(BatchMPArray * array) {
    if (array != nullptr && array->data != nullptr) {
        batch_mp_array_release(*array);
        *array = {};
    }
}

bool is_power_of_two_int(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool parse_int_arg(const char * text, int * value) {
    if (text == nullptr || value == nullptr) {
        return false;
    }
    char * end = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *value = (int)parsed;
    return true;
}

bool parse_digits_arg(const char *text, uint64_t *value) {
    if (!text || !*text || *text == '-') return false;
    errno = 0;
    char *end = nullptr;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end || parsed > 5000000000ull) return false;
    *value = (uint64_t)parsed;
    return true;
}

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s <N> [--benchmark|--profile-stages]\n", program);
    fprintf(stderr, "Dumps chunk P/Q/R for indices in [begin, end); end-begin must be a power of 2.\n");
    fprintf(stderr, "Use --benchmark to print only the binary-splitting GPU time in milliseconds.\n");
    fprintf(stderr, "Use --profile-stages to print timed leaf/add/sub/multiply stages within binary splitting.\n");
}

bool check_cuda(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

bool copy_device_array_to_host(const BatchMPArray & array, uint32_t ** host_data) {
    if (host_data == nullptr) {
        return false;
    }
    *host_data = nullptr;
    if (array.length == 0) {
        return true;
    }
    uint32_t * buffer = new uint32_t[array.length];
    if (!check_cuda(cudaMemcpy(buffer, array.data, array.length * sizeof(uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy(DeviceToHost)")) {
        delete [] buffer;
        return false;
    }
    *host_data = buffer;
    return true;
}

uint32_t compute_mod_hash(const uint32_t *limbs, uint32_t length, uint64_t base = kLimbBaseMod) {
    uint64_t value = 0;
    for (int i = (int)length - 1; i >= 0; --i) {
        value = (value * base + limbs[i]) % kValidationMod;
    }
    return (uint32_t)value;
}

uint32_t compute_decimal_hash(const char * digits, uint32_t length) {
    uint64_t value = 3u;
    for (uint32_t i = 0; i < length; ++i) {
        value = (value * 10u + (uint32_t)(digits[i] - '0')) % kValidationMod;
    }
    return (uint32_t)value;
}

void print_hex_preview_with_hash(const char * label, const uint32_t * limbs, uint32_t length) {
    const uint32_t hash = compute_mod_hash(limbs, length);
    if (length == 0) {
        printf("%s = 0 hash = %u\n", label, hash);
        return;
    }
    const uint32_t first_count = std::min(length, 2u);
    const uint32_t last_count = std::min(length, 2u);
    printf("%s = ", label);
    for (uint32_t i = 0; i < last_count; ++i) {
        const uint32_t limb_index = length - 1 - i;
        if (i == 0) {
            printf("%x", limbs[limb_index]);
        } else {
            printf("%08x", limbs[limb_index]);
        }
    }
    if (length > first_count + last_count) {
        printf("...");
    }
    for (uint32_t i = first_count; i > 0; --i) {
        const uint32_t limb_index = i - 1;
        if (length <= first_count + last_count && limb_index >= length - last_count) {
            continue;
        }
        printf("%08x", limbs[limb_index]);
    }
    printf(" hash = %u\n", hash);
}

bool print_validation_summary(const BatchMPArray &Q, const BatchMPArray &R) {
    uint32_t * q_host = nullptr;
    uint32_t * r_host = nullptr;
    if (!copy_device_array_to_host(Q, &q_host) || !copy_device_array_to_host(R, &r_host)) {
        delete [] q_host;
        delete [] r_host;
        return false;
    }
    print_hex_preview_with_hash("Q", q_host, Q.length);
    print_hex_preview_with_hash("R", r_host, R.length);
    delete [] q_host;
    delete [] r_host;
    return true;
}

constexpr uint32_t kHashThreads = 256;
constexpr uint32_t kHashBlockLimbs = 16 * kHashThreads;

__host__ __device__ constexpr uint64_t limb_base_power(uint32_t n) {
    uint64_t product = 1, base = kLimbBaseMod;
    while (n) {
        if (n & 1u) product = product * base % kValidationMod;
        base = base * base % kValidationMod;
        n >>= 1;
    }
    return product;
}

// Evaluate each block as a base-2^32 integer modulo kValidationMod.
// Thread columns read coalesced limbs; adjacent columns then combine
// with the same polynomial identity as adjacent blocks on the host.
__global__ void hash_limb_blocks(const uint32_t *limbs, uint32_t length, uint32_t *hashes) {
    __shared__ uint32_t partial[kHashThreads];
    const uint32_t lane = threadIdx.x;
    const size_t first = (size_t)blockIdx.x * kHashBlockLimbs + lane;
    uint64_t h = 0;
    for (int i = kHashBlockLimbs - kHashThreads; i >= 0; i -= (int)kHashThreads) {
        const size_t index = first + i;
        const uint32_t word = index < length ? limbs[index] : 0;
        h = (h * limb_base_power(kHashThreads) + word) % kValidationMod;
    }
    partial[lane] = h;
    uint64_t weight = kLimbBaseMod;
    #pragma unroll
    for (uint32_t step = 1; step < kHashThreads; step *= 2) {
        __syncthreads();
        if (lane % (2 * step) == 0)
            partial[lane] = (partial[lane] + weight * partial[lane + step]) % kValidationMod;
        weight = weight * weight % kValidationMod;
    }
    if (!lane) hashes[blockIdx.x] = partial[0];
}

bool print_fractional_preview_summary(const BatchMPArray &ret) {
    if (ret.length <= kHashBlockLimbs) {
        uint32_t *host = nullptr;
        if (!copy_device_array_to_host(ret, &host)) return false;
        print_hex_preview_with_hash("RET", host, ret.length);
        delete [] host;
        return true;
    }

    const uint32_t blocks = (ret.length - 1) / kHashBlockLimbs + 1;
    BatchMPArray partial = batch_mp_array_create(1, blocks);
    if (!partial.data) return false;
    hash_limb_blocks<<<blocks, kHashThreads>>>(ret.data, ret.length, partial.data);
    std::vector<uint32_t> hashes(blocks);
    uint32_t low[2], high[2];
    const bool ok = check_cuda(cudaGetLastError(), "hash_limb_blocks") &&
        check_cuda(cudaMemcpy(hashes.data(), partial.data, blocks * sizeof(uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy(block hashes)") &&
        check_cuda(cudaMemcpy(low, ret.data, sizeof(low), cudaMemcpyDeviceToHost), "cudaMemcpy(low limbs)") &&
        check_cuda(cudaMemcpy(high, ret.data + ret.length - 2, sizeof(high),
            cudaMemcpyDeviceToHost), "cudaMemcpy(high limbs)");
    batch_mp_array_release(partial);
    if (!ok) return false;

    const uint32_t hash = compute_mod_hash(hashes.data(), blocks, limb_base_power(kHashBlockLimbs));
    printf("RET = %x%08x...%08x%08x hash = %u\n", high[1], high[0], low[1], low[0], hash);
    return true;
}

uint64_t round_up_decimal_digits(uint64_t requested_digits) {
    if (requested_digits == 0u) {
        return 0u;
    }
    uint64_t digits = requested_digits;
    uint32_t levels = 0;
    while (digits >= 32 * 18){
        digits = digits >> 1;
        levels ++;
    }
    if ((digits << levels) < requested_digits) {
        digits ++;
    }
    kDecimalLeafDigits = digits;
    return digits << levels;
}

uint32_t compute_prec_limbs_for_digits(uint64_t digits) {
    const uint64_t prec_bits = (uint64_t)((double)digits * 3.322) + 31u;
    // The 3.322 estimate supplies ample margin for large requests, but almost
    // none at small sizes. Allow for a partially filled leading denominator
    // limb, quotient truncation and the decimal converter's guard bits.
    const uint64_t guarded_bits =
        (uint64_t)ceil((double)digits * 3.32192809488736234787) + 128u;
    return std::max(prec_bits / 32u + 1u, (guarded_bits + 31u) / 32u);
}

uint32_t limb_bit_length(uint32_t limb) {
    return limb == 0 ? 0u : 32u - (uint32_t)__builtin_clz(limb);
}

uint64_t compute_bit_length(const uint32_t * limbs, uint32_t length) {
    while (length > 0 && limbs[length - 1] == 0) {
        --length;
    }
    if (length == 0) {
        return 0;
    }
    return (uint64_t)(length - 1) * 32u + limb_bit_length(limbs[length - 1]);
}

cudaError_t copy_top_limb_to_host(const BatchMPArray & array, uint32_t * top_limb) {
    if (top_limb == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (array.length == 0) {
        *top_limb = 0u;
        return cudaSuccess;
    }
    return cudaMemcpy(top_limb, array.data + array.length - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

cudaError_t compact_and_compute_bit_length(BatchMPContext * context, BatchMPArray & array, uint64_t * bit_length) {
    if (bit_length == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = array.compact(context);
    if (err != cudaSuccess) {
        return err;
    }
    if (array.length == 0) {
        *bit_length = 0u;
        return cudaSuccess;
    }
    uint32_t top_limb = 0u;
    err = copy_top_limb_to_host(array, &top_limb);
    if (err != cudaSuccess) {
        return err;
    }
    *bit_length = (uint64_t)(array.length - 1) * 32u + limb_bit_length(top_limb);
    return cudaSuccess;
}

uint32_t mod_mul_u32(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * b) % kValidationMod);
}

uint32_t mod_sub_u32(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (uint32_t)(a + kValidationMod - b);
}

uint64_t compute_low64(const uint32_t * limbs, uint32_t length) {
    if (length == 0) {
        return 0;
    }
    uint64_t value = limbs[0];
    if (length > 1) {
        value |= (uint64_t)limbs[1] << 32;
    }
    return value;
}

bool print_sqrt_validation_summary(const BatchMPArray &P, const BatchMPArray &Q) {
    uint32_t * p_host = nullptr;
    uint32_t * q_host = nullptr;
    if (!copy_device_array_to_host(P, &p_host) || !copy_device_array_to_host(Q, &q_host)) {
        delete [] p_host;
        delete [] q_host;
        return false;
    }

    print_hex_preview_with_hash("P_preview", p_host, P.length);
    print_hex_preview_with_hash("Q_preview", q_host, Q.length);

    const uint32_t p_hash = compute_mod_hash(p_host, P.length);
    const uint32_t q_hash = compute_mod_hash(q_host, Q.length);
    const uint32_t norm_mod = mod_sub_u32(mod_mul_u32(p_hash, p_hash), mod_mul_u32(10005u, mod_mul_u32(q_hash, q_hash)));

    const uint64_t p_low64 = compute_low64(p_host, P.length);
    const uint64_t q_low64 = compute_low64(q_host, Q.length);
    const uint64_t norm_low64 = (uint64_t)((unsigned __int128)p_low64 * p_low64 - (unsigned __int128)10005u * q_low64 * q_low64);

    printf(
        "validation P_bits=%llu Q_bits=%llu norm_mod_%u=%u norm_low64=0x%016llx expected_norm=1\n",
        (unsigned long long)compute_bit_length(p_host, P.length),
        (unsigned long long)compute_bit_length(q_host, Q.length),
        kValidationMod,
        norm_mod,
        (unsigned long long)norm_low64
    );

    delete [] p_host;
    delete [] q_host;
    return true;
}

bool print_full_hex_value(const char * label, const BatchMPArray & array) {
    uint32_t * host = nullptr;
    if (!copy_device_array_to_host(array, &host)) {
        return false;
    }
    printf("%s = ", label);
    if (array.length == 0) {
        printf("0\n");
        delete [] host;
        return true;
    }
    for (int i = (int)array.length - 1; i >= 0; --i) {
        if (i == (int)array.length - 1) {
            printf("%x", host[i]);
        } else {
            printf("%08x", host[i]);
        }
    }
    printf("\n");
    delete [] host;
    return true;
}

cudaError_t shift_right_large(
    BatchMPContext * context,
    BatchMPArray A,
    uint64_t shift_bits,
    BatchMPArray B,
    const char * label = nullptr,
    uint64_t digits = 0u,
    StageProfiler * profiler = nullptr
) {
    const uint64_t limb_shift64 = shift_bits / 32u;
    const uint32_t bit_shift = (uint32_t)(shift_bits % 32u);
    if (limb_shift64 >= A.length) {
        return cudaErrorInvalidValue;
    }
    const uint32_t limb_shift = (uint32_t)limb_shift64;
    BatchMPArray A_preshift = {
        .data = A.data + limb_shift,
        .length = A.length - limb_shift,
        .batch_size = A.batch_size,
        .stride = A.stride - limb_shift
    };
    if (profiler == nullptr || label == nullptr) {
        return batch_mp_shift_bits(context, A_preshift, B, -(int32_t)bit_shift);
    }
    return profile_shift_bits(context, A_preshift, B, -(int32_t)bit_shift, label, digits, profiler);
}

BatchMPArray make_subview(const BatchMPArray & array, uint32_t start_limb, uint32_t length) {
    if (start_limb > array.length) {
        start_limb = array.length;
    }
    const uint32_t max_length = array.length - start_limb;
    if (length > max_length) {
        length = max_length;
    }
    const uint32_t stride = (array.stride >= start_limb) ? (array.stride - start_limb) : 0u;
    return {
        .data = array.data + start_limb,
        .length = length,
        .batch_size = array.batch_size,
        .stride = stride
    };
}

cudaError_t load_mpz_from_device(const BatchMPArray & array, mpz_t value) {
    std::vector<uint32_t> host(array.length == 0 ? 1u : array.length, 0u);
    if (array.length > 0) {
        cudaError_t err = cudaMemcpy(host.data(), array.data, array.length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return err;
        }
    }
    mpz_import(value, array.length, -1, sizeof(uint32_t), 0, 0, host.data());
    return cudaSuccess;
}

cudaError_t store_mpz_to_device(const mpz_t value, BatchMPArray & array) {
    std::vector<uint32_t> host(array.stride == 0 ? 1u : array.stride, 0u);
    size_t written = 0u;
    mpz_export(host.data(), &written, -1, sizeof(uint32_t), 0, 0, value);
    uint32_t used = (uint32_t)written;
    if (used == 0u) {
        used = 1u;
    }
    if (used > array.stride) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaMemcpy(array.data, host.data(), array.stride * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return err;
    }
    array.length = used;
    return cudaSuccess;
}

const uint64_t inverse_cpu_limit = 32 * 1024;

/*
def recursive_inverse(x, L):
    Ls = [L]
    while Ls[-1] > 256:
        m = (Ls[-1] - 2) // 4
        m -= m & 31
        Ls.append(Ls[-1] - m * 2)
    def get_chunk(x, l): # Can be implemented as slicing as ((L - l) // 2) is always multiple of 32
        return x >> ((L - l) // 2)
    L0 = Ls[-1]
    total = mpz(1) << L0
    y, res = gmpy2.f_divmod(total, get_chunk(x, L0))

    for L1, L2 in zip(reversed(Ls), reversed(Ls[:-1])):
        # go from L1 to L2
        m = (L2 - L1) // 2
        x2 = get_chunk(x, L2) & ((mpz(1) << m) - 1)  # Can be implemented as slicing as m is multiple of 32
        x2y1 = x2 * y
        res1_m = res << m
        if res1_m >= x2y1:
            res2 = res1_m - x2y1
            y2 = (res2 * y) >> (L2 - m * 2)
            if L2 != L:
                res = (res2 << m) - y2 * get_chunk(x, L2)
            y = (y << m) + y2
        else:
            res2 = x2y1 - res1_m
            y2 = ((res2 * y) >> (L2 - m * 2)) + 1
            if L2 != L:
                res = y2 * get_chunk(x, L2) - (res2 << m)
            y = (y << m) - y2
    
    return y
*/
cudaError_t recursive_inverse(
    BatchMPContext * context,
    BatchMPArray x, // we must have x.batch_size == 1
    uint64_t L,
    BatchMPArray & y_out,
    uint64_t digits,
    StageProfiler * profiler
) {
    std::vector<uint64_t> levels;
    levels.push_back(L);
    while (levels.back() > inverse_cpu_limit) {
        const uint64_t m = ((levels.back() - 2u) / 4u) & ~31ull;
        levels.push_back(levels.back() - m * 2u);
    }

    size_t L_res = (L + 31) / 32 + 2;
    size_t L_res2 = (L + 31) / 32 + 1;
    size_t L_y2 = x.length + 1;

    y_out = batch_mp_array_create(1, x.length);
    if (y_out.data == nullptr) return cudaErrorMemoryAllocation;
    cudaError_t err = cudaMemset(y_out.data, 0, y_out.length * sizeof(uint32_t));
    if (err != cudaSuccess) return err;

    size_t total_workspace = L_res + L_res2 + L_y2;
    uint32_t * workspace_base = nullptr;
    err = pi_memory::allocate(&workspace_base, total_workspace * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    auto release_all = [&]() {
        pi_memory::release(workspace_base);
    };
    uint32_t * workspace_cur = workspace_base;

    uint32_t * res_base = workspace_cur;
    workspace_cur += L_res;
    uint32_t * res2_base = workspace_cur;
    workspace_cur += L_res2;
    uint32_t * y2_base = workspace_cur;
    workspace_cur += L_y2;

    err = cudaMemset(workspace_base, 0, L_res * sizeof(uint32_t));
    CHECK_AND_RETURN(err, release_all());

    // We put y to the right end of the L_y space so that (y << m) is no-op
    // the same for res

    BatchMPArray res;

    {
        const uint64_t L0 = levels.back();
        const uint32_t start_limb = (uint32_t)((L - L0) >> 6);
        BatchMPArray x0 = make_subview(x, start_limb, x.length - start_limb);
        BatchMPArray y0 = make_subview(y_out, start_limb, y_out.length - start_limb);
        res = {
            .data = res_base + start_limb * 2,
            .length = x0.length + 1,
            .batch_size = 1,
            .stride = x0.length + 1
        };
        mpz_t total, x0_mpz, y0_mpz, res0_mpz;
        
        mpz_inits(total, x0_mpz, y0_mpz, res0_mpz, nullptr);
        cudaError_t err = load_mpz_from_device(x0, x0_mpz);
        if (err != cudaSuccess) {
            mpz_clears(total, x0_mpz, y0_mpz, res0_mpz, nullptr);
            release_all();
            return err;
        }
        mpz_set_ui(total, 1u);
        mpz_mul_2exp(total, total, L0);
        mpz_fdiv_qr(y0_mpz, res0_mpz, total, x0_mpz);
        err = store_mpz_to_device(y0_mpz, y0);
        if (err == cudaSuccess) {
            err = store_mpz_to_device(res0_mpz, res);
        }
        mpz_clears(total, x0_mpz, y0_mpz, res0_mpz, nullptr);
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
    }
    err = res.compact(context);
    CHECK_AND_RETURN(err, release_all());

    for (int i = (int)(levels.size()) - 1; i > 0; i --){
        size_t L1 = levels[i];
        size_t L2 = levels[i - 1];
        size_t m = (L2 - L1) >> 1;
        BatchMPArray x2 = make_subview(x, (L - L2) >> 6, m >> 5);
        BatchMPArray y1 = make_subview(y_out, (L - L1) >> 6, y_out.length - ((L - L1) >> 6));
        BatchMPArray x2y1 = {
            .data = res2_base,
            .length = x2.length + y1.length,
            .batch_size = 1,
            .stride = x2.length + y1.length
        };
        err = profile_mul(context, x2, y1, x2y1, "x2*y", L2 >> 5, profiler);
        CHECK_AND_RETURN(err, release_all());
        BatchMPArray res1_m = {
            .data = res.data - (uint32_t)(m >> 5),
            .length = res.length + (uint32_t)(m >> 5) + 1,
            .batch_size = 1,
            .stride = res.length + (uint32_t)(m >> 5) + 1,
        };
        err = profile_sub(context, res1_m, x2y1, res1_m, "res1_m", L2 >> 5, profiler);
        CHECK_AND_RETURN(err, release_all());
        uint32_t top_limb = 0u;
        err = cudaMemcpy(&top_limb, res1_m.data + res1_m.length - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        CHECK_AND_RETURN(err, release_all());
        bool is_negative = false;
        if (top_limb >= 0x80000000u){
            is_negative = true;
            err = profile_sub(context, {
                .data = res1_m.data,
                .length = 0,
                .batch_size = 1,
                .stride = 0
            }, res1_m, res1_m, "res1_m_neg", L2 >> 5, profiler);
            CHECK_AND_RETURN(err, release_all());
        }
        BatchMPArray res2 = {
            .data = res2_base,
            .length = res1_m.length + y1.length,
            .batch_size = 1,
            .stride = res1_m.length + y1.length
        };
        // Only floor(res1_m*y1 / 2^L1) is consumed. Removing d low
        // limbs from res1_m omits less than B^(d+y1.length), B=2^32.
        // Two extra limbs below the binary point let us certify the exact
        // quotient unless the upper guard limb can propagate a carry.
        const uint32_t cut = L1 / 32;
        const uint32_t drop = cut > y1.length + 2 && y1.length >= 4096
            ? cut - y1.length - 2 : 0;
        uint32_t omitted = 0;
        if (drop && drop + 1 < res1_m.length) {
            auto high = make_subview(res1_m, drop, res1_m.length - drop);
            auto product = res2;
            product.length -= drop;
            err = profile_mul(context, high, y1, product, "res2_high*y", L2 >> 5, profiler);
            CHECK_AND_RETURN(err, release_all());
            uint32_t guard;
            err = cudaMemcpy(&guard, product.data + cut - drop - 1, sizeof(guard),
                cudaMemcpyDeviceToHost);
            CHECK_AND_RETURN(err, release_all());
            if (guard != UINT32_MAX) omitted = drop;
        }
        if (!omitted) {
            err = profile_mul(context, res1_m, y1, res2, "res2*y", L2 >> 5, profiler);
            CHECK_AND_RETURN(err, release_all());
        }
        BatchMPArray y2 = {
            .data = y2_base,
            .length = res2.length - cut,
            .batch_size = 1,
            .stride = res2.length - cut
        };
        res2.length -= omitted;
        err = shift_right_large(context, res2, L1 - (uint64_t)omitted * 32, y2, "y2", L2 >> 5, profiler);
        CHECK_AND_RETURN(err, release_all());
        if (is_negative){
            err = profile_add_small(context, y2, 1, y2, "y2_inc", L2 >> 5, profiler);
            CHECK_AND_RETURN(err, release_all());
        }

        BatchMPArray y_new = make_subview(y_out, (L - L2) >> 6, y_out.length - ((L - L2) >> 6));
        if (!is_negative) {
            err = profile_add(context, y_new, y2, y_new, "(y<<m)+y2", L2 >> 5, profiler);
        } else {
            err = profile_sub(context, y_new, y2, y_new, "(y<<m)-y2", L2 >> 5, profiler);
        }
        CHECK_AND_RETURN(err, release_all());
        if (L2 != L){
            BatchMPArray x_next = make_subview(x, (L - L2) >> 6, x.length - ((L - L2) >> 6));
            res = {
                .data = res1_m.data - (uint32_t)(m >> 5),
                .length = res1_m.length + (uint32_t)(m >> 5),
                .batch_size = 1,
                .stride = res1_m.length + (uint32_t)(m >> 5)
            };
            BatchMPArray y2x = {
                .data = res2_base,
                .length = y2.length + x_next.length,
                .batch_size = 1,
                .stride = y2.length + x_next.length
            };
            err = profile_mul(context, y2, x_next, y2x, "y2*x", L2 >> 5, profiler);
            CHECK_AND_RETURN(err, release_all());
            if (!is_negative){
                err = profile_sub(context, res, y2x, res, "(res<<m)-y2x", L2 >> 5, profiler);
            }else{
                err = profile_sub(context, y2x, res, res, "y2x-(res<<m)", L2 >> 5, profiler);
            }
            CHECK_AND_RETURN(err, release_all());
            err = res.compact(context);
            CHECK_AND_RETURN(err, release_all());
        }
    }
    pi_memory::release(workspace_base);

    return cudaSuccess;
}

// P is odd and P^2 - 10005 Q^2 = 1. Thus the Pell doubling
// recurrence is P' = 2 P^2 - 1; its subtraction cannot borrow beyond limb 0.
__global__ void pell_double_square(const uint32_t *pp, uint32_t *out, uint32_t length) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i <= length;
         i += (size_t)blockDim.x * gridDim.x) {
        uint32_t value = (i < length ? pp[i] << 1 : 0) |
            (i > 0 ? pp[i - 1] >> 31 : 0);
        out[i] = value - (i == 0);
    }
}

cudaError_t sqrt_10005(BatchMPContext * context, uint64_t target_digits, BatchMPArray & P, BatchMPArray & Q, StageProfiler * profiler = nullptr) {
    cudaError_t workspace_err = batch_mp_ensure_workspace(context, 6442450944ull >> 2);
    if (workspace_err != cudaSuccess) return workspace_err;
    /*
    p_15 = mpz(4001)
    q_15 = mpz(40)
    valid_terms = 7.5

    while valid_terms < target_digits:
        valid_terms *= 2
        p_15, q_15 = (
            2 * p_15 * p_15 - 1,
            p_15 * q_15 * 2,
        )
    */
    double log_p_15 = log(4001.0);
    double log_q_15 = log(40.0);
    double valid_digits = 7.5;
    size_t total_work_memory = 0;
    size_t P_memory = 0, Q_memory = 0;
    if (valid_digits < target_digits){
        total_work_memory += 4;
        total_work_memory += 4;
    }else{
        P_memory = 4;
        Q_memory = 4;
    }
    while (valid_digits < target_digits){
        double old_log_p_15 = log_p_15;
        double old_log_q_15 = log_q_15;
        log_p_15 = old_log_p_15 * 2 + log1p(10005.0 * exp(2 * old_log_q_15 - 2 * old_log_p_15));
        log_q_15 = old_log_p_15 + old_log_q_15 + log(2.0);
        valid_digits *= 2;
        if (valid_digits < target_digits){
            total_work_memory += (ceil(log_p_15 / log(2.0) / 32) * 4 + 8) * 3;
            total_work_memory += (ceil(log_q_15 / log(2.0) / 32) * 4 + 8) * 1;
        }else{
            total_work_memory += (ceil(log_p_15 / log(2.0) / 32) * 4 + 8) * 2;
            P_memory = ceil(log_p_15 / log(2.0) / 32) * 4 + 8;
            Q_memory = ceil(log_q_15 / log(2.0) / 32) * 4 + 8;
        }
    }
    uint32_t * d_P = nullptr;
    uint32_t * d_Q = nullptr;
    uint32_t * d_work = nullptr;
    auto release_all = [&]() {
        if (d_P != nullptr) {
            pi_memory::release(d_P);
            d_P = nullptr;
        }
        if (d_Q != nullptr) {
            pi_memory::release(d_Q);
            d_Q = nullptr;
        }
        if (d_work != nullptr) {
            pi_memory::release(d_work);
            d_work = nullptr;
        }
    };
    cudaError_t err = pi_memory::allocate(&d_P, P_memory);
    CHECK_AND_RETURN(err, release_all());
    err = pi_memory::allocate(&d_Q, Q_memory);
    CHECK_AND_RETURN(err, release_all());
    if (total_work_memory > 0) {
        err = pi_memory::allocate(&d_work, total_work_memory);
        CHECK_AND_RETURN(err, release_all());
    }
    uint32_t * cur_work_memory = d_work;
    uint32_t * cur_P, * cur_Q;
    valid_digits = 7.5;
    if (valid_digits < target_digits){
        cur_P = cur_work_memory;
        cur_work_memory += 1;
        cur_Q = cur_work_memory;
        cur_work_memory += 1;
    }else{
        cur_P = d_P;
        cur_Q = d_Q;
    }
    uint32_t p_15_init[1] = {4001};
    uint32_t q_15_init[1] = {40};
    err = cudaMemcpy(cur_P, p_15_init, sizeof(p_15_init), cudaMemcpyHostToDevice);
    CHECK_AND_RETURN(err, release_all());
    err = cudaMemcpy(cur_Q, q_15_init, sizeof(q_15_init), cudaMemcpyHostToDevice);
    CHECK_AND_RETURN(err, release_all());
    P = {.data = cur_P, .length = 1, .batch_size = 1, .stride = 1};
    Q = {.data = cur_Q, .length = 1, .batch_size = 1, .stride = 1};
    while (valid_digits < target_digits){
        valid_digits *= 2;
        const uint64_t stage_digits = (uint64_t)valid_digits;
        BatchMPArray next_PQ = {
            .data = cur_work_memory,
            .length = P.length + Q.length,
            .batch_size = 1,
            .stride = P.length + Q.length
        };
        cur_work_memory += P.length + Q.length;
        err = profile_mul(context, P, Q, next_PQ, "P*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, next_PQ, "P*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        
        BatchMPArray next_Q = {
            .data = nullptr,
            .length = next_PQ.length + 1,
            .batch_size = 1,
            .stride = next_PQ.length + 1
        };
        if (valid_digits < target_digits){
            next_Q.data = cur_work_memory;
            cur_work_memory += next_Q.length;
        }else{
            next_Q.data = d_Q;
        }
        err = profile_shift_bits(context, next_PQ, next_Q, 1, "2*P*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, next_Q, "Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());

        BatchMPArray next_PP = {
            .data = cur_work_memory,
            .length = P.length * 2,
            .batch_size = 1,
            .stride = P.length * 2
        };
        cur_work_memory += next_PP.length;
        err = profile_mul(context, P, P, next_PP, "P*P", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, next_PP, "P*P", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());

        BatchMPArray next_P = {
            .data = nullptr,
            .length = next_PP.length + 1,
            .batch_size = 1,
            .stride = next_PP.length + 1
        };
        if (valid_digits < target_digits){
            next_P.data = cur_work_memory;
            cur_work_memory += next_P.length;
        }else{
            next_P.data = d_P;
        }
        err = profile_stage("shift", "2*P*P-1", stage_digits, 1, next_PP.length, 0,
            false, profiler, [&]() {
                pell_double_square<<<min((size_t)65536, ((size_t)next_PP.length + 256) / 256), 256>>>(
                    next_PP.data, next_P.data, next_PP.length);
                return cudaGetLastError();
            });
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, next_P, "P", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());

        P = next_P;
        Q = next_Q;
    }
    pi_memory::release(d_work);
    return cudaSuccess;
}

// Build bounded batches, retaining just the populated limbs while evaluating
// the sibling tree. All precision cuts use the global Chudnovsky term index.
static cudaError_t split_forest(BatchMPContext *ctx, uint32_t offset, uint32_t blocks,
    uint64_t digits, bool need_p, SplitRow &out, StageProfiler *profiler) {
    constexpr uint32_t max_batch = 1u << 22;
    if (blocks <= max_batch && !(blocks & (blocks - 1))) {
        SplitRow batch{};
        cudaError_t err = binary_split_batched(ctx, blocks, batch.q, batch.r,
            profiler, offset, need_p ? &batch.p : nullptr, digits);
        SplitRow view = batch;
        if (err == cudaSuccess) err = trim_split_row(ctx, view, offset * 17, digits);
        auto pack = [&](BatchMPArray src, BatchMPArray &dest) {
            if (!src.data || err != cudaSuccess) return;
            dest = batch_mp_array_create(1, src.length);
            err = dest.data ? cudaMemcpyAsync(dest.data, src.data, (size_t)src.length * sizeof(uint32_t),
                cudaMemcpyDeviceToDevice) : cudaErrorMemoryAllocation;
        };
        pack(view.p, out.p);
        pack(view.q, out.q);
        pack(view.r, out.r);
        for (auto col : {batch.p, batch.q, batch.r}) batch_mp_array_release(col);
        return err;
    }
    const uint32_t left = 1u << (31 - __builtin_clz(blocks - 1));
    const uint32_t right = blocks - left;
    SplitRow l{}, r{};
    BatchMPArray rq{}, pr{};
    auto cleanup = [&]() {
        for (auto col : {l.p, l.q, l.r, r.p, r.q, r.r, rq, pr}) batch_mp_array_release(col);
    };
    cudaError_t err = split_forest(ctx, offset, left, digits, true, l, profiler);
    CHECK_AND_RETURN(err, cleanup());
    err = split_forest(ctx, offset + left, right, digits, need_p, r, profiler);
    CHECK_AND_RETURN(err, cleanup());
    SplitRow lv = l, rv = r;
    err = trim_split_row(ctx, lv, offset * 17, digits);
    CHECK_AND_RETURN(err, cleanup());
    err = trim_split_row(ctx, rv, (offset + left) * 17, digits);
    CHECK_AND_RETURN(err, cleanup());
    out.q = batch_mp_array_create(1, lv.q.length + rv.q.length);
    rq = batch_mp_array_create(1, lv.r.length + rv.q.length);
    if (!out.q.data || !rq.data) { cleanup(); return cudaErrorMemoryAllocation; }
    err = profile_stage("mul", "forest_Q_RQ", blocks, 1, lv.q.length, rv.q.length, true,
        profiler, [&]() { return batch_mp_mul_shared_b(ctx, lv.q, lv.r, rv.q, out.q, rq); });
    CHECK_AND_RETURN(err, cleanup());
    release_array(&l.q); release_array(&l.r); release_array(&r.q);
    pr = batch_mp_array_create(1, lv.p.length + rv.r.length);
    if (!pr.data) { cleanup(); return cudaErrorMemoryAllocation; }
    if (need_p) {
        out.p = batch_mp_array_create(1, lv.p.length + rv.p.length);
        if (!out.p.data) { cleanup(); return cudaErrorMemoryAllocation; }
        err = profile_stage("mul", "forest_P_PR", blocks, 1, lv.p.length, rv.p.length,
            true, profiler, [&]() { return batch_mp_mul_shared_b(ctx, rv.p, rv.r, lv.p, out.p, pr); });
    } else {
        err = profile_mul(ctx, lv.p, rv.r, pr, "forest_PR", blocks, profiler);
    }
    CHECK_AND_RETURN(err, cleanup());
    release_array(&l.p); release_array(&r.p); release_array(&r.r);
    const uint32_t shift = right * 17u * 15u;
    out.r = batch_mp_array_create(1, std::max(rq.length + (shift+31u)/32u, pr.length) + 1);
    if (!out.r.data) { cleanup(); return cudaErrorMemoryAllocation; }
    err = profile_shift_addsub(ctx, rq, pr, out.r, shift, (left & 1u) != 0, blocks, profiler);
    cleanup();
    if (err == cudaSuccess) err = out.q.compact(ctx);
    return err == cudaSuccess ? out.r.compact(ctx) : err;
}

cudaError_t pi_fractional(BatchMPContext * context, uint64_t target_digits, BatchMPArray & P, BatchMPArray & Q, BatchMPArray & P_base, BatchMPArray & Q_base, StageProfiler * profiler = nullptr) {
    /*
    q, r = binary_split(0, N * 17, True)
    r = r >> (15 * (N * 17) - 8)

    p_15, q_15 = sqrt_10005(digits)

    q, r = truncate_limbs_quotient(q, r, prec_limbs)

    p_final = p_15 * q
    q_final = r * q_15

    p_final, q_final = truncate_limbs_quotient(p_final, q_final, prec_limbs)
    */
    // Chudnovsky gains log10(640320^3 / 1728) = 14.181647... digits
    // per term. Round down and allow a guard term's worth of precision.
    // The old 14.3 estimate undercomputed near power-of-two boundaries.
    const double working_digits = (double)target_digits + 16.0;
    int N = 1;
    while (N * 17 * 14.18 < working_digits) {
        N *= 2;
    }
    int prec_limbs = (int)compute_prec_limbs_for_digits(target_digits);
    BatchMPArray Q_bs{}, R_bs{};
    BatchMPArray Q_scaled{};
    BatchMPArray R_shifted{};
    BatchMPArray P_15{}, Q_15{};
    BatchMPArray P_final{};
    BatchMPArray Q_final{};
    BatchMPArray P_truncated_final{};
    BatchMPArray Q_truncated_final{};
    BatchMPArray inverse_workspace{};
    BatchMPArray inv_q_final{};
    BatchMPArray ret_product{};
    BatchMPArray ret_final{};
    auto release_all = [&]() {
        release_array(&Q_bs);
        release_array(&R_bs);
        release_array(&Q_scaled);
        release_array(&R_shifted);
        release_array(&P_15);
        release_array(&Q_15);
        release_array(&inverse_workspace);
        release_array(&inv_q_final);
        release_array(&ret_product);
        if (P_base.data == nullptr) {
            release_array(&P_final);
            release_array(&ret_final);
        }
        if (Q_base.data == nullptr) {
            release_array(&Q_final);
        }
    };

    uint32_t left_blocks = (uint32_t)N / 2, right_blocks = 1;
    while (((double)left_blocks + right_blocks) * 17.0 * 14.18 < working_digits)
        right_blocks *= 2;
    bool forest = left_blocks > 0 && right_blocks < left_blocks && !getenv("GPUPI_SINGLE_TREE");
    if (forest) N = left_blocks + right_blocks;
    cudaError_t err;
    if (forest || N > (1 << 22)) {
        SplitRow result{};
        err = split_forest(context, 0, N, target_digits, false, result, profiler);
        Q_bs = result.q;
        R_bs = result.r;
    } else {
        err = binary_split_batched(context, N, Q_bs, R_bs, profiler, 0, nullptr, target_digits);
    }
    CHECK_AND_RETURN(err, release_all());
    long r_total_shift = 15ull * N * 17 - 8;
    // At small sizes, common-factor cancellation can leave Q shorter than
    // the requested precision. Preserve more low bits of R by scaling Q
    // and R by the same power of two before truncating R's implicit scale.
    // Without this, e.g. 477 requested digits lose accuracy at digit 445.
    const long guard_scale_bits = std::min(r_total_shift,
        32l * std::max(0l, (long)prec_limbs + 1l - (long)Q_bs.length));
    if (guard_scale_bits != 0) {
        Q_scaled = batch_mp_array_create(1u,
            Q_bs.length + (uint32_t)((guard_scale_bits + 31) / 32));
        if (Q_scaled.data == nullptr) {
            CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
        }
        err = profile_shift_bits(context, Q_bs, Q_scaled, (int64_t)guard_scale_bits,
            "Q_guard_scale", target_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        release_array(&Q_bs);
        Q_bs = Q_scaled;
        Q_scaled = {};
        r_total_shift -= guard_scale_bits;
    }
    int r_total_shift_bits = r_total_shift % 32;
    int r_total_shift_limbs = r_total_shift / 32;

    BatchMPArray R_preshift = {
        .data = R_bs.data + r_total_shift_limbs,
        .length = R_bs.length - r_total_shift_limbs,
        .batch_size = 1,
        .stride = R_bs.length - r_total_shift_limbs
    };
    R_shifted = batch_mp_array_create(1, R_preshift.length + 1);
    if (R_shifted.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }
    err = profile_shift_bits(context, R_preshift, R_shifted, -r_total_shift_bits, "R_shift", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());
    batch_mp_array_release(R_bs);
    R_bs = {};
    err = R_shifted.compact(context);
    CHECK_AND_RETURN(err, release_all());

    int extra_limbs = std::max(0l, std::min((long)Q_bs.length, (long)R_shifted.length) - prec_limbs);
    BatchMPArray Q_truncated = {
        .data = Q_bs.data + extra_limbs,
        .length = Q_bs.length - extra_limbs,
        .batch_size = 1,
        .stride = Q_bs.length - extra_limbs
    };
    BatchMPArray R_truncated = {
        .data = R_shifted.data + extra_limbs,
        .length = R_shifted.length - extra_limbs,
        .batch_size = 1,
        .stride = R_shifted.length - extra_limbs
    };

    err = sqrt_10005(context, (uint64_t)working_digits, P_15, Q_15, profiler);
    CHECK_AND_RETURN(err, release_all());

    P_final = batch_mp_array_create(1, P_15.length + Q_truncated.length);
    if (P_final.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }

    err = profile_mul(context, P_15, Q_truncated, P_final, "P15*Q", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());
    release_array(&Q_bs);
    release_array(&P_15);
    Q_final = batch_mp_array_create(1, Q_15.length + R_truncated.length);
    if (!Q_final.data) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }
    err = profile_mul(context, R_truncated, Q_15, Q_final, "R*Q15", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());

    err = profile_compact(context, P_final, "P", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());
    err = profile_compact(context, Q_final, "Q", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());

    release_array(&R_shifted);
    release_array(&Q_15);

    extra_limbs = std::max(0l, std::min((long)P_final.length, (long)Q_final.length) - prec_limbs);
    P_truncated_final = make_subview(P_final, (uint32_t)extra_limbs, P_final.length - (uint32_t)extra_limbs);
    Q_truncated_final = make_subview(Q_final, (uint32_t)extra_limbs, Q_final.length - (uint32_t)extra_limbs);

    uint64_t q_bits_count_u32 = 0u;
    err = compact_and_compute_bit_length(context, Q_truncated_final, &q_bits_count_u32);
    CHECK_AND_RETURN(err, release_all());

    const uint64_t q_bits_count = q_bits_count_u32;
    const uint64_t inverse_bits = q_bits_count * 2u - 1u;

    // recursive_inverse owns allocation of its output.
    err = recursive_inverse(context, Q_truncated_final, inverse_bits, inv_q_final, target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());

    release_array(&Q_final);

    ret_product = batch_mp_array_create(1u, P_truncated_final.length + inv_q_final.length);
    if (ret_product.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }
    err = profile_mul(context, P_truncated_final, inv_q_final, ret_product, "P*invQ", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());
    err = profile_compact(context, ret_product, "P*invQ", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());

    release_array(&P_final);
    release_array(&inverse_workspace);
    release_array(&inv_q_final);

    ret_final = batch_mp_array_create(1u, ret_product.length - (inverse_bits / 32u) + prec_limbs);
    if (ret_final.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }

    const uint64_t result_bits = (uint64_t)prec_limbs * 32u;
    if (inverse_bits >= result_bits) {
        err = shift_right_large(context, ret_product, inverse_bits - result_bits,
            ret_final, "RET_shift", target_digits, profiler);
    } else {
        // Very small requests can need a left shift after normalization.
        err = profile_shift_bits(context, ret_product, ret_final,
            (int64_t)(result_bits - inverse_bits), "RET_shift", target_digits, profiler);
    }
    CHECK_AND_RETURN(err, release_all());
    err = profile_compact(context, ret_final, "RET", target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());
    release_array(&ret_product);

    P_base = ret_final;
    Q_base = {};
    P = ret_final;
    Q = {};

    return cudaSuccess;
}

}  // namespace

int main_bs(int argc, char ** argv){
    int N = 0;
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    if (!parse_int_arg(argv[1], &N)) {
        print_usage(argv[0]);
        return 1;
    }
    if (N < 0 || !is_power_of_two_int(N)) {
        fprintf(stderr, "Invalid range: N=%d\n", N);
        return 1;
    }
    bool benchmark_only = false;
    bool profile_stages = false;
    bool print_result = true;
    if (argc >= 3) {
        if (strcmp(argv[2], "--benchmark") == 0) {
            benchmark_only = true;
            print_result = false;
        }else if (strcmp(argv[2], "--profile-stages") == 0) {
            profile_stages = true;
            print_result = false;
        }else {
            print_usage(argv[0]);
            return 1;
        }
    }

    BatchMPContext * context = batch_mp_init();
    if (context == nullptr) {
        fprintf(stderr, "batch_mp_init failed\n");
        return 1;
    }

    BatchMPArray Q{};
    BatchMPArray R{};
    StageProfiler profiler{};
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (!check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)") ||
        !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)")) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        batch_mp_destroy(context);
        return 1;
    }
    if (!check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)") ||
        !check_cuda(binary_split_batched(context, N, Q, R, profile_stages ? &profiler : nullptr), "binary_split_batched") ||
        !check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }
    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    if (Q.batch_size != 1 || R.batch_size != 1) {
        fprintf(stderr, "Unexpected output batch size: Q=%u R=%u\n", Q.batch_size, R.batch_size);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }
    if (benchmark_only) {
        printf(
            "N=%d Q_len=%u R_len=%u ellapsed_ms=%.3f\n",
            N,
            Q.length,
            R.length,
            elapsed_ms
        );
        if (!print_validation_summary(Q, R)) {
            release_array(&Q);
            release_array(&R);
            batch_mp_destroy(context);
            return 1;
        }
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 0;
    }
    if (profile_stages) {
        float total_leaf_ms = 0.0f;
        float total_mul_ms = 0.0f;
        float total_add_ms = 0.0f;
        float total_sub_ms = 0.0f;
        float total_compact_ms = 0.0f;
        float total_exactdiv_ms = 0.0f;
        float total_ntt_ms = 0.0f;
        for (int idx = 0; idx < profiler.count; ++idx) {
            const StageProfileEntry & entry = profiler.entries[idx];
            if (strcmp(entry.kind, "leaf") == 0) {
                total_leaf_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "mul") == 0) {
                total_mul_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "add") == 0) {
                total_add_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "sub") == 0) {
                total_sub_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "compact") == 0) {
                total_compact_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "exactdiv") == 0) {
                total_exactdiv_ms += entry.elapsed_ms;
            }
            if (strcmp(entry.kind, "mul") == 0 && entry.uses_ntt) {
                total_ntt_ms += entry.elapsed_ms;
            }
            printf(
                "stage[%03d] kind=%8s label=%14s n1=%8" PRIu64 " batch=%8u lhs_len=%9u rhs_len=%9u kernel_ms=%6.3f uses_ntt=%d\n",
                idx,
                entry.kind,
                entry.label,
                entry.n1,
                entry.batch_size,
                entry.lhs_length,
                entry.rhs_length,
                entry.elapsed_ms,
                entry.uses_ntt ? 1 : 0
            );
        }
        printf(
            "leaf_total_ms=%.3f mul_total_ms=%.3f add_total_ms=%.3f sub_total_ms=%.3f compact_total_ms=%.3f exactdiv_total_ms=%.3f ntt_total_ms=%.3f\n",
            total_leaf_ms,
            total_mul_ms,
            total_add_ms,
            total_sub_ms,
            total_compact_ms,
            total_exactdiv_ms,
            total_ntt_ms
        );
        float accounted = total_leaf_ms + total_mul_ms + total_add_ms + total_sub_ms + total_compact_ms + total_exactdiv_ms;
        printf("N=%d Q_len=%u R_len=%u elapsed_ms=%.3f accounted_ms=%.3f workspace_max=%dMB\n", N, Q.length, R.length, elapsed_ms, accounted, (int)(batch_mp_workspace_size(context) / (1024 * 1024)));
        if (!print_validation_summary(Q, R)) {
            release_array(&Q);
            release_array(&R);
            batch_mp_destroy(context);
            return 1;
        }
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 0;
    }

    printf(
        "N=%d Q_len=%u R_len=%u\n",
        N,
        Q.length,
        R.length
    );
    if (print_result){
        printf("Q = ");
        uint32_t * q_host = new uint32_t[Q.length];
        cudaMemcpy(q_host, Q.data, Q.length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int i = Q.length - 1; i >= 0; i--) {
            if (i == Q.length - 1) {
                printf("%x", q_host[i]);
            } else {
                printf("%08x", q_host[i]);
            }
        }
        delete [] q_host;
        printf("\n");
        printf("R = ");
        uint32_t * r_host = new uint32_t[R.length];
        cudaMemcpy(r_host, R.data, R.length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int i = R.length - 1; i >= 0; i--) {
            if (i == R.length - 1) {
                printf("%x", r_host[i]);
            } else {
                printf("%08x", r_host[i]);
            }
        }
        delete [] r_host;
        printf("\n");
    }
    release_array(&Q);
    release_array(&R);
    batch_mp_destroy(context);
    return 0;
}

int main_sqrt_10005(int argc, char ** argv){
    uint64_t target_digits = 0;
    if (argc < 2 || !parse_digits_arg(argv[1], &target_digits)) {
        fprintf(stderr, "Usage: %s <target_digits> [--benchmark|--profile-stages]\n", argv[0]);
        return 1;
    }
    bool benchmark_only = false;
    bool profile_stages = false;
    bool print_result = true;
    if (argc >= 3) {
        if (strcmp(argv[2], "--benchmark") == 0) {
            benchmark_only = true;
            print_result = false;
        } else if (strcmp(argv[2], "--profile-stages") == 0) {
            profile_stages = true;
            print_result = false;
        } else {
            fprintf(stderr, "Usage: %s <target_digits> [--benchmark|--profile-stages]\n", argv[0]);
            return 1;
        }
    }

    BatchMPContext * context = batch_mp_init();
    if (context == nullptr) {
        fprintf(stderr, "batch_mp_init failed\n");
        return 1;
    }

    BatchMPArray P{};
    BatchMPArray Q{};
    StageProfiler profiler{};
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (!check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)") ||
        !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)")) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        batch_mp_destroy(context);
        return 1;
    }

    if (!check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)") ||
        !check_cuda(sqrt_10005(context, target_digits, P, Q, profile_stages ? &profiler : nullptr), "sqrt_10005") ||
        !check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 1;
    }
    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 1;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    if (P.batch_size != 1 || Q.batch_size != 1) {
        fprintf(stderr, "Unexpected output batch size: P=%u Q=%u\n", P.batch_size, Q.batch_size);
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 1;
    }

    if (benchmark_only) {
        printf(
            "target_digits=%" PRIu64 " RET_len=%u Q_len=%u elapsed_ms=%.3f workspace_max=%dMB\n",
            target_digits,
            P.length,
            Q.length,
            elapsed_ms,
            (int)(batch_mp_workspace_size(context) / (1024 * 1024))
        );
        if (!print_sqrt_validation_summary(P, Q)) {
            release_array(&P);
            release_array(&Q);
            batch_mp_destroy(context);
            return 1;
        }
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 0;
    }

    if (profile_stages) {
        float total_mul_ms = 0.0f;
        float total_mul_small_ms = 0.0f;
        float total_shift_ms = 0.0f;
        float total_add_ms = 0.0f;
        float total_compact_ms = 0.0f;
        float total_ntt_ms = 0.0f;
        for (int idx = 0; idx < profiler.count; ++idx) {
            const StageProfileEntry & entry = profiler.entries[idx];
            if (strcmp(entry.kind, "mul") == 0) {
                total_mul_ms += entry.elapsed_ms;
                if (entry.uses_ntt) {
                    total_ntt_ms += entry.elapsed_ms;
                }
            } else if (strcmp(entry.kind, "mul_small") == 0) {
                total_mul_small_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "shift") == 0) {
                total_shift_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "add") == 0) {
                total_add_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "compact") == 0) {
                total_compact_ms += entry.elapsed_ms;
            }
            printf(
                "stage[%03d] kind=%9s label=%18s digits=%8" PRIu64 " batch=%8u lhs_len=%9u rhs_len=%9u kernel_ms=%6.3f uses_ntt=%d\n",
                idx,
                entry.kind,
                entry.label,
                entry.n1,
                entry.batch_size,
                entry.lhs_length,
                entry.rhs_length,
                entry.elapsed_ms,
                entry.uses_ntt ? 1 : 0
            );
        }
        const float accounted_ms = total_mul_ms + total_mul_small_ms + total_shift_ms + total_add_ms + total_compact_ms;
        printf(
            "mul_total_ms=%.3f mul_small_total_ms=%.3f shift_total_ms=%.3f add_total_ms=%.3f compact_total_ms=%.3f ntt_total_ms=%.3f\n",
            total_mul_ms,
            total_mul_small_ms,
            total_shift_ms,
            total_add_ms,
            total_compact_ms,
            total_ntt_ms
        );
        printf(
            "target_digits=%" PRIu64 " RET_len=%u Q_len=%u elapsed_ms=%.3f accounted_ms=%.3f workspace_max=%dMB\n",
            target_digits,
            P.length,
            Q.length,
            elapsed_ms,
            accounted_ms,
            (int)(batch_mp_workspace_size(context) / (1024 * 1024))
        );
        if (!print_sqrt_validation_summary(P, Q)) {
            release_array(&P);
            release_array(&Q);
            batch_mp_destroy(context);
            return 1;
        }
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 0;
    }

    printf(
        "target_digits=%" PRIu64 " P_len=%u Q_len=%u elapsed_ms=%.3f\n",
        target_digits,
        P.length,
        Q.length,
        elapsed_ms
    );
    if (print_result && (!print_full_hex_value("P", P) || !print_full_hex_value("Q", Q))) {
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 1;
    }
    if (!print_sqrt_validation_summary(P, Q)) {
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 1;
    }

    release_array(&P);
    release_array(&Q);
    batch_mp_destroy(context);
    return 0;
}

int main_fractional(int argc, char ** argv){
    uint64_t target_digits = 0;
    if (argc < 2 || !parse_digits_arg(argv[1], &target_digits)) {
        fprintf(stderr, "Usage: %s <target_digits> [--benchmark|--profile-stages|--full-print]\n", argv[0]);
        return 1;
    }
    const uint64_t requested_digits = target_digits;
    const uint64_t convert_digits = round_up_decimal_digits(requested_digits);
    const uint32_t prec_limbs = compute_prec_limbs_for_digits(requested_digits);
    if (requested_digits > 0u && convert_digits == 0u) {
        fprintf(stderr, "Unsupported target_digits=%" PRIu64 "\n", target_digits);
        return 1;
    }
    bool benchmark_only = false;
    bool profile_stages = false;
    bool full_print = false;
    if (argc >= 3) {
        if (strcmp(argv[2], "--benchmark") == 0) {
            benchmark_only = true;
        } else if (strcmp(argv[2], "--profile-stages") == 0) {
            profile_stages = true;
        } else if (strcmp(argv[2], "--full-print") == 0) {
            full_print = true;
        } else {
            fprintf(stderr, "Usage: %s <target_digits> [--benchmark|--profile-stages|--full-print]\n", argv[0]);
            return 1;
        }
    }

    pi_memory::PoolScope memory_pool;
    BatchMPContext * context = batch_mp_init();
    if (context == nullptr) {
        fprintf(stderr, "batch_mp_init failed\n");
        return 1;
    }

    if (target_digits > 1000000000ull) {
        size_t free_bytes, total_bytes;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes <= (40ull << 30))
            batch_mp_limit_large_products(context, 6ull << 30);
    }

    BatchMPArray P{};
    BatchMPArray Q{};
    BatchMPArray P_base{};
    BatchMPArray Q_base{};
    ConvertDecimalPowerTable decimal_powers{};
    char * d_decimal_digits = nullptr;
    StageProfiler profiler{};
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (!check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)") ||
        !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)")) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        batch_mp_destroy(context);
        return 1;
    }

    auto release_decimal = [&]() {
        if (d_decimal_digits != nullptr) {
            pi_memory::release(d_decimal_digits);
            d_decimal_digits = nullptr;
        }
        convert_decimal_release_powers(&decimal_powers);
    };

    const uint32_t decimal_levels = (convert_digits == 0u)
        ? 0u
        : (uint32_t)__builtin_ctzll(convert_digits / kDecimalLeafDigits);

    if (!check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)") ||
        !check_cuda(pi_fractional(context, requested_digits, P, Q, P_base, Q_base, profile_stages ? &profiler : nullptr), "pi_fractional") ||
        !check_cuda(profile_decimal_precompute_powers(
                context,
                &decimal_powers,
                kDecimalLeafDigits,
                decimal_levels,
                profile_stages ? &profiler : nullptr
            ), "convert_decimal_precompute_powers")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_decimal();
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }
    if (requested_digits > 0u) {
        if (!check_cuda(pi_memory::allocate(&d_decimal_digits, convert_digits), "cudaMalloc(decimal_digits)")) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            release_decimal();
            release_array(&P_base);
            release_array(&Q_base);
            batch_mp_destroy(context);
            return 1;
        }
        BatchMPArray fractional_bits = make_subview(P, 0u, std::min<uint32_t>(prec_limbs, P.length));
        if (!check_cuda(profile_decimal_convert(
                context,
                fractional_bits,
                (uint64_t)prec_limbs * 32u,
                convert_digits,
                kDecimalLeafDigits,
                &decimal_powers,
                d_decimal_digits,
                profile_stages ? &profiler : nullptr
            ), "convert_decimal_equal_split")) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            release_decimal();
            release_array(&P_base);
            release_array(&Q_base);
            batch_mp_destroy(context);
            return 1;
        }
    }
    if (!check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_decimal();
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }

    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_decimal();
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    if (P.batch_size != 1) {
        fprintf(stderr, "Unexpected output batch size: RET=%u\n", P.batch_size);
        release_decimal();
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }

    if (benchmark_only) {
        printf(
            "target_digits=%" PRIu64 " convert_digits=%" PRIu64 " RET_len=%u elapsed_ms=%.3f workspace_max=%dMB\n",
            requested_digits,
            convert_digits,
            P.length,
            elapsed_ms,
            (int)(batch_mp_workspace_size(context) / (1024 * 1024))
        );
        if (!print_fractional_preview_summary(P)) {
            release_decimal();
            release_array(&P_base);
            release_array(&Q_base);
            batch_mp_destroy(context);
            return 1;
        }
        if (requested_digits > 0u) {
            const uint32_t prefix_frac = std::min<uint64_t>(requested_digits, 63u);
            const uint32_t suffix_frac = std::min<uint64_t>(requested_digits, 64u);
            char prefix[65] = {};
            char suffix[65] = {};
            prefix[0] = '3';
            if (!check_cuda(cudaMemcpy(prefix + 1, d_decimal_digits, prefix_frac, cudaMemcpyDeviceToHost), "cudaMemcpy(decimal_prefix)") ||
                !check_cuda(cudaMemcpy(suffix, d_decimal_digits + (requested_digits - suffix_frac), suffix_frac, cudaMemcpyDeviceToHost), "cudaMemcpy(decimal_suffix)")) {
                release_decimal();
                release_array(&P_base);
                release_array(&Q_base);
                batch_mp_destroy(context);
                return 1;
            }
            printf("DEC = %s...%.*s\n", prefix, (int)suffix_frac, suffix);
        } else {
            printf("DEC = 3\n");
        }
        release_decimal();
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 0;
    }

    if (profile_stages) {
        float total_leaf_ms = 0.0f;
        float total_mul_ms = 0.0f;
        float total_mul_small_ms = 0.0f;
        float total_shift_ms = 0.0f;
        float total_add_ms = 0.0f;
        float total_sub_ms = 0.0f;
        float total_compact_ms = 0.0f;
        float total_exactdiv_ms = 0.0f;
        float total_decimal_powers_ms = 0.0f;
        float total_decimal_convert_ms = 0.0f;
        float total_ntt_ms = 0.0f;
        for (int idx = 0; idx < profiler.count; ++idx) {
            const StageProfileEntry & entry = profiler.entries[idx];
            if (strcmp(entry.kind, "leaf") == 0) {
                total_leaf_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "mul") == 0) {
                total_mul_ms += entry.elapsed_ms;
                if (entry.uses_ntt) {
                    total_ntt_ms += entry.elapsed_ms;
                }
            } else if (strcmp(entry.kind, "mul_small") == 0) {
                total_mul_small_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "shift") == 0) {
                total_shift_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "add") == 0) {
                total_add_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "sub") == 0) {
                total_sub_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "compact") == 0) {
                total_compact_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "exactdiv") == 0) {
                total_exactdiv_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "decimal_powers") == 0) {
                total_decimal_powers_ms += entry.elapsed_ms;
            } else if (strcmp(entry.kind, "decimal_convert") == 0) {
                total_decimal_convert_ms += entry.elapsed_ms;
            }
            printf(
                "stage[%03d] kind=%9s label=%18s digits=%8" PRIu64 " batch=%8u lhs_len=%9u rhs_len=%9u kernel_ms=%6.3f uses_ntt=%d\n",
                idx,
                entry.kind,
                entry.label,
                entry.n1,
                entry.batch_size,
                entry.lhs_length,
                entry.rhs_length,
                entry.elapsed_ms,
                entry.uses_ntt ? 1 : 0
            );
        }
        printf(
            "leaf_total_ms=%.3f mul_total_ms=%.3f mul_small_total_ms=%.3f shift_total_ms=%.3f add_total_ms=%.3f sub_total_ms=%.3f compact_total_ms=%.3f exactdiv_total_ms=%.3f decimal_powers_total_ms=%.3f decimal_convert_total_ms=%.3f ntt_total_ms=%.3f\n",
            total_leaf_ms,
            total_mul_ms,
            total_mul_small_ms,
            total_shift_ms,
            total_add_ms,
            total_sub_ms,
            total_compact_ms,
            total_exactdiv_ms,
            total_decimal_powers_ms,
            total_decimal_convert_ms,
            total_ntt_ms
        );
        const float accounted_ms =
            total_leaf_ms +
            total_mul_ms +
            total_mul_small_ms +
            total_shift_ms +
            total_add_ms +
            total_sub_ms +
            total_compact_ms +
            total_exactdiv_ms +
            total_decimal_powers_ms +
            total_decimal_convert_ms;
        printf(
            "target_digits=%" PRIu64 " convert_digits=%" PRIu64 " RET_len=%u elapsed_ms=%.3f accounted_ms=%.3f workspace_max=%dMB\n",
            requested_digits,
            convert_digits,
            P.length,
            elapsed_ms,
            accounted_ms,
            (int)(batch_mp_workspace_size(context) / (1024 * 1024))
        );
        if (!print_fractional_preview_summary(P)) {
            release_decimal();
            release_array(&P_base);
            release_array(&Q_base);
            batch_mp_destroy(context);
            return 1;
        }
        if (requested_digits > 0u) {
            const uint32_t copy_digits = std::min<uint64_t>(requested_digits, 4096u);
            std::vector<char> decimal_preview(copy_digits);
            if (!check_cuda(cudaMemcpy(decimal_preview.data(), d_decimal_digits, copy_digits, cudaMemcpyDeviceToHost), "cudaMemcpy(decimal_preview)")) {
                release_decimal();
                release_array(&P_base);
                release_array(&Q_base);
                batch_mp_destroy(context);
                return 1;
            }
            printf("DEC_preview_hash = %u\n", compute_decimal_hash(decimal_preview.data(), copy_digits));
        }
        release_decimal();
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 0;
    }

    fprintf(
        stderr,
        "target_digits=%" PRIu64 " convert_digits=%" PRIu64 " RET_len=%u elapsed_ms=%.3f\n",
        requested_digits,
        convert_digits,
        P.length,
        elapsed_ms
    );
    // No more arithmetic is needed while the output is streamed.
    batch_mp_destroy(context);
    context = nullptr;
    if (full_print) {
        if (!print_full_hex_value("RET", P)) {
            release_decimal();
            release_array(&P_base);
            release_array(&Q_base);
            batch_mp_destroy(context);
            return 1;
        }
        if (requested_digits == 0u) {
            printf("DEC = 3\n");
        } else {
            std::vector<char> decimal_digits(requested_digits);
            if (!check_cuda(cudaMemcpy(
                    decimal_digits.data(),
                    d_decimal_digits,
                    requested_digits,
                    cudaMemcpyDeviceToHost
                ), "cudaMemcpy(decimal_digits)")) {
                release_decimal();
                release_array(&P_base);
                release_array(&Q_base);
                batch_mp_destroy(context);
                return 1;
            }
            printf("DEC = 3");
            fwrite(decimal_digits.data(), 1, requested_digits, stdout);
            printf("\n");
        }
    } else {
        fputc('3', stdout);
        if (requested_digits > 0u) {
            std::vector<char> decimal_digits(requested_digits);
            if (!check_cuda(cudaMemcpy(
                    decimal_digits.data(),
                    d_decimal_digits,
                    requested_digits,
                    cudaMemcpyDeviceToHost
                ), "cudaMemcpy(decimal_digits)")) {
                release_decimal();
                release_array(&P_base);
                release_array(&Q_base);
                batch_mp_destroy(context);
                return 1;
            }
            fwrite(decimal_digits.data(), 1, requested_digits, stdout);
        }
    }
    fputc('\n', stdout);

    release_decimal();
    release_array(&P_base);
    release_array(&Q_base);
    batch_mp_destroy(context);
    return 0;
}

int main(int argc, char ** argv) {
    //return main_bs(argc, argv);
    //return main_sqrt_10005(argc, argv);
    return main_fractional(argc, argv);
}
