#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include <gmp.h>

#include "batch_arith.h"
#include "batch_mul_naive.h"

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
__global__ void construct_leaf_17(uint32_t * arr_P, uint32_t * arr_Q, uint32_t * arr_R, int N){
    for (int i2 = blockIdx.x * blockDim.x + threadIdx.x; i2 < N; i2 += blockDim.x * gridDim.x){
        int i;
        if (is_head){
            i = i2 * 17 + 16;
        }else{
            i = i2 + (i2 >> 4);
        }
        uint32_t p1 = 6 * i + 1;
        uint32_t p2 = 2 * i + 1;
        uint32_t p3 = 6 * i + 5;
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
    uint32_t n1;
    uint32_t batch_size;
    uint32_t lhs_length;
    uint32_t rhs_length;
    float elapsed_ms;
    bool uses_ntt;
};

struct StageProfiler {
    StageProfileEntry entries[512];
    int count;
};

namespace {

constexpr uint32_t kBlockCommonDivisorLo = 830297u;
constexpr uint32_t kBlockCommonDivisorHi = 156590819u;
constexpr uint32_t kValidationMod = 1000000007u;
constexpr uint64_t kLimbBaseMod = (1ull << 32) % kValidationMod;

cudaError_t append_stage_profile(
    StageProfiler * profiler,
    const char * kind,
    const char * label,
    uint32_t n1,
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
    uint32_t n1,
    uint32_t batch_size,
    uint32_t lhs_length,
    uint32_t rhs_length,
    bool uses_ntt,
    StageProfiler * profiler,
    Fn && fn
) {
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
    uint32_t n1,
    StageProfiler * profiler
) {
    return profile_stage(
        "mul",
        label,
        n1,
        A.batch_size,
        A.length,
        B.length,
        ((size_t)A.length + (size_t)B.length) > BATCH_MUL_NAIVE_L_MAX,
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
    uint32_t n1,
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
    uint32_t n1,
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
    uint32_t n1,
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
    uint32_t digits,
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
    uint32_t digits,
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

cudaError_t profile_shift_bits(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    int32_t shift_bits,
    const char * label,
    uint32_t digits,
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

// do binary splitting for N * 17 elements, where N is a power of 2
cudaError_t binary_split_batched(BatchMPContext * context, int N, BatchMPArray &Q, BatchMPArray &R, StageProfiler * profiler = nullptr){
    // pre-allocate workspace for the 2^28 case
    batch_mp_ensure_workspace(context, 6442450944ull);

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
    cudaError_t temp_workspace_err = cudaMalloc(&temp_prod_workspace,(
        N * 3ull + 
        N * 4ull +
        N * 6ull +
        N * 88ull +
        N * 72ull
    ) * 4);
    if (temp_workspace_err != cudaSuccess || temp_prod_workspace == nullptr) {
        return temp_workspace_err != cudaSuccess ? temp_workspace_err : cudaErrorMemoryAllocation;
    }
    uint32_t * const temp_prod_workspace_base = temp_prod_workspace;
    
    if (context == nullptr || (N & (N - 1)) != 0) {
        return cudaErrorInvalidValue;
    }

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
        cudaFree(temp_prod_workspace_base);
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
            construct_leaf_17<true><<<num_blocks, threads_per_block>>>(P0.data, Q0.data, R0.data, N);
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
            construct_leaf_17<false><<<num_blocks, threads_per_block>>>(P.data, Q.data, R.data, N * 16);
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

        int ignore_last_P = n1 <= N;

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

        if (P_odd.batch_size > 0) {
            err = profile_mul(context, P_even, P_odd, P_next, "P", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
            err = profile_compact(context, P_next, "P", n1, profiler);
            CHECK_AND_RETURN(err, release_all());
        }

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
        err = profile_mul(context, Q_even, Q_odd, Q_next, "Q", n1, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, Q_next, "Q", n1, profiler);
        CHECK_AND_RETURN(err, release_all());
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

        err = profile_mul(context, R_even, Q_odd, R_prod_1, "R*Q", n1, profiler);
        CHECK_AND_RETURN(err, release_all());

        int shift_amount = 15 * ((n / n1) + ((n / n1) >> 4));

        R_prod_2.batch_size = n1 / 2;
        R_prod_2.length = R.length + P.length;
        R_prod_2.stride = R.length + P.length;

        P_even.batch_size = n1 / 2;

        err = profile_mul(context, P_even, R_odd, R_prod_2, "P*R", n1, profiler);
        CHECK_AND_RETURN(err, release_all());
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
    }
    release_array(P);
    release_array(P_next);
    release_array(Q_next);
    release_array(R_next);
    cudaFree(temp_prod_workspace_base);
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

uint32_t compute_mod_hash(const uint32_t * limbs, uint32_t length) {
    uint64_t value = 0;
    for (int i = (int)length - 1; i >= 0; --i) {
        value = (value * kLimbBaseMod + limbs[i]) % kValidationMod;
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

bool print_fractional_preview_summary(const BatchMPArray &ret) {
    uint32_t * ret_host = nullptr;
    if (!copy_device_array_to_host(ret, &ret_host)) {
        delete [] ret_host;
        return false;
    }
    print_hex_preview_with_hash("RET", ret_host, ret.length);
    delete [] ret_host;
    return true;
}

uint32_t limb_bit_length(uint32_t limb) {
    return limb == 0 ? 0u : 32u - (uint32_t)__builtin_clz(limb);
}

uint32_t compute_bit_length(const uint32_t * limbs, uint32_t length) {
    while (length > 0 && limbs[length - 1] == 0) {
        --length;
    }
    if (length == 0) {
        return 0;
    }
    return (length - 1) * 32u + limb_bit_length(limbs[length - 1]);
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

cudaError_t compact_and_compute_bit_length(BatchMPContext * context, BatchMPArray & array, uint32_t * bit_length) {
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
    *bit_length = (array.length - 1) * 32u + limb_bit_length(top_limb);
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
        "validation P_bits=%u Q_bits=%u norm_mod_%u=%u norm_low64=0x%016llx expected_norm=1\n",
        compute_bit_length(p_host, P.length),
        compute_bit_length(q_host, Q.length),
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

BatchMPArray make_array_view(uint32_t * data, uint32_t length, uint32_t stride = 0u) {
    return {
        .data = data,
        .length = length,
        .batch_size = 1u,
        .stride = (stride == 0u) ? length : stride
    };
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

struct RecursiveInverseArena {
    BatchMPArray storage;
    uint32_t next_limb;
};

BatchMPArray arena_alloc(RecursiveInverseArena * arena, uint32_t length) {
    if (arena == nullptr || arena->storage.data == nullptr) {
        return {};
    }
    const uint64_t end = (uint64_t)arena->next_limb + length;
    if (end > arena->storage.length) {
        return {};
    }
    BatchMPArray out = make_array_view(arena->storage.data + arena->next_limb, length);
    arena->next_limb += length;
    return out;
}

size_t recursive_inverse_workspace_limbs(uint32_t x_length, uint32_t L, bool is_initial) {
    if (L <= 256u) {
        return 0u;
    }
    const uint32_t m = ((L - 2u) / 4u) & ~31u;
    const uint32_t m_limbs = m / 32u;
    const uint32_t L1 = L - m * 2u;
    const uint32_t x1_length = (x_length > m_limbs) ? (x_length - m_limbs) : 1u;
    const uint32_t x2_length = std::min(x_length, m_limbs);
    const uint32_t y1_length = x1_length + 1u;
    const uint32_t res1_length = x1_length + 1u;
    const uint32_t x2_y1_length = x2_length + y1_length;
    const uint32_t shifted_res1_length = res1_length + m_limbs + 1u;
    const uint32_t res2_length = std::max(shifted_res1_length, x2_y1_length) + 1u;
    const uint32_t y2_prod_length = res2_length + y1_length;
    const uint32_t y2_length = y2_prod_length;
    const uint32_t shifted_y1_length = y1_length + m_limbs + 1u;
    const uint32_t shifted_res2_length = res2_length + m_limbs + 1u;
    const uint32_t y2_x_length = y2_length + x_length;

    size_t total = y1_length + res1_length;
    total += recursive_inverse_workspace_limbs(x1_length, L1, false);
    total += x2_y1_length + shifted_res1_length + res2_length + y2_prod_length + y2_length + shifted_y1_length;
    total += y2_x_length + shifted_res2_length;
    return total;
}

cudaError_t subtract_with_sign(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray & diff,
    bool * a_ge_b
) {
    if (a_ge_b == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = batch_mp_sub(context, A, B, diff);
    if (err != cudaSuccess) {
        return err;
    }
    uint32_t sign_limb = 0u;
    err = copy_top_limb_to_host(diff, &sign_limb);
    if (err != cudaSuccess) {
        return err;
    }
    if (sign_limb != 0xffffffffu) {
        *a_ge_b = true;
        return diff.compact(context);
    }
    err = batch_mp_sub(context, B, A, diff);
    if (err != cudaSuccess) {
        return err;
    }
    *a_ge_b = false;
    return diff.compact(context);
}

cudaError_t recursive_inverse(
    BatchMPContext * context,
    BatchMPArray x,
    uint32_t L,
    bool is_initial,
    BatchMPArray & y_out,
    BatchMPArray * res_out,
    RecursiveInverseArena * arena
) {
    if (L <= 256u) {
        mpz_t x_mpz;
        mpz_t total_mpz;
        mpz_t y_mpz;
        mpz_t res_mpz;
        mpz_inits(x_mpz, total_mpz, y_mpz, res_mpz, NULL);
        cudaError_t err = load_mpz_from_device(x, x_mpz);
        if (err == cudaSuccess) {
            mpz_set_ui(total_mpz, 1u);
            mpz_mul_2exp(total_mpz, total_mpz, (mp_bitcnt_t)L);
            mpz_fdiv_q(y_mpz, total_mpz, x_mpz);
            mpz_mul(res_mpz, y_mpz, x_mpz);
            mpz_sub(res_mpz, total_mpz, res_mpz);
            err = store_mpz_to_device(y_mpz, y_out);
            if (err == cudaSuccess && !is_initial && res_out != nullptr) {
                err = store_mpz_to_device(res_mpz, *res_out);
            }
        }
        mpz_clears(x_mpz, total_mpz, y_mpz, res_mpz, NULL);
        return err;
    }

    const uint32_t m = ((L - 2u) / 4u) & ~31u;
    const uint32_t m_limbs = m / 32u;
    const uint32_t L1 = L - m * 2u;
    BatchMPArray x1 = make_subview(x, m_limbs, x.length > m_limbs ? (x.length - m_limbs) : 0u);
    BatchMPArray x2 = make_subview(x, 0u, std::min(x.length, m_limbs));

    BatchMPArray y1 = arena_alloc(arena, (x1.length == 0 ? 1u : x1.length) + 1u);
    if (y1.data == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    BatchMPArray res1 = arena_alloc(arena, (x1.length == 0 ? 1u : x1.length) + 1u);
    if (res1.data == nullptr) {
        return cudaErrorMemoryAllocation;
    }

    cudaError_t err = recursive_inverse(context, x1, L1, false, y1, &res1, arena);
    if (err != cudaSuccess) {
        return err;
    }
    err = y1.compact(context);
    if (err != cudaSuccess) {
        return err;
    }
    err = res1.compact(context);
    if (err != cudaSuccess) {
        return err;
    }

    BatchMPArray x2_y1 = arena_alloc(arena, x2.length + y1.length);
    BatchMPArray shifted_res1 = arena_alloc(arena, res1.length + m_limbs + 1u);
    BatchMPArray res2 = arena_alloc(arena, std::max(shifted_res1.length, x2_y1.length) + 1u);
    if (x2_y1.data == nullptr || shifted_res1.data == nullptr || res2.data == nullptr) {
        return cudaErrorMemoryAllocation;
    }

    err = batch_mp_mul(context, x2, y1, x2_y1);
    if (err != cudaSuccess) {
        return err;
    }
    err = x2_y1.compact(context);
    if (err != cudaSuccess) {
        return err;
    }

    err = batch_mp_shift_bits(context, res1, shifted_res1, (int32_t)m);
    if (err != cudaSuccess) {
        return err;
    }
    err = shifted_res1.compact(context);
    if (err != cudaSuccess) {
        return err;
    }

    bool positive_branch = false;
    err = subtract_with_sign(context, shifted_res1, x2_y1, res2, &positive_branch);
    if (err != cudaSuccess) {
        return err;
    }

    BatchMPArray y2_prod = arena_alloc(arena, res2.length + y1.length);
    if (y2_prod.data == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    err = batch_mp_mul(context, res2, y1, y2_prod);
    if (err != cudaSuccess) {
        return err;
    }
    err = y2_prod.compact(context);
    if (err != cudaSuccess) {
        return err;
    }

    BatchMPArray y2 = arena_alloc(arena, std::max<uint32_t>(2u, y2_prod.length));
    if (y2.data == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    err = batch_mp_shift_bits(context, y2_prod, y2, -(int32_t)L1);
    if (err != cudaSuccess) {
        return err;
    }
    err = y2.compact(context);
    if (err != cudaSuccess) {
        return err;
    }

    if (!positive_branch) {
        err = batch_mp_add_small(context, y2, 1u, y2);
        if (err != cudaSuccess) {
            return err;
        }
        err = y2.compact(context);
        if (err != cudaSuccess) {
            return err;
        }
    }

    BatchMPArray shifted_y1 = arena_alloc(arena, y1.length + m_limbs + 1u);
    if (shifted_y1.data == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    err = batch_mp_shift_bits(context, y1, shifted_y1, (int32_t)m);
    if (err != cudaSuccess) {
        return err;
    }
    err = shifted_y1.compact(context);
    if (err != cudaSuccess) {
        return err;
    }

    if (positive_branch) {
        err = batch_mp_add(context, shifted_y1, y2, y_out);
    } else {
        err = batch_mp_sub(context, shifted_y1, y2, y_out);
    }
    if (err != cudaSuccess) {
        return err;
    }
    err = y_out.compact(context);
    if (err != cudaSuccess) {
        return err;
    }

    if (!is_initial && res_out != nullptr) {
        BatchMPArray y2_x = arena_alloc(arena, y2.length + x.length);
        BatchMPArray shifted_res2 = arena_alloc(arena, res2.length + m_limbs + 1u);
        if (y2_x.data == nullptr || shifted_res2.data == nullptr) {
            return cudaErrorMemoryAllocation;
        }
        err = batch_mp_mul(context, y2, x, y2_x);
        if (err != cudaSuccess) {
            return err;
        }
        err = y2_x.compact(context);
        if (err != cudaSuccess) {
            return err;
        }
        err = batch_mp_shift_bits(context, res2, shifted_res2, (int32_t)m);
        if (err != cudaSuccess) {
            return err;
        }
        err = shifted_res2.compact(context);
        if (err != cudaSuccess) {
            return err;
        }
        if (positive_branch) {
            err = batch_mp_sub(context, shifted_res2, y2_x, *res_out);
        } else {
            err = batch_mp_sub(context, y2_x, shifted_res2, *res_out);
        }
        if (err != cudaSuccess) {
            return err;
        }
        return res_out->compact(context);
    }

    return cudaSuccess;
}

cudaError_t sqrt_10005(BatchMPContext * context, int target_digits, BatchMPArray & P, BatchMPArray & Q, StageProfiler * profiler = nullptr) {
    batch_mp_ensure_workspace(context, 6442450944ull >> 2);
    /*
    p_15 = mpz(4001)
    q_15 = mpz(40)
    valid_terms = 7.5

    while valid_terms < target_digits:
        valid_terms *= 2
        p_15, q_15 = (
            p_15 * p_15 + 10005 * q_15 * q_15,
            p_15 * q_15 * 2,
        )
    */
    double log_p_15 = log(4001.0);
    double log_q_15 = log(40.0);
    double valid_digits = 7.5;
    uint32_t total_work_memory = 0;
    uint32_t P_memory = 0, Q_memory = 0;
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
            total_work_memory += (ceil(log_p_15 / log(2.0) / 32) * 4 + 8) * 4;
            total_work_memory += (ceil(log_q_15 / log(2.0) / 32) * 4 + 8) * 2;
        }else{
            total_work_memory += (ceil(log_p_15 / log(2.0) / 32) * 4 + 8) * 3;
            total_work_memory += (ceil(log_q_15 / log(2.0) / 32) * 4 + 8) * 1;
            P_memory = ceil(log_p_15 / log(2.0) / 32) * 4 + 8;
            Q_memory = ceil(log_q_15 / log(2.0) / 32) * 4 + 8;
        }
    }
    uint32_t * d_P = nullptr;
    uint32_t * d_Q = nullptr;
    uint32_t * d_work = nullptr;
    auto release_all = [&]() {
        if (d_P != nullptr) {
            cudaFree(d_P);
            d_P = nullptr;
        }
        if (d_Q != nullptr) {
            cudaFree(d_Q);
            d_Q = nullptr;
        }
        if (d_work != nullptr) {
            cudaFree(d_work);
            d_work = nullptr;
        }
    };
    cudaError_t err = cudaMalloc(&d_P, P_memory);
    CHECK_AND_RETURN(err, release_all());
    err = cudaMalloc(&d_Q, Q_memory);
    CHECK_AND_RETURN(err, release_all());
    if (total_work_memory > 0) {
        err = cudaMalloc(&d_work, total_work_memory);
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
        const uint32_t stage_digits = (uint32_t)valid_digits;
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

        BatchMPArray next_QQ = {
            .data = cur_work_memory,
            .length = Q.length * 2,
            .batch_size = 1,
            .stride = Q.length * 2
        };
        cur_work_memory += next_QQ.length;
        err = profile_mul(context, Q, Q, next_QQ, "Q*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, next_QQ, "Q*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());

        BatchMPArray next_QQ_10005 = {
            .data = cur_work_memory,
            .length = next_QQ.length + 1,
            .batch_size = 1,
            .stride = next_QQ.length + 1
        };
        cur_work_memory += next_QQ_10005.length;
        err = profile_mul_small(context, next_QQ, 10005, next_QQ_10005, "10005*Q*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, next_QQ_10005, "10005*Q*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        BatchMPArray next_P = {
            .data = nullptr,
            .length = max(next_PP.length, next_QQ_10005.length) + 1,
            .batch_size = 1,
            .stride = max(next_PP.length, next_QQ_10005.length) + 1
        };
        if (valid_digits < target_digits){
            next_P.data = cur_work_memory;
            cur_work_memory += next_P.length;
        }else{
            next_P.data = d_P;
        }
        err = profile_add(context, next_PP, next_QQ_10005, next_P, "P*P+10005*Q*Q", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());
        err = profile_compact(context, next_P, "P", stage_digits, profiler);
        CHECK_AND_RETURN(err, release_all());

        P = next_P;
        Q = next_Q;
    }
    cudaFree(d_work);
    return cudaSuccess;
}

cudaError_t pi_fractional(BatchMPContext * context, int target_digits, BatchMPArray & P, BatchMPArray & Q, BatchMPArray & P_base, BatchMPArray & Q_base, StageProfiler * profiler = nullptr) {
    /*
    q, r = binary_split(0, N * 17, True)
    r = r >> (15 * (N * 17) - 8)

    p_15, q_15 = sqrt_10005(digits)

    q, r = truncate_limbs_quotient(q, r, prec_limbs)

    p_final = p_15 * q
    q_final = r * q_15

    p_final, q_final = truncate_limbs_quotient(p_final, q_final, prec_limbs)
    */
    int N = 1;
    while (N * 17 * 14.3 < target_digits) {
        N *= 2;
    }
    int prec_limbs = ((long)(target_digits * 3.322) + 31) / 32 + 1;
    BatchMPArray Q_bs{}, R_bs{};
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

    cudaError_t err = binary_split_batched(context, N, Q_bs, R_bs, profiler);
    CHECK_AND_RETURN(err, release_all());
    long r_total_shift = 15ull * N * 17 - 8;
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
    err = batch_mp_shift_bits(context, R_preshift, R_shifted, -r_total_shift_bits);
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

    err = sqrt_10005(context, target_digits, P_15, Q_15, profiler);
    CHECK_AND_RETURN(err, release_all());

    P_final = batch_mp_array_create(1, P_15.length + Q_truncated.length);
    Q_final = batch_mp_array_create(1, Q_15.length + R_truncated.length);
    if (P_final.data == nullptr || Q_final.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }

    err = profile_mul(context, P_15, Q_truncated, P_final, "P15*Q", (uint32_t)target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());
    err = profile_mul(context, R_truncated, Q_15, Q_final, "R*Q15", (uint32_t)target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());

    err = profile_compact(context, P_final, "P", (uint32_t)target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());
    err = profile_compact(context, Q_final, "Q", (uint32_t)target_digits, profiler);
    CHECK_AND_RETURN(err, release_all());

    extra_limbs = std::max(0l, std::min((long)P_final.length, (long)Q_final.length) - prec_limbs);
    P_truncated_final = make_subview(P_final, (uint32_t)extra_limbs, P_final.length - (uint32_t)extra_limbs);
    Q_truncated_final = make_subview(Q_final, (uint32_t)extra_limbs, Q_final.length - (uint32_t)extra_limbs);

    uint32_t q_bits_count = 0u;
    err = compact_and_compute_bit_length(context, Q_truncated_final, &q_bits_count);
    CHECK_AND_RETURN(err, release_all());

    const uint32_t inverse_bits = q_bits_count * 2u - 1u;
    inverse_workspace = batch_mp_array_create(1u, (uint32_t)recursive_inverse_workspace_limbs(Q_truncated_final.length, inverse_bits, true));
    inv_q_final = batch_mp_array_create(1u, Q_truncated_final.length + 1u);
    if ((inverse_workspace.length != 0u && inverse_workspace.data == nullptr) || inv_q_final.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }

    RecursiveInverseArena arena{
        .storage = inverse_workspace,
        .next_limb = 0u
    };
    err = recursive_inverse(context, Q_truncated_final, inverse_bits, true, inv_q_final, nullptr, &arena);
    CHECK_AND_RETURN(err, release_all());
    err = inv_q_final.compact(context);
    CHECK_AND_RETURN(err, release_all());

    ret_product = batch_mp_array_create(1u, P_truncated_final.length + inv_q_final.length);
    ret_final = batch_mp_array_create(1u, P_truncated_final.length + inv_q_final.length);
    if (ret_product.data == nullptr || ret_final.data == nullptr) {
        CHECK_AND_RETURN(cudaErrorMemoryAllocation, release_all());
    }
    err = batch_mp_mul(context, P_truncated_final, inv_q_final, ret_product);
    CHECK_AND_RETURN(err, release_all());
    err = ret_product.compact(context);
    CHECK_AND_RETURN(err, release_all());

    const int32_t final_shift = -((int32_t)inverse_bits - prec_limbs * 32);
    err = batch_mp_shift_bits(context, ret_product, ret_final, final_shift);
    CHECK_AND_RETURN(err, release_all());
    err = ret_final.compact(context);
    CHECK_AND_RETURN(err, release_all());

    release_array(&Q_bs);
    release_array(&R_shifted);
    release_array(&P_15);
    release_array(&Q_15);
    release_array(&inverse_workspace);
    release_array(&inv_q_final);
    release_array(&ret_product);
    release_array(&P_final);

    P_base = ret_final;
    Q_base = Q_final;
    P = ret_final;
    Q = Q_truncated_final;

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
                "stage[%03d] kind=%8s label=%14s n1=%8u batch=%8u lhs_len=%9u rhs_len=%9u kernel_ms=%6.3f uses_ntt=%d\n",
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
    int target_digits = 0;
    if (argc < 2 || !parse_int_arg(argv[1], &target_digits) || target_digits < 0) {
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
            "target_digits=%d RET_len=%u Q_len=%u elapsed_ms=%.3f workspace_max=%dMB\n",
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
                "stage[%03d] kind=%9s label=%18s digits=%8u batch=%8u lhs_len=%9u rhs_len=%9u kernel_ms=%6.3f uses_ntt=%d\n",
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
            "target_digits=%d RET_len=%u Q_len=%u elapsed_ms=%.3f accounted_ms=%.3f workspace_max=%dMB\n",
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
        "target_digits=%d P_len=%u Q_len=%u elapsed_ms=%.3f\n",
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
    int target_digits = 0;
    if (argc < 2 || !parse_int_arg(argv[1], &target_digits) || target_digits < 0) {
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
    BatchMPArray P_base{};
    BatchMPArray Q_base{};
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
        !check_cuda(pi_fractional(context, target_digits, P, Q, P_base, Q_base, profile_stages ? &profiler : nullptr), "pi_fractional") ||
        !check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }

    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    if (P.batch_size != 1 || Q.batch_size != 1) {
        fprintf(stderr, "Unexpected output batch size: P=%u Q=%u\n", P.batch_size, Q.batch_size);
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }

    if (benchmark_only) {
        printf(
            "target_digits=%d RET_len=%u elapsed_ms=%.3f workspace_max=%dMB\n",
            target_digits,
            P.length,
            elapsed_ms,
            (int)(batch_mp_workspace_size(context) / (1024 * 1024))
        );
        if (!print_fractional_preview_summary(P)) {
            release_array(&P_base);
            release_array(&Q_base);
            batch_mp_destroy(context);
            return 1;
        }
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
            }
            printf(
                "stage[%03d] kind=%9s label=%18s digits=%8u batch=%8u lhs_len=%9u rhs_len=%9u kernel_ms=%6.3f uses_ntt=%d\n",
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
            "leaf_total_ms=%.3f mul_total_ms=%.3f mul_small_total_ms=%.3f shift_total_ms=%.3f add_total_ms=%.3f sub_total_ms=%.3f compact_total_ms=%.3f exactdiv_total_ms=%.3f ntt_total_ms=%.3f\n",
            total_leaf_ms,
            total_mul_ms,
            total_mul_small_ms,
            total_shift_ms,
            total_add_ms,
            total_sub_ms,
            total_compact_ms,
            total_exactdiv_ms,
            total_ntt_ms
        );
        const float accounted_ms = total_leaf_ms + total_mul_ms + total_mul_small_ms + total_shift_ms + total_add_ms + total_sub_ms + total_compact_ms + total_exactdiv_ms;
        printf(
            "target_digits=%d RET_len=%u elapsed_ms=%.3f accounted_ms=%.3f workspace_max=%dMB\n",
            target_digits,
            P.length,
            elapsed_ms,
            accounted_ms,
            (int)(batch_mp_workspace_size(context) / (1024 * 1024))
        );
        if (!print_fractional_preview_summary(P)) {
            release_array(&P_base);
            release_array(&Q_base);
            batch_mp_destroy(context);
            return 1;
        }
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 0;
    }

    printf(
        "target_digits=%d RET_len=%u elapsed_ms=%.3f\n",
        target_digits,
        P.length,
        elapsed_ms
    );
    if (print_result && !print_full_hex_value("RET", P)) {
        release_array(&P_base);
        release_array(&Q_base);
        batch_mp_destroy(context);
        return 1;
    }

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
