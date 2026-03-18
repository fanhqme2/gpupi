#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

cudaError_t sqrt_10005(BatchMPContext * context, int target_digits, BatchMPArray & P, BatchMPArray & Q) {
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
        BatchMPArray next_PQ = {
            .data = cur_work_memory,
            .length = P.length + Q.length,
            .batch_size = 1,
            .stride = P.length + Q.length
        };
        cur_work_memory += P.length + Q.length;
        err = batch_mp_mul(context, P, Q, next_PQ);
        CHECK_AND_RETURN(err, release_all());
        err = next_PQ.compact(context);
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
        err = batch_mp_shift_bits(context, next_PQ, next_Q, 1);
        CHECK_AND_RETURN(err, release_all());
        err = next_Q.compact(context);
        CHECK_AND_RETURN(err, release_all());

        BatchMPArray next_PP = {
            .data = cur_work_memory,
            .length = P.length * 2,
            .batch_size = 1,
            .stride = P.length * 2
        };
        cur_work_memory += next_PP.length;
        err = batch_mp_mul(context, P, P, next_PP);
        CHECK_AND_RETURN(err, release_all());
        err = next_PP.compact(context);
        CHECK_AND_RETURN(err, release_all());

        BatchMPArray next_QQ = {
            .data = cur_work_memory,
            .length = Q.length * 2,
            .batch_size = 1,
            .stride = Q.length * 2
        };
        cur_work_memory += next_QQ.length;
        err = batch_mp_mul(context, Q, Q, next_QQ);
        CHECK_AND_RETURN(err, release_all());
        err = next_QQ.compact(context);
        CHECK_AND_RETURN(err, release_all());

        BatchMPArray next_QQ_10005 = {
            .data = cur_work_memory,
            .length = next_QQ.length + 1,
            .batch_size = 1,
            .stride = next_QQ.length + 1
        };
        cur_work_memory += next_QQ_10005.length;
        err = batch_mp_mul_small(context, next_QQ, 10005, next_QQ_10005);
        CHECK_AND_RETURN(err, release_all());
        err = next_QQ_10005.compact(context);
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
        err = batch_mp_add(context, next_PP, next_QQ_10005, next_P);
        CHECK_AND_RETURN(err, release_all());
        err = next_P.compact(context);
        CHECK_AND_RETURN(err, release_all());

        P = next_P;
        Q = next_Q;
    }
    cudaFree(d_work);
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
        fprintf(stderr, "Usage: %s <target_digits>\n", argv[0]);
        return 1;
    }

    BatchMPContext * context = batch_mp_init();
    if (context == nullptr) {
        fprintf(stderr, "batch_mp_init failed\n");
        return 1;
    }

    BatchMPArray P{};
    BatchMPArray Q{};
    if (!check_cuda(sqrt_10005(context, target_digits, P, Q), "sqrt_10005")) {
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 1;
    }

    if (P.batch_size != 1 || Q.batch_size != 1) {
        fprintf(stderr, "Unexpected output batch size: P=%u Q=%u\n", P.batch_size, Q.batch_size);
        release_array(&P);
        release_array(&Q);
        batch_mp_destroy(context);
        return 1;
    }

    printf(
        "target_digits=%d P_len=%u Q_len=%u\n",
        target_digits,
        P.length,
        Q.length
    );
    if (!print_full_hex_value("P", P) || !print_full_hex_value("Q", Q)) {
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

int main(int argc, char ** argv) {
    return main_bs(argc, argv);
    //return main_sqrt_10005(argc, argv);
}
