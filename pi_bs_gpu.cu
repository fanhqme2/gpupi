#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// leaf case of binary splitting
// 0 <= begin < end <= 71428572
__global__ void construct_leaf(uint32_t * arr_P, uint32_t * arr_Q, uint32_t * arr_R, int begin, int end){
    for (int i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x){
        uint32_t p1 = 6 * i + 1;
        uint32_t p2 = 2 * i + 1;
        uint32_t p3 = 6 * i + 5;
        uint32_t q1 = i + 1;
        uint32_t q2 = i + 1;
        uint32_t q3 = i + 1;
        uint32_t q4 = 640320;
        uint32_t q5 = 40020;
        uint32_t q6 = 426880;
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

        uint32_t Q[5] = {q1, 0, 0, 0, 0};
        uint32_t q_components[5] = {q2, q3, q4, q5, q6};

        for (int k = 0; k < 5; k ++){
            uint32_t carry = 0;
            for (int j = 0; j < 5; j++){
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

        arr_P[(i - begin) * 3 + 0] = P[0];
        arr_P[(i - begin) * 3 + 1] = P[1];
        arr_P[(i - begin) * 3 + 2] = P[2];

        arr_Q[(i - begin) * 5 + 0] = Q[0];
        arr_Q[(i - begin) * 5 + 1] = Q[1];
        arr_Q[(i - begin) * 5 + 2] = Q[2];
        arr_Q[(i - begin) * 5 + 3] = Q[3];
        arr_Q[(i - begin) * 5 + 4] = Q[4];

        arr_R[(i - begin) * 6 + 0] = R[0];
        arr_R[(i - begin) * 6 + 1] = R[1];
        arr_R[(i - begin) * 6 + 2] = R[2];
        arr_R[(i - begin) * 6 + 3] = R[3];
        arr_R[(i - begin) * 6 + 4] = R[4];
        arr_R[(i - begin) * 6 + 5] = R[5];
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
    StageProfileEntry entries[256];
    int count;
};

namespace {

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

cudaError_t profile_addsub(
    BatchMPContext * context,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C,
    bool subtract,
    uint32_t n1,
    StageProfiler * profiler
) {
    return profile_stage(
        subtract ? "sub" : "add",
        subtract ? "R*Q-P*R" : "R*Q+P*R",
        n1,
        A.batch_size,
        A.length,
        B.length,
        false,
        profiler,
        [&]() {
            return subtract
                ? batch_mp_sub(context, A, B, C)
                : batch_mp_add(context, A, B, C);
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

}  // namespace

// the batched level. We must have j-i is a power of 2.
cudaError_t binary_split_batched(BatchMPContext * context, int i, int j, BatchMPArray &P, BatchMPArray &Q, BatchMPArray &R, StageProfiler * profiler = nullptr){

    batch_mp_ensure_workspace(context, 6442450944ull);

    P = {};
    Q = {};
    R = {};
    BatchMPArray P_next{};
    BatchMPArray Q_next{};
    BatchMPArray R_next{};
    BatchMPArray R_prod_1{};
    BatchMPArray R_prod_2{};
    const int n = j - i;
    if (context == nullptr || i < 0 || j <= i || (n & (n - 1)) != 0) {
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
        release_array(R_prod_1);
        release_array(R_prod_2);
    };

    P = batch_mp_array_create(n, 3);
    Q = batch_mp_array_create(n, 5);
    R = batch_mp_array_create(n, 6);
    if (P.data == nullptr || Q.data == nullptr || R.data == nullptr) {
        release_all();
        return cudaErrorMemoryAllocation;
    }

    const int threads_per_block = 128;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 65535) {
        num_blocks = 65535;
    }
    cudaError_t err = profile_stage(
        "leaf",
        "construct_leaf",
        (uint32_t)n,
        (uint32_t)n,
        P.length,
        0,
        false,
        profiler,
        [&]() {
            construct_leaf<<<num_blocks, threads_per_block>>>(P.data, Q.data, R.data, i, j);
            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                return kernel_err;
            }
            return cudaDeviceSynchronize();
        }
    );
    if (err != cudaSuccess) {
        release_all();
        return err;
    }
    if (n == 1) {
        err = profile_compact(context, P, "P", (uint32_t)n, profiler);
        if (err == cudaSuccess) {
            err = profile_compact(context, Q, "Q", (uint32_t)n, profiler);
        }
        if (err == cudaSuccess) {
            err = profile_compact(context, R, "R", (uint32_t)n, profiler);
        }
        if (err != cudaSuccess) {
            release_all();
        }
        return err;
    }

    P_next = batch_mp_array_create(n / 2, P.length * 2);
    Q_next = batch_mp_array_create(n / 2, Q.length * 2);
    R_next = batch_mp_array_create(n / 2, R.length * 2);
    R_prod_1 = batch_mp_array_create(n / 2, R.length + Q.length);
    R_prod_2 = batch_mp_array_create(n / 2, R.length + P.length);
    if (P_next.data == nullptr || Q_next.data == nullptr || R_next.data == nullptr ||
        R_prod_1.data == nullptr || R_prod_2.data == nullptr) {
        release_all();
        return cudaErrorMemoryAllocation;
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
        const bool subtract = (n1 == (uint32_t)n);
        //BatchMPArray P_next = batch_mp_array_create(n1 / 2, P.length * 2);
        P_next.batch_size = n1 / 2;
        P_next.length = P.length * 2;
        P_next.stride = P.length * 2;

        BatchMPArray P_even = {
            .data = P.data,
            .length = P.length,
            .batch_size = n1 / 2,
            .stride = P.stride * 2
        };
        BatchMPArray P_odd = {
            .data = P.data + P.stride,
            .length = P.length,
            .batch_size = n1 / 2,
            .stride = P.stride * 2
        };
        err = profile_mul(context, P_even, P_odd, P_next, "P", n1, profiler);
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
        err = profile_compact(context, P_next, "P", n1 / 2, profiler);
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
        // BatchMPArray Q_next = batch_mp_array_create(n1 / 2, Q.length * 2);
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
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
        err = profile_compact(context, Q_next, "Q", n1 / 2, profiler);
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
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
        //BatchMPArray R_prod_1 = batch_mp_array_create(n1 / 2, R.length + Q.length);
        R_prod_1.batch_size = n1 / 2;
        R_prod_1.length = R.length + Q.length;
        R_prod_1.stride = R.length + Q.length;

        err = profile_mul(context, R_even, Q_odd, R_prod_1, "R*Q", n1, profiler);
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
        // BatchMPArray R_prod_2 = batch_mp_array_create(n1 / 2, R.length + P.length);
        R_prod_2.batch_size = n1 / 2;
        R_prod_2.length = R.length + P.length;
        R_prod_2.stride = R.length + P.length;

        err = profile_mul(context, P_even, R_odd, R_prod_2, "P*R", n1, profiler);
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
        /*BatchMPArray R_next = batch_mp_array_create(
            n1 / 2,
            std::max(R_prod_1.length, R_prod_2.length) + 1
        );*/
        R_next.batch_size = n1 / 2;
        R_next.length = std::max(R_prod_1.length, R_prod_2.length) + 1;
        R_next.stride = std::max(R_prod_1.length, R_prod_2.length) + 1;

        err = profile_addsub(context, R_prod_1, R_prod_2, R_next, subtract, n1, profiler);
        if (err == cudaSuccess) {
            err = profile_compact(context, R_next, "R", n1 / 2, profiler);
        }
        if (err != cudaSuccess) {
            release_all();
            return err;
        }
        // batch_mp_array_release(P);
        // batch_mp_array_release(Q);
        // batch_mp_array_release(R);
        // P = P_next;
        // Q = Q_next;
        // R = R_next;
        std::swap(P, P_next);
        std::swap(Q, Q_next);
        std::swap(R, R_next);
    }
    release_array(P_next);
    release_array(Q_next);
    release_array(R_next);
    release_array(R_prod_1);
    release_array(R_prod_2);
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
    fprintf(stderr, "Usage: %s <begin> <end> [--benchmark|--profile-stages|--profile-mul]\n", program);
    fprintf(stderr, "Dumps chunk P/Q/R for indices in [begin, end); end-begin must be a power of 2.\n");
    fprintf(stderr, "Use --benchmark to print only the binary-splitting GPU time in milliseconds.\n");
    fprintf(stderr, "Use --profile-stages to print timed leaf/add/sub/multiply stages within binary splitting.\n");
    fprintf(stderr, "Use --profile-mul as a backward-compatible alias for --profile-stages.\n");
}

bool check_cuda(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

}  // namespace

int main(int argc, char ** argv){
    int begin = 0;
    int end = 0;
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    if (!parse_int_arg(argv[1], &begin) || !parse_int_arg(argv[2], &end)) {
        print_usage(argv[0]);
        return 1;
    }
    if (begin < 0 || end <= begin) {
        fprintf(stderr, "Invalid range: begin=%d end=%d\n", begin, end);
        return 1;
    }
    bool benchmark_only = false;
    bool profile_stages = false;
    if (argc >= 4) {
        if (strcmp(argv[3], "--benchmark") == 0) {
            benchmark_only = true;
        } else if (strcmp(argv[3], "--profile-stages") == 0) {
            profile_stages = true;
        } else if (strcmp(argv[3], "--profile-mul") == 0) {
            profile_stages = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    const int count = end - begin;
    if (!is_power_of_two_int(count)) {
        fprintf(stderr, "Range length must be a power of 2: begin=%d end=%d\n", begin, end);
        return 1;
    }

    BatchMPContext * context = batch_mp_init();
    if (context == nullptr) {
        fprintf(stderr, "batch_mp_init failed\n");
        return 1;
    }

    BatchMPArray P{};
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
        !check_cuda(binary_split_batched(context, begin, end, P, Q, R, profile_stages ? &profiler : nullptr), "binary_split_batched") ||
        !check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }
    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime")) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    if (P.batch_size != 1 || Q.batch_size != 1 || R.batch_size != 1) {
        fprintf(stderr, "Unexpected output batch size: P=%u Q=%u R=%u\n", P.batch_size, Q.batch_size, R.batch_size);
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }
    if (benchmark_only) {
        printf("begin=%d end=%d elapsed_ms=%.3f\n", begin, end, elapsed_ms);
        release_array(&P);
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
        float total_ntt_ms = 0.0f;
        printf("begin=%d end=%d elapsed_ms=%.3f\n", begin, end, elapsed_ms);
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
            }
            if (strcmp(entry.kind, "mul") == 0 && entry.uses_ntt) {
                total_ntt_ms += entry.elapsed_ms;
            }
            printf(
                "stage[%02d] kind=%s label=%s n1=%u batch=%u lhs_len=%u rhs_len=%u kernel_ms=%.3f uses_ntt=%d\n",
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
            "leaf_total_ms=%.3f mul_total_ms=%.3f add_total_ms=%.3f sub_total_ms=%.3f compact_total_ms=%.3f ntt_total_ms=%.3f\n",
            total_leaf_ms,
            total_mul_ms,
            total_add_ms,
            total_sub_ms,
            total_compact_ms,
            total_ntt_ms
        );
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 0;
    }

    uint32_t p_bits = 0;
    uint32_t q_bits = 0;
    uint32_t r_bits = 0;
    if (!check_cuda(batch_mp_bitlength_max(context, P, &p_bits), "batch_mp_bitlength_max(P)") ||
        !check_cuda(batch_mp_bitlength_max(context, Q, &q_bits), "batch_mp_bitlength_max(Q)") ||
        !check_cuda(batch_mp_bitlength_max(context, R, &r_bits), "batch_mp_bitlength_max(R)")) {
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }

    printf(
        "begin=%d end=%d P_len=%u P_bits=%u Q_len=%u Q_bits=%u R_len=%u R_bits=%u\n",
        begin, end,
        P.length, p_bits,
        Q.length, q_bits,
        R.length, r_bits
    );
    release_array(&P);
    release_array(&Q);
    release_array(&R);
    batch_mp_destroy(context);
    return 0;
}
