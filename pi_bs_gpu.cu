#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "batch_arith.h"

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

// the batched level. We must have j-i is a power of 2.
cudaError_t binary_split_batched(BatchMPContext * context, int i, int j, BatchMPArray &P, BatchMPArray &Q, BatchMPArray &R){
    P = {};
    Q = {};
    R = {};
    const int n = j - i;
    if (context == nullptr || i < 0 || j <= i || (n & (n - 1)) != 0) {
        return cudaErrorInvalidValue;
    }

    P = batch_mp_array_create(n, 3);
    Q = batch_mp_array_create(n, 5);
    R = batch_mp_array_create(n, 6);
    if (P.data == nullptr || Q.data == nullptr || R.data == nullptr) {
        batch_mp_array_release(P);
        batch_mp_array_release(Q);
        batch_mp_array_release(R);
        P = {};
        Q = {};
        R = {};
        return cudaErrorMemoryAllocation;
    }

    const int threads_per_block = 128;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 65535) {
        num_blocks = 65535;
    }
    construct_leaf<<<num_blocks, threads_per_block>>>(P.data, Q.data, R.data, i, j);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        batch_mp_array_release(P);
        batch_mp_array_release(Q);
        batch_mp_array_release(R);
        P = {};
        Q = {};
        R = {};
        return err;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        batch_mp_array_release(P);
        batch_mp_array_release(Q);
        batch_mp_array_release(R);
        P = {};
        Q = {};
        R = {};
        return err;
    }
    if (n == 1) {
        err = P.compact(context);
        if (err == cudaSuccess) {
            err = Q.compact(context);
        }
        if (err == cudaSuccess) {
            err = R.compact(context);
        }
        if (err != cudaSuccess) {
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
        }
        return err;
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
        BatchMPArray P_next = batch_mp_array_create(n1 / 2, P.length * 2);
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
        err = batch_mp_mul(context, P_even, P_odd, P_next);
        if (err != cudaSuccess) {
            batch_mp_array_release(P_next);
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
            return err;
        }
        err = P_next.compact(context);
        if (err != cudaSuccess) {
            batch_mp_array_release(P_next);
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
            return err;
        }
        BatchMPArray Q_next = batch_mp_array_create(n1 / 2, Q.length * 2);
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
        err = batch_mp_mul(context, Q_even, Q_odd, Q_next);
        if (err != cudaSuccess) {
            batch_mp_array_release(P_next);
            batch_mp_array_release(Q_next);
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
            return err;
        }
        err = Q_next.compact(context);
        if (err != cudaSuccess) {
            batch_mp_array_release(P_next);
            batch_mp_array_release(Q_next);
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
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
        BatchMPArray R_prod_1 = batch_mp_array_create(n1 / 2, R.length + Q.length);
        err = batch_mp_mul(context, R_even, Q_odd, R_prod_1);
        if (err != cudaSuccess) {
            batch_mp_array_release(P_next);
            batch_mp_array_release(Q_next);
            batch_mp_array_release(R_prod_1);
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
            return err;
        }
        BatchMPArray R_prod_2 = batch_mp_array_create(n1 / 2, R.length + P.length);
        err = batch_mp_mul(context, P_even, R_odd, R_prod_2);
        if (err != cudaSuccess) {
            batch_mp_array_release(P_next);
            batch_mp_array_release(Q_next);
            batch_mp_array_release(R_prod_1);
            batch_mp_array_release(R_prod_2);
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
            return err;
        }
        BatchMPArray R_next = batch_mp_array_create(
            n1 / 2,
            std::max(R_prod_1.length, R_prod_2.length) + 1
        );
        err = subtract
            ? batch_mp_sub(context, R_prod_1, R_prod_2, R_next)
            : batch_mp_add(context, R_prod_1, R_prod_2, R_next);
        if (err == cudaSuccess) {
            err = R_next.compact(context);
        }
        batch_mp_array_release(R_prod_1);
        batch_mp_array_release(R_prod_2);
        if (err != cudaSuccess) {
            batch_mp_array_release(P_next);
            batch_mp_array_release(Q_next);
            batch_mp_array_release(R_next);
            batch_mp_array_release(P);
            batch_mp_array_release(Q);
            batch_mp_array_release(R);
            P = {};
            Q = {};
            R = {};
            return err;
        }
        batch_mp_array_release(P);
        batch_mp_array_release(Q);
        batch_mp_array_release(R);
        P = P_next;
        Q = Q_next;
        R = R_next;
    }
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
    fprintf(stderr, "Usage: %s <begin> <end>\n", program);
    fprintf(stderr, "Dumps chunk P/Q/R for indices in [begin, end); end-begin must be a power of 2.\n");
}

bool check_cuda(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

bool copy_array_to_host(const BatchMPArray & array, uint32_t ** host_data) {
    *host_data = nullptr;
    if (array.length == 0) {
        return true;
    }
    *host_data = (uint32_t *)malloc((size_t)array.length * sizeof(uint32_t));
    if (*host_data == nullptr) {
        fprintf(stderr, "Host allocation failed\n");
        return false;
    }
    return check_cuda(
        cudaMemcpy(
            *host_data,
            array.data,
            (size_t)array.length * sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy(result)"
    );
}

void print_hex_value(const char * name, const uint32_t * limbs, uint32_t length) {
    printf(" %s_len=%u %s=", name, length, name);
    if (length == 0) {
        printf("0");
        return;
    }
    int top = (int)length - 1;
    while (top > 0 && limbs[top] == 0u) {
        --top;
    }
    printf("%x", limbs[top]);
    for (int idx = top - 1; idx >= 0; --idx) {
        printf("%08x", limbs[idx]);
    }
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
    if (!check_cuda(binary_split_batched(context, begin, end, P, Q, R), "binary_split_batched")) {
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }

    if (P.batch_size != 1 || Q.batch_size != 1 || R.batch_size != 1) {
        fprintf(stderr, "Unexpected output batch size: P=%u Q=%u R=%u\n", P.batch_size, Q.batch_size, R.batch_size);
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }

    uint32_t *host_P = nullptr;
    uint32_t *host_Q = nullptr;
    uint32_t *host_R = nullptr;
    if (!copy_array_to_host(P, &host_P) ||
        !copy_array_to_host(Q, &host_Q) ||
        !copy_array_to_host(R, &host_R)) {
        free(host_P);
        free(host_Q);
        free(host_R);
        release_array(&P);
        release_array(&Q);
        release_array(&R);
        batch_mp_destroy(context);
        return 1;
    }

    printf("begin=%d end=%d", begin, end);
    print_hex_value("P", host_P, P.length);
    print_hex_value("Q", host_Q, Q.length);
    print_hex_value("R", host_R, R.length);
    printf("\n");

    free(host_P);
    free(host_Q);
    free(host_R);
    release_array(&P);
    release_array(&Q);
    release_array(&R);
    batch_mp_destroy(context);
    return 0;
}
