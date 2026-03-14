#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/warp>
#include "batch_mul_naive.h"
#include "batch_mul_addsub_asm.h"

template<int L_MAX>
__global__ void batch_mul_naive_single_kernel(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L_a, int L_b, int stride_A, int stride_B, int stride_ret){
    int ab[L_MAX];
    int r[L_MAX];

    int L = L_a + L_b;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x){
        for (int j = 0; j < L_a; j ++){
            ab[j] = A[idx * stride_A + j];
        }
        for (int j = 0; j < L_b; j ++){
            ab[L_a + j] = B[idx * stride_B + j];
        }
        __syncthreads();
        for (int i = 0; i < L; i ++){
            r[i] = 0;
        }
        for (int i = 0; i < L_a; i += 2){
            uint32_t a0_value, a1_value;
            uint32_t c0_value, c1_value;
            a0_value = ab[i];
            a1_value = (i + 1 < L_a) ? ab[i + 1] : 0;
            c0_value = 0;
            c1_value = 0;

            for (int j = 0; j < L_b; j += 2){
                uint32_t b0_value = ab[L_a + j];
                uint32_t b1_value = (j + 1 < L_b) ? ab[L_a + j + 1] : 0;
                uint32_t r0_value = r[i + j];
                uint32_t r1_value = (i + j + 1 < L) ? r[i + j + 1] : 0;
                uint64_t mul00 = (uint64_t)a0_value * (uint64_t)b0_value + r0_value + c0_value;
                r0_value = (uint32_t)mul00;
                uint32_t mid10 = mul00 >> 32;
                uint64_t mul01 = (uint64_t)a0_value * (uint64_t)b1_value + r1_value + c1_value;
                uint32_t mid11 = (uint32_t)mul01;
                uint32_t mid20 = mul01 >> 32;
                uint64_t mul10 = (uint64_t)a1_value * (uint64_t)b0_value + mid10 + mid11;
                r1_value = (uint32_t)mul10;
                uint32_t mid21 = mul10 >> 32;
                uint64_t mul11 = (uint64_t)a1_value * (uint64_t)b1_value + mid20 + mid21;
                c0_value = (uint32_t)mul11;
                c1_value = mul11 >> 32;
                r[i + j] = r0_value;
                if (i + j + 1 < L){
                    r[i + j + 1] = r1_value;
                }
            }
            int L_b_2 = L_b + (L_b & 1);
            if (i + L_b_2 < L){
                r[i + L_b_2] = c0_value;
            }
            if (i + 1 + L_b_2 < L){
                r[i + 1 + L_b_2] = c1_value;
            }
        }
        __syncthreads();
        for (int j = 0; j < L; j ++){
            ret[idx * stride_ret + j] = r[j];
        }
        __syncthreads();
    }
}

template<int L_MAX, int N_BLOCK>
__global__ void batch_mul_naive_seq_kernel(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L_a, int L_b, int stride_A, int stride_B, int stride_ret){
    __shared__ uint32_t ab[L_MAX][N_BLOCK + 1];
    __shared__ uint32_t r[L_MAX][N_BLOCK + 1];

    int L = L_a + L_b;
    
    for (int idx0 = blockIdx.x * N_BLOCK; idx0 < N; idx0 += gridDim.x * N_BLOCK){
        int batch_len = min(N_BLOCK, N - idx0);
        for (int i = 0; i < batch_len; i ++){
            for (int j = threadIdx.x; j < L_a; j += blockDim.x){
                ab[j][i] = A[(idx0 + i) * stride_A + j];
            }
            for (int j = threadIdx.x; j < L_b; j += blockDim.x){
                ab[L_a + j][i] = B[(idx0 + i) * stride_B + j];
            }
        }
        __syncthreads();
        for (int i = 0; i < L; i ++){
            r[i][threadIdx.x] = 0;
        }
        for (int i = 0; i < L_a; i += 2){
            uint32_t a0_value, a1_value;
            uint32_t c0_value, c1_value;
            a0_value = ab[i][threadIdx.x];
            a1_value = (i + 1 < L_a) ? ab[i + 1][threadIdx.x] : 0;
            c0_value = 0;
            c1_value = 0;

            for (int j = 0; j < L_b; j += 2){
                uint32_t b0_value = ab[L_a + j][threadIdx.x];
                uint32_t b1_value = (j + 1 < L_b) ? ab[L_a + j + 1][threadIdx.x] : 0;
                uint32_t r0_value = r[i + j][threadIdx.x];
                uint32_t r1_value = (i + j + 1 < L) ? r[i + j + 1][threadIdx.x] : 0;
                uint64_t mul00 = (uint64_t)a0_value * (uint64_t)b0_value + r0_value + c0_value;
                r0_value = (uint32_t)mul00;
                uint32_t mid10 = mul00 >> 32;
                uint64_t mul01 = (uint64_t)a0_value * (uint64_t)b1_value + r1_value + c1_value;
                uint32_t mid11 = (uint32_t)mul01;
                uint32_t mid20 = mul01 >> 32;
                uint64_t mul10 = (uint64_t)a1_value * (uint64_t)b0_value + mid10 + mid11;
                r1_value = (uint32_t)mul10;
                uint32_t mid21 = mul10 >> 32;
                uint64_t mul11 = (uint64_t)a1_value * (uint64_t)b1_value + mid20 + mid21;
                c0_value = (uint32_t)mul11;
                c1_value = mul11 >> 32;
                r[i + j][threadIdx.x] = r0_value;
                if (i + j + 1 < L){
                    r[i + j + 1][threadIdx.x] = r1_value;
                }
            }
            int L_b_2 = L_b + (L_b & 1);
            if (i + L_b_2 < L){
                r[i + L_b_2][threadIdx.x] = c0_value;
            }
            if (i + 1 + L_b_2 < L){
                r[i + 1 + L_b_2][threadIdx.x] = c1_value;
            }
        }
        __syncthreads();
        for (int i = 0; i < batch_len; i ++){
            for (int j = threadIdx.x; j < L; j += blockDim.x){
                ret[(idx0 + i) * stride_ret + j] = r[j][i];
            }
        }
        __syncthreads();
    }
}

template<int L_MAX, int BLOCK_SIZE>
__global__ void batch_mul_naive_par_kernel(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L_a, int L_b, int stride_A, int stride_B, int stride_ret){
    __shared__ uint32_t ab[L_MAX];
    __shared__ uint32_t r[L_MAX];

    int L = L_a + L_b;

    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        for (int i = threadIdx.x; i < L_a; i += BLOCK_SIZE){
            ab[i] = A[idx * stride_A + i];
        }
        for (int i = threadIdx.x; i < L_b; i += BLOCK_SIZE){
            ab[L_a + i] = B[idx * stride_B + i];
        }
        for (int i = threadIdx.x; i < L; i += BLOCK_SIZE){
            r[i] = 0;
        }
        __syncthreads();

        for (int i0 = 0; i0 < L_a; i0 += BLOCK_SIZE * 2){
            int i = i0 + threadIdx.x * 2;
            uint32_t a0_value, a1_value;
            uint32_t r0_value, r1_value;
            a0_value = (i < L_a) ? ab[i] : 0;
            a1_value = (i + 1 < L_a) ? ab[i + 1] : 0;
            r0_value = (i < L) ? r[i] : 0;
            r1_value = (i + 1 < L) ? r[i + 1] : 0;

            uint32_t c0_value = 0, c1_value = 0;
            for (int j = 0; j < L_b; j += 2){
                uint32_t b0_value = (j < L_b) ? ab[L_a + j] : 0;
                uint32_t b1_value = (j + 1 < L_b) ? ab[L_a + j + 1] : 0;
                uint64_t mul00 = (uint64_t)a0_value * (uint64_t)b0_value + r0_value + c0_value;
                r0_value = (uint32_t)mul00;
                uint32_t mid10 = mul00 >> 32;
                uint64_t mul01 = (uint64_t)a0_value * (uint64_t)b1_value + r1_value + c1_value;
                uint32_t mid11 = (uint32_t)mul01;
                uint32_t mid20 = mul01 >> 32;
                uint64_t mul10 = (uint64_t)a1_value * (uint64_t)b0_value + mid10 + mid11;
                r1_value = (uint32_t)mul10;
                uint32_t mid21 = mul10 >> 32;
                uint64_t mul11 = (uint64_t)a1_value * (uint64_t)b1_value + mid20 + mid21;
                c0_value = (uint32_t)mul11;
                c1_value = mul11 >> 32;
                if (threadIdx.x == 0){
                    if (i0 + j < L){
                        r[i0 + j] = r0_value;
                    }
                    if (i0 + j + 1 < L){
                        r[i0 + j + 1] = r1_value;
                    }
                }
                r0_value = cuda::device::warp_shuffle_down<BLOCK_SIZE, uint32_t>(r0_value, 1);
                r1_value = cuda::device::warp_shuffle_down<BLOCK_SIZE, uint32_t>(r1_value, 1);
                if (threadIdx.x == BLOCK_SIZE - 1){
                    r0_value = (i0 + BLOCK_SIZE * 2 + j < L) ? r[i0 + BLOCK_SIZE * 2 + j] : 0;
                    r1_value = (i0 + 1 + BLOCK_SIZE * 2 + j < L) ? r[i0 + 1 + BLOCK_SIZE * 2 + j] : 0;
                }
            }
            int L_b_2 = (L_b + 1) & -2;
            r0_value = add_cc(r0_value, c0_value);
            r1_value = addc_cc(r1_value, c1_value);
            int carry_state = addc(0, 0);
            add_cc(1, r0_value);
            addc_cc(0, r1_value);
            carry_state = addc(carry_state, carry_state);
            // carry_state:  0   no carry    2  carry    1  depends on previous
            for (int delta = 1; delta < BLOCK_SIZE; delta *= 2){
                int prev_carry = cuda::device::warp_shuffle_up<BLOCK_SIZE, int>(carry_state, delta);
                if (carry_state == 1){
                    carry_state = (threadIdx.x >= delta) ? prev_carry : 0;
                }
            }
            carry_state = cuda::device::warp_shuffle_up<BLOCK_SIZE, int>(carry_state, 1);
            if (threadIdx.x == 0){
                carry_state = 0;
            }
            r0_value = add_cc(r0_value, carry_state >> 1);
            r1_value = addc_cc(r1_value, 0);
            if (i + L_b_2 < L){
                r[i + L_b_2] = r0_value;
            }
            if (i + 1 + L_b_2 < L){
                r[i + 1 + L_b_2] = r1_value;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < L; i += BLOCK_SIZE){
            ret[idx * stride_ret + i] = r[i];
        }
        __syncthreads();
    }
}

void batch_mul_naive(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L_a, int L_b, int stride_A, int stride_B, int stride_ret){
    if (L_a < L_b){
        std::swap(A, B);
        std::swap(L_a, L_b);
        std::swap(stride_A, stride_B);
    }
    int L = L_a + L_b;
    if (L > 32){
        if (L > 512){
            const int threads_per_block = 32;
            int num_blocks = min(N, 65536);
            batch_mul_naive_par_kernel<1024, threads_per_block><<<num_blocks, threads_per_block>>>(A, B, ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
        }else if (L > 256){
            const int threads_per_block = 32;
            int num_blocks = min(N, 65536);
            batch_mul_naive_par_kernel<512, threads_per_block><<<num_blocks, threads_per_block>>>(A, B, ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
        }else{
            const int threads_per_block = 32;
            int num_blocks = min(N, 65536);
            batch_mul_naive_par_kernel<256, threads_per_block><<<num_blocks, threads_per_block>>>(A, B, ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
        }
    }else if (L > 24){
        const int threads_per_block = 32;
        int num_blocks = min((N + threads_per_block - 1) / threads_per_block, 65536);
        batch_mul_naive_seq_kernel<32, 32><<<num_blocks, threads_per_block>>>(A, B, ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
    }else{
        const int threads_per_block = 32;
        int num_blocks = min((N + threads_per_block - 1) / threads_per_block, 65536);
        batch_mul_naive_single_kernel<24><<<num_blocks, threads_per_block>>>(A, B, ret, N, L_a, L_b, stride_A, stride_B, stride_ret);
    }
}
