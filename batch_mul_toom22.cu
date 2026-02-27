#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include "batch_mul_toom22.h"
#include "batch_mul_direct.h"
#include "batch_mul_8x8_block.h"
#include "batch_mul_addsub_warp.h"

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Transform kernel: decomposes inputs and prepares 3N interleaved arrays
// L_split = ceil(L/2) is the actual split point
// L_half = L_split + 1 is the buffer size for padded storage
template<int BLOCK_SIZE>
__global__ void batch_mul_toom22_transform_kernel(
    const uint32_t * A, const uint32_t * B,
    uint32_t * A_out, uint32_t * B_out,
    int N, int L, int L_split, int L_half
) {
    const uint32_t * src = (blockIdx.y == 0) ? A : B;
    uint32_t * dst = (blockIdx.y == 0) ? A_out : B_out;
    int j_idx = (threadIdx.y * BLOCK_SIZE + threadIdx.x) * 2;
    __shared__ uint32_t carry_prop[BATCH_MUL_TOOM22_L_MAX / 4 / BLOCK_SIZE + 1];
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        uint32_t r0_value, r1_value;
        uint32_t c0_value, c1_value;
        r0_value = (j_idx < L_split) ? src[idx * L + j_idx] : 0;
        r1_value = (j_idx + 1 < L_split) ? src[idx * L + j_idx + 1] : 0;
        c0_value = (j_idx + L_split < L) ? src[idx * L + j_idx + L_split] : 0;
        c1_value = (j_idx + L_split + 1 < L) ? src[idx * L + j_idx + L_split + 1] : 0;
        if (j_idx < L_half){
            dst[idx * L_half + j_idx] = c0_value;
            dst[(N + idx) * L_half + j_idx] = r0_value;
        }
        if (j_idx + 1 < L_half){
            dst[idx * L_half + j_idx + 1] = c1_value;
            dst[(N + idx) * L_half + j_idx + 1] = r1_value;
        }
        batch_mul_add_64_all_warp<BLOCK_SIZE>(r0_value, r1_value, c0_value, c1_value, carry_prop);
        if (j_idx < L_half){
            dst[(N * 2 + idx) * L_half + j_idx] = r0_value;
        }
        if (j_idx + 1 < L_half){
            dst[(N * 2 + idx) * L_half + j_idx + 1] = r1_value;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
__global__ void batch_mul_toom22_reconstruct_kernel(
    uint32_t * ret,
    const uint32_t * C,
    int N, int L_total, int L_split, int L_half
) {
    __shared__ uint32_t carry_prop[BATCH_MUL_TOOM22_L_MAX / BLOCK_SIZE + 1];
    int participating_warps = (L_half * 2 + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        int j_idx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

        uint32_t r0_value, r1_value;
        uint32_t c0_value, c1_value;
        uint32_t t0_value, t1_value;
        uint32_t s0_value, s1_value;
        if (j_idx * 2 < L_half * 2){
            r0_value = C[(N * 2 + idx) * L_half * 2 + j_idx * 2];
            r1_value = C[(N * 2 + idx) * L_half * 2 + j_idx * 2 + 1];
            c0_value = C[(N + idx) * L_half * 2 + j_idx * 2];
            c1_value = C[(N + idx) * L_half * 2 + j_idx * 2 + 1];
            t0_value = C[idx * L_half * 2 + j_idx * 2];
            t1_value = C[idx * L_half * 2 + j_idx * 2 + 1];
        }else{
            r0_value = 0;
            r1_value = 0;
            c0_value = 0;
            c1_value = 0;
            t0_value = 0;
            t1_value = 0;
        }
        if (j_idx * 2 < L_split){
            s0_value = C[(N + idx) * L_half * 2 + j_idx * 2 + L_split];
        }else if (j_idx * 2 - L_split < L_half * 2){
            s0_value = C[idx * L_half * 2 + j_idx * 2 - L_split];
        }else{
            s0_value = 0;
        }
        if (j_idx * 2 + 1 < L_split){
            s1_value = C[(N + idx) * L_half * 2 + j_idx * 2 + 1 + L_split];
        }else if (j_idx * 2 + 1 - L_split < L_half * 2){
            s1_value = C[idx * L_half * 2 + j_idx * 2 + 1 - L_split];
        }else{
            s1_value = 0;
        }

        if (j_idx * 2 < L_split * 2){
            ret[idx * L_total * 2 + j_idx * 2] = c0_value;
            ret[idx * L_total * 2 + j_idx * 2 + 1] = c1_value;
        }
                
        batch_mul_sub3_64_grouped_warp<BLOCK_SIZE>(
            r0_value, r1_value, c0_value, c1_value, t0_value, t1_value,
            threadIdx.y, participating_warps, threadIdx.y < participating_warps,
            reinterpret_cast<ushort2*>(carry_prop)
        );

        batch_mul_add_64_all_warp<BLOCK_SIZE>(r0_value, r1_value, s0_value, s1_value, carry_prop);

        if (j_idx * 2 + L_split < L_total * 2){
            ret[idx * L_total * 2 + j_idx * 2 + L_split] = r0_value;
        }
        if (j_idx * 2 + 1 + L_split < L_total * 2){
            ret[idx * L_total * 2 + j_idx * 2 + 1 + L_split] = r1_value;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
__global__ void batch_mul_toom22_directlv1_kernel(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L_total){
    int L_split = (L_total + 1) >> 1;
    int L = L_split + 1;
    __shared__ uint32_t a[3][BLOCK_SIZE * 2];
    __shared__ uint32_t b[3][BLOCK_SIZE * 2];
    __shared__ uint32_t r[3][BLOCK_SIZE * 4];
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    for (int i0 = 0; i0 < L; i0 += BLOCK_SIZE * 2){
        a[threadIdx.y][i0 + threadIdx.x * 2] = 0;
        a[threadIdx.y][i0 + threadIdx.x * 2 + 1] = 0;
        b[threadIdx.y][i0 + threadIdx.x * 2] = 0;
        b[threadIdx.y][i0 + threadIdx.x * 2 + 1] = 0;
    }
    __syncthreads();
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        uint32_t * src_a, * src_b;
        int src_l;
        if (threadIdx.y == 0){
            src_a = A + L_split;
            src_b = B + L_split;
            src_l = L_total - L_split;
        }else{
            src_a = A;
            src_b = B;
            src_l = L_split;
        }
        if (threadIdx.y < 2){
            for (int j = threadIdx.x; j < L; j += BLOCK_SIZE){
                a[threadIdx.y][j] = (j < src_l) ? src_a[idx * L_total + j] : 0;
                b[threadIdx.y][j] = (j < src_l) ? src_b[idx * L_total + j] : 0;
            }
        }
        __syncthreads();
        if (threadIdx.y < 2){
            uint32_t r0_value, r1_value;
            uint32_t c0_value, c1_value;
            if (threadIdx.y == 0){
                r0_value = a[0][threadIdx.x * 2 + 0];
                r1_value = a[0][threadIdx.x * 2 + 1];
                c0_value = a[1][threadIdx.x * 2 + 0];
                c1_value = a[1][threadIdx.x * 2 + 1];
            }else{
                r0_value = b[0][threadIdx.x * 2 + 0];
                r1_value = b[0][threadIdx.x * 2 + 1];
                c0_value = b[1][threadIdx.x * 2 + 0];
                c1_value = b[1][threadIdx.x * 2 + 1];
            }

            batch_mul_add_64_single_warp<BLOCK_SIZE>(r0_value, r1_value, c0_value, c1_value);
            if (threadIdx.y == 0){
                if (threadIdx.x * 2 + 0 < L){
                    a[2][threadIdx.x * 2 + 0] = r0_value;
                }
                if (threadIdx.x * 2 + 1 < L){
                    a[2][threadIdx.x * 2 + 1] = r1_value;
                }
            }else{
                if (threadIdx.x * 2 + 0 < L){
                    b[2][threadIdx.x * 2 + 0] = r0_value;
                }
                if (threadIdx.x * 2 + 1 < L){
                    b[2][threadIdx.x * 2 + 1] = r1_value;
                }
            }
        }
        __syncthreads();

        if (true){
            uint32_t a0_value, a1_value;
            uint32_t r0_value, r1_value;
            a0_value = a[threadIdx.y][threadIdx.x * 2];
            a1_value = a[threadIdx.y][threadIdx.x * 2 + 1];
            r0_value = 0;
            r1_value = 0;
            uint32_t c0_value = 0, c1_value = 0;
            for (int j = 0; j < L; j += 2){
                uint32_t b0_value = b[threadIdx.y][j];
                uint32_t b1_value = b[threadIdx.y][j + 1];

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
                    r[threadIdx.y][j] = r0_value;
                    r[threadIdx.y][j + 1] = r1_value;
                }
                r0_value = __shfl_down_sync(warp_mask, r0_value, 1, BLOCK_SIZE);
                r1_value = __shfl_down_sync(warp_mask, r1_value, 1, BLOCK_SIZE);
                if (threadIdx.x == BLOCK_SIZE - 1){
                    r0_value = 0;
                    r1_value = 0;
                }
            }
            int L2 = (L + 1) & -2;

            batch_mul_add_64_single_warp<BLOCK_SIZE>(r0_value, r1_value, c0_value, c1_value);

            r[threadIdx.y][threadIdx.x * 2 + L2] = r0_value;
            r[threadIdx.y][threadIdx.x * 2 + 1 + L2] = r1_value;
        }
        __syncthreads();
        
        // r[2] -= r[0] + r[1]
        if (threadIdx.y == 0){
            uint32_t r0_value, r1_value;
            uint32_t r2_value, r3_value;
            uint32_t c0_value, c1_value;
            uint32_t c2_value, c3_value;
            r0_value = r[2][threadIdx.x * 4 + 0];
            r1_value = r[2][threadIdx.x * 4 + 1];
            r2_value = r[2][threadIdx.x * 4 + 2];
            r3_value = r[2][threadIdx.x * 4 + 3];
            c0_value = r[0][threadIdx.x * 4 + 0];
            c1_value = r[0][threadIdx.x * 4 + 1];
            c2_value = r[0][threadIdx.x * 4 + 2];
            c3_value = r[0][threadIdx.x * 4 + 3];
            batch_mul_sub_128_single_warp<BLOCK_SIZE>(r0_value, r1_value, r2_value, r3_value, c0_value, c1_value, c2_value, c3_value);
            c0_value = r[1][threadIdx.x * 4 + 0];
            c1_value = r[1][threadIdx.x * 4 + 1];
            c2_value = r[1][threadIdx.x * 4 + 2];
            c3_value = r[1][threadIdx.x * 4 + 3];
            batch_mul_sub_128_single_warp<BLOCK_SIZE>(r0_value, r1_value, r2_value, r3_value, c0_value, c1_value, c2_value, c3_value);
            if (threadIdx.x * 4 + 0 < L * 2){
                r[2][threadIdx.x * 4 + 0] = r0_value;
                r[2][threadIdx.x * 4 + 1] = r1_value;
            }
            if (threadIdx.x * 4 + 2 < L * 2){
                r[2][threadIdx.x * 4 + 2] = r2_value;
                r[2][threadIdx.x * 4 + 3] = r3_value;
            }
        }
        __syncthreads();

        __shared__ uint32_t carry_prop[3];

        // add r[2] back to r[1] and r[0]
        // r[1] : [0 .. L_split)  [L_split .. L_split * 2)
        // r[0] :                                          [L_split * 2 .. L_split + L * 2) [ L_split + L * 2 .. L_total * 2)
        // r[2] :                 [L_split .. L_split * 2) [L_split * 2 .. L_split + L * 2)
        //                          threadIdx.y == 0             threadIdx.y == 1               threadIdx.y == 2
        uint32_t r0_value, r1_value;
        uint32_t c0_value, c1_value;
        if (threadIdx.y == 0){
            r0_value = (threadIdx.x * 2 < L_split) ? r[1][threadIdx.x * 2 + L_split] : 0;
            r1_value = (threadIdx.x * 2 + 1 < L_split) ? r[1][threadIdx.x * 2 + 1 + L_split] : 0;
            c0_value = (threadIdx.x * 2 < L_split) ? r[2][threadIdx.x * 2] : 0xffffffff;
            c1_value = (threadIdx.x * 2 + 1 < L_split) ? r[2][threadIdx.x * 2 + 1] : 0xffffffff;
        }else if (threadIdx.y == 1){
            r0_value = (threadIdx.x * 2 < L * 2 - L_split) ? r[2][threadIdx.x * 2 + L_split] : 0;
            r1_value = (threadIdx.x * 2 + 1 < L * 2 - L_split) ? r[2][threadIdx.x * 2 + 1 + L_split] : 0;
            c0_value = (threadIdx.x * 2 < L * 2 - L_split) ? r[0][threadIdx.x * 2] : 0xffffffff;
            c1_value = (threadIdx.x * 2 + 1 < L * 2 - L_split) ? r[0][threadIdx.x * 2 + 1] : 0xffffffff;
        }else{
            r0_value = (threadIdx.x * 2 < L_total * 2 - L_split - L * 2) ? r[0][threadIdx.x * 2 - L_split + L * 2] : 0;
            r1_value = (threadIdx.x * 2 + 1 < L_total * 2 - L_split - L * 2) ? r[0][threadIdx.x * 2 + 1 - L_split + L * 2] : 0;
            c0_value = 0;
            c1_value = 0;
        }

        batch_mul_add_64_all_warp<BLOCK_SIZE>(r0_value, r1_value, c0_value, c1_value, carry_prop);

        if (threadIdx.y == 0){
            if (threadIdx.x * 2 < L_split){
                r[1][threadIdx.x * 2 + L_split] = r0_value;
            }
            if (threadIdx.x * 2 + 1 < L_split){
                r[1][threadIdx.x * 2 + 1 + L_split] = r1_value;
            }
        }else if (threadIdx.y == 1){
            if (threadIdx.x * 2 < L * 2 - L_split){
                r[0][threadIdx.x * 2] = r0_value;
            }
            if (threadIdx.x * 2 + 1 < L * 2 - L_split){
                r[0][threadIdx.x * 2 + 1] = r1_value;
            }
        }else if (threadIdx.y == 2){
            if (threadIdx.x * 2 < L_total * 2 - L_split - L * 2){
                r[0][threadIdx.x * 2 - L_split + L * 2] = r0_value;
            }
            if (threadIdx.x * 2 + 1 < L_total * 2 - L_split - L * 2){
                r[0][threadIdx.x * 2 + 1 - L_split + L * 2] = r1_value;
            }
        }
        __syncthreads();

        if (threadIdx.y == 0){
            for (int j = threadIdx.x; j < L_split * 2; j += BLOCK_SIZE){
                ret[idx * L_total * 2 + j] = r[1][j];
            }
        }else if (threadIdx.y == 1){
            for (int j = threadIdx.x; j < L_total * 2 - L_split * 2; j += BLOCK_SIZE){
                ret[idx * L_total * 2 + L_split * 2 + j] = r[0][j];
            }
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
__global__ void batch_mul_toom22_directlv2_kernel(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L_total){
    int L_split = (L_total + 3) >> 2;
    int L_quad = L_split + 1;
    __shared__ uint32_t a[9][BLOCK_SIZE * 2];
    __shared__ uint32_t b[9][BLOCK_SIZE * 2];
    __shared__ uint32_t r[9][BLOCK_SIZE * 4];
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    for (int i0 = 0; i0 < L_quad; i0 += BLOCK_SIZE * 2){
        a[threadIdx.y][i0 + threadIdx.x * 2] = 0;
        a[threadIdx.y][i0 + threadIdx.x * 2 + 1] = 0;
        b[threadIdx.y][i0 + threadIdx.x * 2] = 0;
        b[threadIdx.y][i0 + threadIdx.x * 2 + 1] = 0;
    }
    __syncthreads();
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        // load and fuse:
        // a[2] = a[0] + a[1], a[5] = a[3] + a[4]
        // b[2] = b[0] + b[1], b[5] = b[3] + b[4]
        if (threadIdx.y < 4){
            bool is_b = (threadIdx.y >= 2);
            bool high_pair = (threadIdx.y & 1);
            uint32_t * src = is_b ? B : A;
            uint32_t (*dst)[BLOCK_SIZE * 2] = is_b ? b : a;

            int lower_part = high_pair ? 2 : 0;
            int upper_part = high_pair ? 3 : 1;
            int lower_idx = high_pair ? 3 : 0;
            int upper_idx = high_pair ? 4 : 1;
            int sum_idx = high_pair ? 5 : 2;
            int upper_len = high_pair ? (L_total - L_split * 3) : L_split;

            for (int j_base = 0; j_base < L_quad; j_base += BLOCK_SIZE * 2){
                int j = j_base + threadIdx.x * 2;
                uint32_t r0_value = (j < L_split) ? src[idx * L_total + lower_part * L_split + j] : 0;
                uint32_t r1_value = (j + 1 < L_split) ? src[idx * L_total + lower_part * L_split + j + 1] : 0;
                uint32_t c0_value = (j < upper_len) ? src[idx * L_total + upper_part * L_split + j] : 0;
                uint32_t c1_value = (j + 1 < upper_len) ? src[idx * L_total + upper_part * L_split + j + 1] : 0;
                batch_mul_add_64_single_warp<BLOCK_SIZE>(r0_value, r1_value, c0_value, c1_value);

                if (j < L_quad){
                    dst[lower_idx][j] = (j < L_split) ? src[idx * L_total + lower_part * L_split + j] : 0;
                    dst[upper_idx][j] = (j < upper_len) ? src[idx * L_total + upper_part * L_split + j] : 0;
                    dst[sum_idx][j] = r0_value;
                }
                if (j + 1 < L_quad){
                    dst[lower_idx][j + 1] = (j + 1 < L_split) ? src[idx * L_total + lower_part * L_split + j + 1] : 0;
                    dst[upper_idx][j + 1] = (j + 1 < upper_len) ? src[idx * L_total + upper_part * L_split + j + 1] : 0;
                    dst[sum_idx][j + 1] = r1_value;
                }
            }
        }
        __syncthreads();

        // transform a and b
        // a[2], a[5], b[2], b[5] are fused into load stage
        // a[6] = a[0] + a[3], a[7] = a[1] + a[4], a[8] = a[2] + a[5]
        // b[6] = b[0] + b[3], b[7] = b[1] + b[4], b[8] = b[2] + b[5]
        // blockDim.y is fixed to 9 for this kernel launch
        if (threadIdx.y < 6){
            bool is_b = (threadIdx.y >= 3);
            int local = is_b ? (threadIdx.y - 3) : threadIdx.y; // 0,1,2
            int src0_id = local;
            int src1_id = local + 3;
            int dst_id = local + 6;
            uint32_t (*src_dst)[BLOCK_SIZE * 2] = is_b ? b : a;

            uint32_t r0_value = src_dst[src0_id][threadIdx.x * 2 + 0];
            uint32_t r1_value = src_dst[src0_id][threadIdx.x * 2 + 1];
            uint32_t c0_value = src_dst[src1_id][threadIdx.x * 2 + 0];
            uint32_t c1_value = src_dst[src1_id][threadIdx.x * 2 + 1];
            batch_mul_add_64_single_warp<BLOCK_SIZE>(r0_value, r1_value, c0_value, c1_value);
            if (threadIdx.x * 2 + 0 < L_quad){
                src_dst[dst_id][threadIdx.x * 2 + 0] = r0_value;
            }
            if (threadIdx.x * 2 + 1 < L_quad){
                src_dst[dst_id][threadIdx.x * 2 + 1] = r1_value;
            }
        }
        __syncthreads();

        // r[i] = a[i] * b[i]
        if (true){ // 2.569ms
            uint32_t a0_value, a1_value;
            uint32_t r0_value, r1_value;
            a0_value = a[threadIdx.y][threadIdx.x * 2];
            a1_value = a[threadIdx.y][threadIdx.x * 2 + 1];
            r0_value = 0;
            r1_value = 0;
            uint32_t c0_value = 0, c1_value = 0;
            for (int j = 0; j < L_quad; j += 2){
                uint32_t b0_value = b[threadIdx.y][j];
                uint32_t b1_value = b[threadIdx.y][j + 1];

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
                    r[threadIdx.y][j] = r0_value;
                    r[threadIdx.y][j + 1] = r1_value;
                }
                r0_value = __shfl_down_sync(warp_mask, r0_value, 1, BLOCK_SIZE);
                r1_value = __shfl_down_sync(warp_mask, r1_value, 1, BLOCK_SIZE);
                if (threadIdx.x == BLOCK_SIZE - 1){
                    r0_value = 0;
                    r1_value = 0;
                }
            }
            int L2 = (L_quad + 1) & -2;

            batch_mul_add_64_single_warp<BLOCK_SIZE>(r0_value, r1_value, c0_value, c1_value);

            r[threadIdx.y][threadIdx.x * 2 + L2] = r0_value;
            r[threadIdx.y][threadIdx.x * 2 + 1 + L2] = r1_value;
        }
        __syncthreads();

        // r[2] -= r[0] + r[1], r[5] -= r[3] + r[4], r[8] -= r[6] + r[7]
        if (threadIdx.y < 3){ // 0.116ms
            uint32_t r0_value, r1_value;
            uint32_t r2_value, r3_value;
            uint32_t c0_value, c1_value;
            uint32_t c2_value, c3_value;
            r0_value = r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 0];
            r1_value = r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 1];
            r2_value = r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 2];
            r3_value = r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 3];
            c0_value = r[threadIdx.y * 3 + 0][threadIdx.x * 4 + 0];
            c1_value = r[threadIdx.y * 3 + 0][threadIdx.x * 4 + 1];
            c2_value = r[threadIdx.y * 3 + 0][threadIdx.x * 4 + 2];
            c3_value = r[threadIdx.y * 3 + 0][threadIdx.x * 4 + 3];
            batch_mul_sub_128_single_warp<BLOCK_SIZE>(r0_value, r1_value, r2_value, r3_value, c0_value, c1_value, c2_value, c3_value);
            c0_value = r[threadIdx.y * 3 + 1][threadIdx.x * 4 + 0];
            c1_value = r[threadIdx.y * 3 + 1][threadIdx.x * 4 + 1];
            c2_value = r[threadIdx.y * 3 + 1][threadIdx.x * 4 + 2];
            c3_value = r[threadIdx.y * 3 + 1][threadIdx.x * 4 + 3];
            batch_mul_sub_128_single_warp<BLOCK_SIZE>(r0_value, r1_value, r2_value, r3_value, c0_value, c1_value, c2_value, c3_value);
            if (threadIdx.x * 4 + 0 < L_quad * 2){
                r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 0] = r0_value;
                r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 1] = r1_value;
            }
            if (threadIdx.x * 4 + 2 < L_quad * 2){
                r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 2] = r2_value;
                r[threadIdx.y * 3 + 2][threadIdx.x * 4 + 3] = r3_value;
            }
        }
        __syncthreads();

        // r[6] -= r[0] + r[3], r[7] -= r[1] + r[4], r[8] -= r[2] + r[5]
        if (threadIdx.y < 3){ // 0.104ms
            uint32_t r0_value, r1_value;
            uint32_t r2_value, r3_value;
            uint32_t c0_value, c1_value;
            uint32_t c2_value, c3_value;
            r0_value = r[threadIdx.y + 6][threadIdx.x * 4 + 0];
            r1_value = r[threadIdx.y + 6][threadIdx.x * 4 + 1];
            r2_value = r[threadIdx.y + 6][threadIdx.x * 4 + 2];
            r3_value = r[threadIdx.y + 6][threadIdx.x * 4 + 3];
            c0_value = r[threadIdx.y + 0][threadIdx.x * 4 + 0];
            c1_value = r[threadIdx.y + 0][threadIdx.x * 4 + 1];
            c2_value = r[threadIdx.y + 0][threadIdx.x * 4 + 2];
            c3_value = r[threadIdx.y + 0][threadIdx.x * 4 + 3];
            batch_mul_sub_128_single_warp<BLOCK_SIZE>(r0_value, r1_value, r2_value, r3_value, c0_value, c1_value, c2_value, c3_value);
            c0_value = r[threadIdx.y + 3][threadIdx.x * 4 + 0];
            c1_value = r[threadIdx.y + 3][threadIdx.x * 4 + 1];
            c2_value = r[threadIdx.y + 3][threadIdx.x * 4 + 2];
            c3_value = r[threadIdx.y + 3][threadIdx.x * 4 + 3];
            batch_mul_sub_128_single_warp<BLOCK_SIZE>(r0_value, r1_value, r2_value, r3_value, c0_value, c1_value, c2_value, c3_value);
            if (threadIdx.x * 4 + 0 < L_quad * 2){
                r[threadIdx.y + 6][threadIdx.x * 4 + 0] = r0_value;
                r[threadIdx.y + 6][threadIdx.x * 4 + 1] = r1_value;
            }
            if (threadIdx.x * 4 + 2 < L_quad * 2){
                r[threadIdx.y + 6][threadIdx.x * 4 + 2] = r2_value;
                r[threadIdx.y + 6][threadIdx.x * 4 + 3] = r3_value;
            }
        }
        __syncthreads();

        // collapse r[2] to r[1]|r[0], r[5] to r[4]|r[3], r[8] to r[7]|r[6]
        // we will use a trick: just access r[0][x] with out-of-bound x, and it automatically wraps to r[1]
        if (true){ // 0.191ms
            int group_idx = threadIdx.y / 3;
            int rank_in_group = threadIdx.y % 3;
            int j_idx = rank_in_group * BLOCK_SIZE * 2 + threadIdx.x * 2;
            uint32_t r0_value, r1_value;
            uint32_t c0_value, c1_value;
            uint32_t t0_value, t1_value;
            c0_value = (j_idx < L_quad * 2) ? r[group_idx * 3 + 2][j_idx] : 0;
            c1_value = (j_idx + 1 < L_quad * 2) ? r[group_idx * 3 + 2][j_idx + 1] : 0;
            r0_value = (L_split + j_idx < L_quad * 2) ? r[group_idx * 3 + 0][L_split + j_idx] : 0;
            r1_value = (L_split + j_idx + 1 < L_quad * 2) ? r[group_idx * 3 + 0][L_split + j_idx + 1] : 0;
            t0_value = (j_idx >= L_split) ? r[group_idx * 3 + 1][j_idx - L_split] : 0;
            t1_value = (j_idx + 1 >= L_split) ? r[group_idx * 3 + 1][j_idx + 1 - L_split] : 0;

            __shared__ ushort2 carry_prop[9];
            batch_mul_add3_64_grouped_warp<BLOCK_SIZE>(
                r0_value, r1_value, c0_value, c1_value, t0_value, t1_value,
                rank_in_group, 3, true,
                carry_prop + group_idx * 3
            );
            if (j_idx < L_quad * 2 + L_split){
                r[group_idx * 3 + 0][L_split + j_idx] = r0_value;
                r[group_idx * 3 + 0][L_split + j_idx + 1] = r1_value;
            }
        }
        __syncthreads();

        // collapse r[6] to r[3]|r[0]
        if (true){ // 0.065ms
            int j_idx = threadIdx.y * BLOCK_SIZE * 2 + threadIdx.x * 2;
            uint32_t r0_value, r1_value;
            uint32_t c0_value, c1_value;
            if (threadIdx.y < 6){
                c0_value = (j_idx < L_quad * 2 + L_split * 2) ? r[6][j_idx] : 0;
                c1_value = (j_idx + 1 < L_quad * 2 + L_split * 2) ? r[6][j_idx + 1] : 0;
                if (j_idx < L_split * 2){
                    r0_value = r[0][j_idx + L_split * 2];
                }else if (j_idx < L_split * 4 + L_quad * 2){
                    r0_value = r[3][j_idx - L_split * 2];
                }else{
                    r0_value = 0;
                }
                if (j_idx + 1 < L_split * 2){
                    r1_value = r[0][j_idx + 1 + L_split * 2];
                }else if (j_idx + 1 < L_split * 4 + L_quad * 2){
                    r1_value = r[3][j_idx + 1 - L_split * 2];
                }else{
                    r1_value = 0;
                }
            }
            __shared__ uint32_t carry_prop[9];
            batch_mul_add_64_grouped_warp<BLOCK_SIZE>(
                r0_value, r1_value, c0_value, c1_value,
                threadIdx.y, 6, threadIdx.y < 6,
                carry_prop
            );
            if (j_idx < L_quad * 2 + L_split * 4){
                r[0][L_split * 2 + j_idx] = r0_value;
                r[0][L_split * 2 + j_idx + 1] = r1_value;
            }
        }
        __syncthreads();

        for (int j = threadIdx.x + threadIdx.y * BLOCK_SIZE; j < L_total * 2; j += blockDim.y * BLOCK_SIZE){
            ret[idx * L_total * 2 + j] = r[0][j];
        }
        __syncthreads();
    }
}

static const int BATCH_MUL_TOOM22_LV2_MAX = (64 - 1) * 4; // 252

// Internal recursive function
static void batch_mul_toom22_internal(uint32_t * A, uint32_t * B, uint32_t * ret, 
    uint32_t * workspace, int N, int L) {
        
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        batch_mul_direct(A, B, ret, N, L);
        return;
    }
    
    int L_split = ceil_div(L, 2);  // Actual split point = ceil(L/2)
    int L_half = L_split + 1;       // Padded buffer size
    
    int c_size = L_half * 2;

    uint32_t * C_combined;

    if (L_half <= BATCH_MUL_DIRECT_L_MAX){
        int num_blocks = N;
        if (num_blocks > 65536) num_blocks = 65536;
        batch_mul_toom22_directlv1_kernel<32><<<num_blocks, dim3(32, 3, 1)>>>(
            A, B, ret, N, L
        );
    }else if (L <= BATCH_MUL_TOOM22_LV2_MAX){
        int num_blocks = N;
        if (num_blocks > 65536) num_blocks = 65536;
        batch_mul_toom22_directlv2_kernel<32><<<num_blocks, dim3(32, 9, 1)>>>(
            A, B, ret, N, L
        );
    }else{
        uint32_t * A_combined = workspace;
        uint32_t * B_combined = A_combined + (size_t)3 * N * L_half;
        C_combined = B_combined + (size_t)3 * N * L_half;
        uint32_t * next_workspace = C_combined + (size_t)3 * N * c_size;

        int num_blocks = N;
        if (num_blocks > 65536) num_blocks = 65536;
        int num_warps = (L_half + 64 - 1) / 64;
        batch_mul_toom22_transform_kernel<32><<<dim3(num_blocks, 2, 1), dim3(32, num_warps, 1)>>>(
            A, B, A_combined, B_combined, N, L, L_split, L_half
        );
        
        // Single recursive call with 3N instances
        batch_mul_toom22_internal(A_combined, B_combined, C_combined, next_workspace, 3 * N, L_half);     

        int num_blocks_2 = N;
        if (num_blocks_2 > 65536) num_blocks_2 = 65536;
        int num_warps_2 = (L_split + L_half * 2 + 64 - 1) / 64;
        batch_mul_toom22_reconstruct_kernel<32><<<num_blocks_2, dim3(32, num_warps_2, 1)>>>(
            ret, C_combined, N, L, L_split, L_half
        );

    }
}

// Compute total workspace size recursively for internal use
static size_t workspace_size_words_internal(int N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    int L_split = ceil_div(L, 2);
    int L_half = L_split + 1;
    int c_size = L_half * 2;

    size_t total = 0;

    if (L > BATCH_MUL_TOOM22_LV2_MAX) {
        total += (size_t)3 * N * L_half * 2; // A_combined + B_combined
        total += (size_t)3 * N * c_size; // C_combined
    }
    
    total += workspace_size_words_internal(3 * N, L_half);
    return total;
}

static int get_N_max(int N, int L){
    if (L > BATCH_MUL_TOOM22_LV2_MAX) {
        return min(170 * 128 * 1, N);
    } else {
        return N; // unlimitted as we do not need workspace
    }
}

size_t batch_mul_toom22_workspace_size(int N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    int chunk_N = get_N_max(N, L);
    
    return workspace_size_words_internal(chunk_N, L) * sizeof(uint32_t);
}

void batch_mul_toom22(uint32_t * A, uint32_t * B, uint32_t * ret, uint32_t * workspace, int N, int L) {
    if (L > BATCH_MUL_TOOM22_L_MAX) {
        return;
    }
    
    int N_max = get_N_max(N, L);

    for (int offset = 0; offset < N; offset += N_max) {
        int chunk_N = (offset + N_max <= N) ? N_max : (N - offset);
        
        uint32_t * A_chunk = A + (size_t)offset * L;
        uint32_t * B_chunk = B + (size_t)offset * L;
        uint32_t * ret_chunk = ret + (size_t)offset * (L * 2);
        batch_mul_toom22_internal(A_chunk, B_chunk, ret_chunk, workspace, chunk_N, L);
    }
}
