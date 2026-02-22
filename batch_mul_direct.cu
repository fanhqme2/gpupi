#include <cuda.h>
#include <cuda_runtime.h>
#include "batch_mul_direct.h"
#include "batch_mul_8x8_block.h"

// 32 thread version
template<int L_BLOCK, int BLOCK_SIZE>
__global__ void batch_mul_direct_kernel3(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L){
    __shared__ uint32_t a[BATCH_MUL_DIRECT_L_MAX][BLOCK_SIZE + 1];
    __shared__ uint32_t b[BATCH_MUL_DIRECT_L_MAX][BLOCK_SIZE + 1], outer_carry[BLOCK_SIZE + 1];
    __shared__ uint32_t r[L_BLOCK * 2][BLOCK_SIZE + 1];
    for (int idx0 = blockIdx.x * BLOCK_SIZE; idx0 < N; idx0 += gridDim.x * BLOCK_SIZE){
        int batch_len = min(BLOCK_SIZE, N - idx0);
        if (batch_len == BLOCK_SIZE){
            for (int j = 0; j < L; j += BLOCK_SIZE){
                if (j + threadIdx.x < L){
                    for (int i = 0; i < BLOCK_SIZE; i ++){                
                        a[j + threadIdx.x][i] = A[(idx0 + i) * L + j + threadIdx.x];
                        b[j + threadIdx.x][i] = B[(idx0 + i) * L + j + threadIdx.x];
                    }
                }else{
                    for (int i = 0; i < BLOCK_SIZE; i ++){                
                        a[j + threadIdx.x][i] = 0;
                        b[j + threadIdx.x][i] = 0;
                    }
                }
            }
        }else{
            for (int j = 0; j < L; j += BLOCK_SIZE){
                if (j + threadIdx.x < L){
                    for (int i = 0; i < batch_len; i ++){
                        a[j + threadIdx.x][i] = A[(idx0 + i) * L + j + threadIdx.x];
                        b[j + threadIdx.x][i] = B[(idx0 + i) * L + j + threadIdx.x];
                    }
                }else{
                    for (int i = 0; i < batch_len; i ++){
                        a[j + threadIdx.x][i] = 0;
                        b[j + threadIdx.x][i] = 0;
                    }
                }
            }
        }
        for (int i = 0; i < L_BLOCK * 2; i ++){
            r[i][threadIdx.x] = 0;
        }
        outer_carry[threadIdx.x] = 0;
        __syncthreads();
        int L_down = (L - 1) & -L_BLOCK;
        for (int sum_ij = 0; sum_ij <= L_down * 2; sum_ij += L_BLOCK){
            for (int i0 = max(0, sum_ij - L_down); i0 < L && i0 <= sum_ij; i0 += L_BLOCK){
                int j0 = sum_ij - i0;
                uint64_t standing_carry = 0;
                // r  r  r  r  r  r  r  r
                //          s  x  x  x  x
                //       s  x  x  x  x
                //    s  x  x  x  x
                // s  x  x  x  x
                /*for (int i = 0; i < 32; i ++){
                    uint64_t running_carry = 0;
                    for (int j = 0; j < 32; j ++){
                        running_carry += (uint64_t)a[i0 + i][threadIdx.x] * (uint64_t)b[j0 + j][threadIdx.x] + r[i + j][threadIdx.x];
                        r[i + j][threadIdx.x] = (uint32_t)running_carry;
                        running_carry >>= 32;
                    }
                    standing_carry += r[i + 32 - 1][threadIdx.x];
                    r[i + 32 - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += running_carry;
                }*/
                for (int i = 0; i < L_BLOCK; i += 8){
                    uint4 carryv1 = make_uint4(0, 0, 0, 0);
                    uint4 carryv2 = make_uint4(0, 0, 0, 0);
                    for (int j = 0; j < L_BLOCK; j += 8){
                        uint4 av1 = make_uint4(a[i0 + i][threadIdx.x], a[i0 + i + 1][threadIdx.x], a[i0 + i + 2][threadIdx.x], a[i0 + i + 3][threadIdx.x]);
                        uint4 av2 = make_uint4(a[i0 + i + 4][threadIdx.x], a[i0 + i + 5][threadIdx.x], a[i0 + i + 6][threadIdx.x], a[i0 + i + 7][threadIdx.x]);
                        uint4 bv1 = make_uint4(b[j0 + j][threadIdx.x], b[j0 + j + 1][threadIdx.x], b[j0 + j + 2][threadIdx.x], b[j0 + j + 3][threadIdx.x]);
                        uint4 bv2 = make_uint4(b[j0 + j + 4][threadIdx.x], b[j0 + j + 5][threadIdx.x], b[j0 + j + 6][threadIdx.x], b[j0 + j + 7][threadIdx.x]);
                        uint4 r0v = make_uint4(r[i + j][threadIdx.x], r[i + j + 1][threadIdx.x], r[i + j + 2][threadIdx.x], r[i + j + 3][threadIdx.x]);
                        uint4 r1v = make_uint4(r[i + j + 4][threadIdx.x], r[i + j + 5][threadIdx.x], r[i + j + 6][threadIdx.x], r[i + j + 7][threadIdx.x]);
                        uint4 r2v = make_uint4(r[i + j + 8][threadIdx.x], r[i + j + 9][threadIdx.x], r[i + j + 10][threadIdx.x], r[i + j + 11][threadIdx.x]);
                        uint4 r3v = make_uint4(r[i + j + 12][threadIdx.x], r[i + j + 13][threadIdx.x], r[i + j + 14][threadIdx.x], r[i + j + 15][threadIdx.x]);

                        BATCH_MUL_8X8_BLOCK(av1, av2, bv1, bv2, r0v, r1v, r2v, r3v, carryv1, carryv2);
                        
                        r[i + j][threadIdx.x] = r0v.x;
                        r[i + j + 1][threadIdx.x] = r0v.y;
                        r[i + j + 2][threadIdx.x] = r0v.z;
                        r[i + j + 3][threadIdx.x] = r0v.w;
                        r[i + j + 4][threadIdx.x] = r1v.x;
                        r[i + j + 5][threadIdx.x] = r1v.y;
                        r[i + j + 6][threadIdx.x] = r1v.z;
                        r[i + j + 7][threadIdx.x] = r1v.w;
                        r[i + j + 8][threadIdx.x] = r2v.x;
                        r[i + j + 9][threadIdx.x] = r2v.y;
                        r[i + j + 10][threadIdx.x] = r2v.z;
                        r[i + j + 11][threadIdx.x] = r2v.w;
                        r[i + j + 12][threadIdx.x] = r3v.x;
                        r[i + j + 13][threadIdx.x] = r3v.y;
                        r[i + j + 14][threadIdx.x] = r3v.z;
                        r[i + j + 15][threadIdx.x] = r3v.w;
                    }
                    standing_carry += r[i + L_BLOCK - 1][threadIdx.x];
                    r[i + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv1.x;

                    standing_carry += r[i + 1 + L_BLOCK - 1][threadIdx.x];
                    r[i + 1 + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv1.y;

                    standing_carry += r[i + 2 + L_BLOCK - 1][threadIdx.x];
                    r[i + 2 + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv1.z;

                    standing_carry += r[i + 3 + L_BLOCK - 1][threadIdx.x];
                    r[i + 3 + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv1.w;

                    standing_carry += r[i + 4 + L_BLOCK - 1][threadIdx.x];
                    r[i + 4 + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv2.x;

                    standing_carry += r[i + 5 + L_BLOCK - 1][threadIdx.x];
                    r[i + 5 + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv2.y;

                    standing_carry += r[i + 6 + L_BLOCK - 1][threadIdx.x];
                    r[i + 6 + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv2.z;

                    standing_carry += r[i + 7 + L_BLOCK - 1][threadIdx.x];
                    r[i + 7 + L_BLOCK - 1][threadIdx.x] = (uint32_t)standing_carry;
                    standing_carry >>= 32;
                    standing_carry += carryv2.w;
                }
                standing_carry += r[L_BLOCK * 2 - 1][threadIdx.x];
                r[L_BLOCK * 2 - 1][threadIdx.x] = (uint32_t)standing_carry;
                outer_carry[threadIdx.x] += standing_carry >> 32;
            }
            __syncthreads();

            if (sum_ij + threadIdx.x < L * 2 && threadIdx.x < L_BLOCK){
                for (int i = 0; i < batch_len; i ++){
                    ret[(idx0 + i) * (L * 2) + sum_ij + threadIdx.x] = r[threadIdx.x][i];
                }
            }
            __syncthreads();
            for (int i = 0; i < L_BLOCK; i ++){
                r[i][threadIdx.x] = r[i + L_BLOCK][threadIdx.x];
                r[i + L_BLOCK][threadIdx.x] = 0;
            }
            r[L_BLOCK][threadIdx.x] = outer_carry[threadIdx.x];
            outer_carry[threadIdx.x] = 0;
        }
        __syncthreads();
        if (L_down * 2 + L_BLOCK + threadIdx.x < L * 2 && threadIdx.x < L_BLOCK){
            for (int i = 0; i < batch_len; i ++){
                ret[(idx0 + i) * (L * 2) + L_down * 2 + L_BLOCK + threadIdx.x] = r[threadIdx.x][i];
            }
        }
        __syncthreads();
    }
}

void batch_mul_direct(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L){
    const int threads_per_block = 32;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (num_blocks >= 170 * 8) {
        num_blocks = 170 * 8;
    }
    batch_mul_direct_kernel3<16, threads_per_block><<<num_blocks, threads_per_block>>>(A, B, ret, N, L);
}