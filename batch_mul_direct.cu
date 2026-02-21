#include <cuda.h>
#include <cuda_runtime.h>
#include "batch_mul_direct.h"

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

                        uint64_t mul00 = (uint64_t)bv1.x * (uint64_t)av1.x + r0v.x + carryv1.x;
                        r0v.x = (uint32_t)mul00;
                        carryv1.x = mul00 >> 32;
                        uint64_t mul01 = (uint64_t)bv1.x * (uint64_t)av1.y + r0v.y + carryv1.x;
                        r0v.y = (uint32_t)mul01;
                        carryv1.x = mul01 >> 32;
                        uint64_t mul02 = (uint64_t)bv1.x * (uint64_t)av1.z + r0v.z + carryv1.x;
                        r0v.z = (uint32_t)mul02;
                        carryv1.x = mul02 >> 32;
                        uint64_t mul03 = (uint64_t)bv1.x * (uint64_t)av1.w + r0v.w + carryv1.x;
                        r0v.w = (uint32_t)mul03;
                        carryv1.x = mul03 >> 32;
                        uint64_t mul04 = (uint64_t)bv1.x * (uint64_t)av2.x + r1v.x + carryv1.x;
                        r1v.x = (uint32_t)mul04;
                        carryv1.x = mul04 >> 32;
                        uint64_t mul05 = (uint64_t)bv1.x * (uint64_t)av2.y + r1v.y + carryv1.x;
                        r1v.y = (uint32_t)mul05;
                        carryv1.x = mul05 >> 32;
                        uint64_t mul06 = (uint64_t)bv1.x * (uint64_t)av2.z + r1v.z + carryv1.x;
                        r1v.z = (uint32_t)mul06;
                        carryv1.x = mul06 >> 32;
                        uint64_t mul07 = (uint64_t)bv1.x * (uint64_t)av2.w + r1v.w + carryv1.x;
                        r1v.w = (uint32_t)mul07;
                        carryv1.x = mul07 >> 32;

                        uint64_t mul10 = (uint64_t)bv1.y * (uint64_t)av1.x + r0v.y + carryv1.y;
                        r0v.y = (uint32_t)mul10;
                        carryv1.y = mul10 >> 32;
                        uint64_t mul11 = (uint64_t)bv1.y * (uint64_t)av1.y + r0v.z + carryv1.y;
                        r0v.z = (uint32_t)mul11;
                        carryv1.y = mul11 >> 32;
                        uint64_t mul12 = (uint64_t)bv1.y * (uint64_t)av1.z + r0v.w + carryv1.y;
                        r0v.w = (uint32_t)mul12;
                        carryv1.y = mul12 >> 32;
                        uint64_t mul13 = (uint64_t)bv1.y * (uint64_t)av1.w + r1v.x + carryv1.y;
                        r1v.x = (uint32_t)mul13;
                        carryv1.y = mul13 >> 32;
                        uint64_t mul14 = (uint64_t)bv1.y * (uint64_t)av2.x + r1v.y + carryv1.y;
                        r1v.y = (uint32_t)mul14;
                        carryv1.y = mul14 >> 32;
                        uint64_t mul15 = (uint64_t)bv1.y * (uint64_t)av2.y + r1v.z + carryv1.y;
                        r1v.z = (uint32_t)mul15;
                        carryv1.y = mul15 >> 32;
                        uint64_t mul16 = (uint64_t)bv1.y * (uint64_t)av2.z + r1v.w + carryv1.y;
                        r1v.w = (uint32_t)mul16;
                        carryv1.y = mul16 >> 32;
                        uint64_t mul17 = (uint64_t)bv1.y * (uint64_t)av2.w + r2v.x + carryv1.y;
                        r2v.x = (uint32_t)mul17;
                        carryv1.y = mul17 >> 32;

                        uint64_t mul20 = (uint64_t)bv1.z * (uint64_t)av1.x + r0v.z + carryv1.z;
                        r0v.z = (uint32_t)mul20;
                        carryv1.z = mul20 >> 32;
                        uint64_t mul21 = (uint64_t)bv1.z * (uint64_t)av1.y + r0v.w + carryv1.z;
                        r0v.w = (uint32_t)mul21;
                        carryv1.z = mul21 >> 32;
                        uint64_t mul22 = (uint64_t)bv1.z * (uint64_t)av1.z + r1v.x + carryv1.z;
                        r1v.x = (uint32_t)mul22;
                        carryv1.z = mul22 >> 32;
                        uint64_t mul23 = (uint64_t)bv1.z * (uint64_t)av1.w + r1v.y + carryv1.z;
                        r1v.y = (uint32_t)mul23;
                        carryv1.z = mul23 >> 32;
                        uint64_t mul24 = (uint64_t)bv1.z * (uint64_t)av2.x + r1v.z + carryv1.z;
                        r1v.z = (uint32_t)mul24;
                        carryv1.z = mul24 >> 32;
                        uint64_t mul25 = (uint64_t)bv1.z * (uint64_t)av2.y + r1v.w + carryv1.z;
                        r1v.w = (uint32_t)mul25;
                        carryv1.z = mul25 >> 32;
                        uint64_t mul26 = (uint64_t)bv1.z * (uint64_t)av2.z + r2v.x + carryv1.z;
                        r2v.x = (uint32_t)mul26;
                        carryv1.z = mul26 >> 32;
                        uint64_t mul27 = (uint64_t)bv1.z * (uint64_t)av2.w + r2v.y + carryv1.z;
                        r2v.y = (uint32_t)mul27;
                        carryv1.z = mul27 >> 32;

                        uint64_t mul30 = (uint64_t)bv1.w * (uint64_t)av1.x + r0v.w + carryv1.w;
                        r0v.w = (uint32_t)mul30;
                        carryv1.w = mul30 >> 32;
                        uint64_t mul31 = (uint64_t)bv1.w * (uint64_t)av1.y + r1v.x + carryv1.w;
                        r1v.x = (uint32_t)mul31;
                        carryv1.w = mul31 >> 32;
                        uint64_t mul32 = (uint64_t)bv1.w * (uint64_t)av1.z + r1v.y + carryv1.w;
                        r1v.y = (uint32_t)mul32;
                        carryv1.w = mul32 >> 32;
                        uint64_t mul33 = (uint64_t)bv1.w * (uint64_t)av1.w + r1v.z + carryv1.w;
                        r1v.z = (uint32_t)mul33;
                        carryv1.w = mul33 >> 32;
                        uint64_t mul34 = (uint64_t)bv1.w * (uint64_t)av2.x + r1v.w + carryv1.w;
                        r1v.w = (uint32_t)mul34;
                        carryv1.w = mul34 >> 32;
                        uint64_t mul35 = (uint64_t)bv1.w * (uint64_t)av2.y + r2v.x + carryv1.w;
                        r2v.x = (uint32_t)mul35;
                        carryv1.w = mul35 >> 32;
                        uint64_t mul36 = (uint64_t)bv1.w * (uint64_t)av2.z + r2v.y + carryv1.w;
                        r2v.y = (uint32_t)mul36;
                        carryv1.w = mul36 >> 32;
                        uint64_t mul37 = (uint64_t)bv1.w * (uint64_t)av2.w + r2v.z + carryv1.w;
                        r2v.z = (uint32_t)mul37;
                        carryv1.w = mul37 >> 32;

                        uint64_t mul40 = (uint64_t)bv2.x * (uint64_t)av1.x + r1v.x + carryv2.x;
                        r1v.x = (uint32_t)mul40;
                        carryv2.x = mul40 >> 32;
                        uint64_t mul41 = (uint64_t)bv2.x * (uint64_t)av1.y + r1v.y + carryv2.x;
                        r1v.y = (uint32_t)mul41;
                        carryv2.x = mul41 >> 32;
                        uint64_t mul42 = (uint64_t)bv2.x * (uint64_t)av1.z + r1v.z + carryv2.x;
                        r1v.z = (uint32_t)mul42;
                        carryv2.x = mul42 >> 32;
                        uint64_t mul43 = (uint64_t)bv2.x * (uint64_t)av1.w + r1v.w + carryv2.x;
                        r1v.w = (uint32_t)mul43;
                        carryv2.x = mul43 >> 32;
                        uint64_t mul44 = (uint64_t)bv2.x * (uint64_t)av2.x + r2v.x + carryv2.x;
                        r2v.x = (uint32_t)mul44;
                        carryv2.x = mul44 >> 32;
                        uint64_t mul45 = (uint64_t)bv2.x * (uint64_t)av2.y + r2v.y + carryv2.x;
                        r2v.y = (uint32_t)mul45;
                        carryv2.x = mul45 >> 32;
                        uint64_t mul46 = (uint64_t)bv2.x * (uint64_t)av2.z + r2v.z + carryv2.x;
                        r2v.z = (uint32_t)mul46;
                        carryv2.x = mul46 >> 32;
                        uint64_t mul47 = (uint64_t)bv2.x * (uint64_t)av2.w + r2v.w + carryv2.x;
                        r2v.w = (uint32_t)mul47;
                        carryv2.x = mul47 >> 32;

                        uint64_t mul50 = (uint64_t)bv2.y * (uint64_t)av1.x + r1v.y + carryv2.y;
                        r1v.y = (uint32_t)mul50;
                        carryv2.y = mul50 >> 32;
                        uint64_t mul51 = (uint64_t)bv2.y * (uint64_t)av1.y + r1v.z + carryv2.y;
                        r1v.z = (uint32_t)mul51;
                        carryv2.y = mul51 >> 32;
                        uint64_t mul52 = (uint64_t)bv2.y * (uint64_t)av1.z + r1v.w + carryv2.y;
                        r1v.w = (uint32_t)mul52;
                        carryv2.y = mul52 >> 32;
                        uint64_t mul53 = (uint64_t)bv2.y * (uint64_t)av1.w + r2v.x + carryv2.y;
                        r2v.x = (uint32_t)mul53;
                        carryv2.y = mul53 >> 32;
                        uint64_t mul54 = (uint64_t)bv2.y * (uint64_t)av2.x + r2v.y + carryv2.y;
                        r2v.y = (uint32_t)mul54;
                        carryv2.y = mul54 >> 32;
                        uint64_t mul55 = (uint64_t)bv2.y * (uint64_t)av2.y + r2v.z + carryv2.y;
                        r2v.z = (uint32_t)mul55;
                        carryv2.y = mul55 >> 32;
                        uint64_t mul56 = (uint64_t)bv2.y * (uint64_t)av2.z + r2v.w + carryv2.y;
                        r2v.w = (uint32_t)mul56;
                        carryv2.y = mul56 >> 32;
                        uint64_t mul57 = (uint64_t)bv2.y * (uint64_t)av2.w + r3v.x + carryv2.y;
                        r3v.x = (uint32_t)mul57;
                        carryv2.y = mul57 >> 32;

                        uint64_t mul60 = (uint64_t)bv2.z * (uint64_t)av1.x + r1v.z + carryv2.z;
                        r1v.z = (uint32_t)mul60;
                        carryv2.z = mul60 >> 32;
                        uint64_t mul61 = (uint64_t)bv2.z * (uint64_t)av1.y + r1v.w + carryv2.z;
                        r1v.w = (uint32_t)mul61;
                        carryv2.z = mul61 >> 32;
                        uint64_t mul62 = (uint64_t)bv2.z * (uint64_t)av1.z + r2v.x + carryv2.z;
                        r2v.x = (uint32_t)mul62;
                        carryv2.z = mul62 >> 32;
                        uint64_t mul63 = (uint64_t)bv2.z * (uint64_t)av1.w + r2v.y + carryv2.z;
                        r2v.y = (uint32_t)mul63;
                        carryv2.z = mul63 >> 32;
                        uint64_t mul64 = (uint64_t)bv2.z * (uint64_t)av2.x + r2v.z + carryv2.z;
                        r2v.z = (uint32_t)mul64;
                        carryv2.z = mul64 >> 32;
                        uint64_t mul65 = (uint64_t)bv2.z * (uint64_t)av2.y + r2v.w + carryv2.z;
                        r2v.w = (uint32_t)mul65;
                        carryv2.z = mul65 >> 32;
                        uint64_t mul66 = (uint64_t)bv2.z * (uint64_t)av2.z + r3v.x + carryv2.z;
                        r3v.x = (uint32_t)mul66;
                        carryv2.z = mul66 >> 32;
                        uint64_t mul67 = (uint64_t)bv2.z * (uint64_t)av2.w + r3v.y + carryv2.z;
                        r3v.y = (uint32_t)mul67;
                        carryv2.z = mul67 >> 32;

                        uint64_t mul70 = (uint64_t)bv2.w * (uint64_t)av1.x + r1v.w + carryv2.w;
                        r1v.w = (uint32_t)mul70;
                        carryv2.w = mul70 >> 32;
                        uint64_t mul71 = (uint64_t)bv2.w * (uint64_t)av1.y + r2v.x + carryv2.w;
                        r2v.x = (uint32_t)mul71;
                        carryv2.w = mul71 >> 32;
                        uint64_t mul72 = (uint64_t)bv2.w * (uint64_t)av1.z + r2v.y + carryv2.w;
                        r2v.y = (uint32_t)mul72;
                        carryv2.w = mul72 >> 32;
                        uint64_t mul73 = (uint64_t)bv2.w * (uint64_t)av1.w + r2v.z + carryv2.w;
                        r2v.z = (uint32_t)mul73;
                        carryv2.w = mul73 >> 32;
                        uint64_t mul74 = (uint64_t)bv2.w * (uint64_t)av2.x + r2v.w + carryv2.w;
                        r2v.w = (uint32_t)mul74;
                        carryv2.w = mul74 >> 32;
                        uint64_t mul75 = (uint64_t)bv2.w * (uint64_t)av2.y + r3v.x + carryv2.w;
                        r3v.x = (uint32_t)mul75;
                        carryv2.w = mul75 >> 32;
                        uint64_t mul76 = (uint64_t)bv2.w * (uint64_t)av2.z + r3v.y + carryv2.w;
                        r3v.y = (uint32_t)mul76;
                        carryv2.w = mul76 >> 32;
                        uint64_t mul77 = (uint64_t)bv2.w * (uint64_t)av2.w + r3v.z + carryv2.w;
                        r3v.z = (uint32_t)mul77;
                        carryv2.w = mul77 >> 32;
                        
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