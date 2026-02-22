#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include "batch_mul_toom22.h"
#include "batch_mul_direct.h"
#include "batch_mul_8x8_block.h"

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Add with carry-out (sets carry flag)
__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Add with carry-in and carry-out (uses and sets carry flag)
__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Add with carry-in (uses carry flag, no output carry)
__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Subtract with borrow-out (sets carry flag)
__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Subtract with borrow-in and borrow-out (uses and sets carry flag)
__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Subtract with borrow-in (uses carry flag, no output borrow)
__device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Transform kernel: decomposes inputs and prepares 3N interleaved arrays
// L_split = ceil(L/2) is the actual split point
// L_half = L_split + 1 is the buffer size for padded storage
template<int L_BLOCK, int BLOCK_SIZE>
__global__ void batch_mul_toom22_transform_kernel(
    const uint32_t * A, const uint32_t * B,
    uint32_t * A_out, uint32_t * B_out,
    int N, int L, int L_split, int L_half
) {
    const uint32_t * src;
    uint32_t * dst;
    if (blockIdx.y == 0) {
        src = A;
        dst = A_out;
    } else {
        src = B;
        dst = B_out;
    }
    // L_split = ceil(L/2)
    // L_half = L_split + 1 (padded size)
    int L_lower = L_split;  // = ceil(L/2)
    int L_upper = L / 2;    // = floor(L/2)
    __shared__ uint32_t a0[L_BLOCK][BLOCK_SIZE + 1];
    __shared__ uint32_t a1[L_BLOCK][BLOCK_SIZE + 1];
    __shared__ uint32_t a_sum[L_BLOCK][BLOCK_SIZE + 1];
    for (int idx0 = blockIdx.x * BLOCK_SIZE; idx0 < N; idx0 += BLOCK_SIZE * gridDim.x) {
        uint32_t carry = 0;
        int batch_len = min(BLOCK_SIZE, N - idx0);
        for (int j0 = 0; j0 < L_half; j0 += L_BLOCK){
            for (int j = j0 + threadIdx.x; j < j0 + L_BLOCK && j < L_half; j += BLOCK_SIZE){
                for (int i = 0; i < batch_len; i ++){
                    a1[j - j0][i] = (j < L_lower) ? src[(idx0 + i) * L + j] : 0;
                    a0[j - j0][i] = (j < L_upper) ? src[(idx0 + i) * L + j + L_lower] : 0;
                }
            }
            __syncthreads();
            add_cc(carry, 0xFFFFFFFF);
            for (int j = 0; j < L_BLOCK; j++){
                a_sum[j][threadIdx.x] = addc_cc(a0[j][threadIdx.x], a1[j][threadIdx.x]);
            }
            carry = addc(0, 0);
            __syncthreads();
            for (int j = j0 + threadIdx.x; j < j0 + L_BLOCK && j < L_half; j += BLOCK_SIZE){
                for (int i = 0; i < batch_len; i ++){
                    dst[(idx0 + i) * L_half + j] = a0[j - j0][i];
                    dst[(N + idx0 + i) * L_half + j] = a1[j - j0][i];
                    dst[(2 * N + idx0 + i) * L_half + j] = a_sum[j - j0][i];
                }
            }
            __syncthreads();
        }
    }
}

// Reconstruct kernel: combines partial results
// L_split = ceil(L/2) is the actual split point (shift amount)
// L_half = L_split + 1 is the buffer size
template<int L_BLOCK, int BLOCK_SIZE>
__global__ void batch_mul_toom22_reconstruct_kernel(
    uint32_t * ret,
    const uint32_t * C,
    int N, int L, int L_split, int L_half
) {
    const uint32_t * C00 = C;
    const uint32_t * C11 = C + N * L_half * 2;
    const uint32_t * C01 = C + 2 * N * L_half * 2;
    __shared__ uint32_t part1[L_BLOCK][BLOCK_SIZE + 1];
    __shared__ uint32_t part2[L_BLOCK][BLOCK_SIZE + 1];
    __shared__ uint32_t part3[L_BLOCK][BLOCK_SIZE + 1];
    __shared__ uint32_t part4[L_BLOCK][BLOCK_SIZE + 1];
    for (int idx0 = blockIdx.x * BLOCK_SIZE; idx0 < N; idx0 += BLOCK_SIZE * gridDim.x) {
        int batch_len = min(BLOCK_SIZE, N - idx0);
        uint32_t carry = 0;
        uint32_t borrow_0 = 0;
        uint32_t borrow_1 = 0;
        for (int j0 = 0; j0 < L * 2; j0 += L_BLOCK){
            for (int j = j0 + threadIdx.x; j < j0 + L_BLOCK && j < L * 2; j += BLOCK_SIZE){
                for (int i = 0; i < batch_len; i ++){
                    part1[j - j0][i] = (
                        (j < L_split * 2) ?
                        C11[(idx0 + i) * L_half * 2 + j] : 
                        C00[(idx0 + i) * L_half * 2 + j - L_split * 2]
                    );
                    part2[j - j0][i] = (
                        (L_split <= j && j < L_split + L_half * 2) ?
                        C01[(idx0 + i) * L_half * 2 + j - L_split] :
                        0
                    );
                    part3[j - j0][i] = (
                        (L_split <= j && j < L_split + L_half * 2) ?
                        C00[(idx0 + i) * L_half * 2 + j - L_split] :
                        0
                    );
                    part4[j - j0][i] = (
                        (L_split <= j && j < L_split + L_half * 2) ?
                        C11[(idx0 + i) * L_half * 2 + j - L_split] :
                        0
                    );
                }
            }
            __syncthreads();
            sub_cc(0, borrow_0);
            for (int i = 0; i < L_BLOCK; i ++){
                part2[i][threadIdx.x] = subc_cc(part2[i][threadIdx.x], part3[i][threadIdx.x]);
            }
            borrow_0 = -subc(0, 0);

            sub_cc(0, borrow_1);
            for (int i = 0; i < L_BLOCK; i ++){
                part2[i][threadIdx.x] = subc_cc(part2[i][threadIdx.x], part4[i][threadIdx.x]);
            }
            borrow_1 = -subc(0, 0);

            add_cc(carry, 0xFFFFFFFF);
            for (int i = 0; i < L_BLOCK; i ++){
                part1[i][threadIdx.x] = addc_cc(part1[i][threadIdx.x], part2[i][threadIdx.x]);
            }
            carry = addc(0, 0);
            __syncthreads();
            for (int j = j0 + threadIdx.x; j < j0 + L_BLOCK && j < L * 2; j += BLOCK_SIZE){
                for (int i = 0; i < batch_len; i ++){
                    ret[(idx0 + i) * (L * 2) + j] = part1[j - j0][i];
                }
            }
            __syncthreads();
        }
    }
}

template<int L_BLOCK, int BLOCK_SIZE>
__global__ void batch_mul_toom22_directlv1_kernel(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L_total){
    int L_split = (L_total + 1) >> 1;
    int L = L_split + 1;
    L += -L & 3;
    __shared__ uint32_t a[BATCH_MUL_DIRECT_L_MAX][BLOCK_SIZE + 1];
    __shared__ uint32_t b[BATCH_MUL_DIRECT_L_MAX][BLOCK_SIZE + 1], outer_carry[BLOCK_SIZE + 1];
    __shared__ uint32_t r[L_BLOCK * 2][BLOCK_SIZE + 1];
    ret += blockIdx.y * N * L * 2;
    for (int idx0 = blockIdx.x * BLOCK_SIZE; idx0 < N; idx0 += gridDim.x * BLOCK_SIZE){
        int batch_len = min(BLOCK_SIZE, N - idx0);
        uint32_t * src_a, * src_b;
        int src_l;
        if (blockIdx.y == 0 || blockIdx.y == 2){
            src_a = A + L_split;
            src_b = B + L_split;
            src_l = L_total - L_split;
        }else{
            src_a = A;
            src_b = B;
            src_l = L_split;
        }

        if (batch_len == BLOCK_SIZE){
            for (int j = 0; j < L; j += BLOCK_SIZE){
                if (j + threadIdx.x < src_l){
                    for (int i = 0; i < BLOCK_SIZE; i ++){                
                        a[j + threadIdx.x][i] = src_a[(idx0 + i) * L_total + j + threadIdx.x];
                        b[j + threadIdx.x][i] = src_b[(idx0 + i) * L_total + j + threadIdx.x];
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
                if (j + threadIdx.x < src_l){
                    for (int i = 0; i < batch_len; i ++){                
                        a[j + threadIdx.x][i] = src_a[(idx0 + i) * L_total + j + threadIdx.x];
                        b[j + threadIdx.x][i] = src_b[(idx0 + i) * L_total + j + threadIdx.x];
                    }
                }else{
                    for (int i = 0; i < batch_len; i ++){                
                        a[j + threadIdx.x][i] = 0;
                        b[j + threadIdx.x][i] = 0;
                    }
                }
            }
        }
        __syncthreads();
        if (blockIdx.y == 2){
            uint32_t carrya = 0, carryb = 0;
            src_a = A;
            src_b = B;
            src_l = L_split;
            for (int j = 0; j < L; j += BLOCK_SIZE){
                if (j + threadIdx.x < src_l){
                    for (int i = 0; i < batch_len; i ++){
                        r[threadIdx.x][i] = src_a[(idx0 + i) * L_total + j + threadIdx.x];
                    }
                }else{
                    for (int i = 0; i < batch_len; i ++){
                        r[threadIdx.x][i] = 0;
                    }
                }
                __syncthreads();
                add_cc(carrya, 0xFFFFFFFF);
                for (int j1 = 0; j1 < BLOCK_SIZE; j1 ++){
                    a[j + j1][threadIdx.x] = addc_cc(a[j + j1][threadIdx.x], r[j1][threadIdx.x]);
                }
                carrya = addc(0, 0);
                __syncthreads();
                if (j + threadIdx.x < src_l){
                    for (int i = 0; i < batch_len; i ++){
                        r[threadIdx.x][i] = src_b[(idx0 + i) * L_total + j + threadIdx.x];
                    }
                }else{
                    for (int i = 0; i < batch_len; i ++){
                        r[threadIdx.x][i] = 0;
                    }
                }
                __syncthreads();
                add_cc(carryb, 0xFFFFFFFF);
                for (int j1 = 0; j1 < BLOCK_SIZE; j1 ++){
                    b[j + j1][threadIdx.x] = addc_cc(b[j + j1][threadIdx.x], r[j1][threadIdx.x]);
                }
                carryb = addc(0, 0);
                __syncthreads();
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

// Forward declaration
static void batch_mul_toom22_internal(uint32_t * A, uint32_t * B, uint32_t * ret, 
    uint32_t * workspace, int N, int L);

// Internal recursive function
static void batch_mul_toom22_internal(uint32_t * A, uint32_t * B, uint32_t * ret, 
    uint32_t * workspace, int N, int L) {
        
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        batch_mul_direct(A, B, ret, N, L);
        return;
    }
    
    int L_split = ceil_div(L, 2);  // Actual split point = ceil(L/2)
    int L_half = L_split + 1;       // Padded buffer size
    
    // When about to use direct method, round L_half up to multiple of 4
    // This ensures better memory alignment for the direct multiplication kernel
    if (L_half <= BATCH_MUL_DIRECT_L_MAX) {
        L_half = (L_half + 3) & ~3;  // Round up to next multiple of 4
    }
    
    int c_size = L_half * 2;

    uint32_t * C_combined;
    
    if (L_half > BATCH_MUL_DIRECT_L_MAX){
        uint32_t * A_combined = workspace;
        uint32_t * B_combined = A_combined + (size_t)3 * N * L_half;
        C_combined = B_combined + (size_t)3 * N * L_half;
        uint32_t * next_workspace = C_combined + (size_t)3 * N * c_size;

        int num_blocks = (N + 32 - 1) / 32;
        if (num_blocks > 170 * 8) num_blocks = 170 * 8;
        batch_mul_toom22_transform_kernel<32, 32><<<dim3(num_blocks, 2, 1), 32>>>(
            A, B, A_combined, B_combined, N, L, L_split, L_half
        );
        
        // Single recursive call with 3N instances
        batch_mul_toom22_internal(A_combined, B_combined, C_combined, next_workspace, 3 * N, L_half);        
    }else{
        C_combined = workspace;
        
        batch_mul_toom22_directlv1_kernel<16, 32><<<dim3((N + 32 - 1) / 32, 3, 1), 32>>>(
            A, B, C_combined, N, L
        );
    }

    int num_blocks_2 = (N + 16 - 1) / 16;
    if (num_blocks_2 > 170 * 16) num_blocks_2 = 170 * 16;
    batch_mul_toom22_reconstruct_kernel<16, 16><<<num_blocks_2, 16>>>(
         ret, C_combined, N, L, L_split, L_half
    );
}

// Compute total workspace size recursively for internal use
static size_t workspace_size_words_internal(int N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    int L_split = ceil_div(L, 2);
    int L_half = L_split + 1;
    // Round up to multiple of 4 when about to use direct method
    if (L_half <= BATCH_MUL_DIRECT_L_MAX) {
        L_half = (L_half + 3) & ~3;
    }
    int c_size = L_half * 2;

    size_t total = 0;

    if (L_half > BATCH_MUL_DIRECT_L_MAX) {
         total += (size_t)3 * N * L_half * 2; // A_combined + B_combined
    }
    total += (size_t)3 * N * c_size; // C_combined
    
    total += workspace_size_words_internal(3 * N, L_half);
    return total;
}

static int get_N_max(int L){
    if (L > 250) {
        return 170 * 128 * 1;
    } else if (L > 126) {
        return 170 * 128 * 2;
    } else {
        return 170 * 128 * 2;
    }
}

size_t batch_mul_toom22_workspace_size(int N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    int N_max = get_N_max(L);
    
    int chunk_N = (N < N_max) ? N : N_max;
    
    return workspace_size_words_internal(chunk_N, L) * sizeof(uint32_t);
}

void batch_mul_toom22(uint32_t * A, uint32_t * B, uint32_t * ret, uint32_t * workspace, int N, int L) {
    if (L > BATCH_MUL_TOOM22_L_MAX) {
        return;
    }
    
    int N_max = get_N_max(L);

    for (int offset = 0; offset < N; offset += N_max) {
        int chunk_N = (offset + N_max <= N) ? N_max : (N - offset);
        
        uint32_t * A_chunk = A + (size_t)offset * L;
        uint32_t * B_chunk = B + (size_t)offset * L;
        uint32_t * ret_chunk = ret + (size_t)offset * (L * 2);
        batch_mul_toom22_internal(A_chunk, B_chunk, ret_chunk, workspace, chunk_N, L);
    }
}
