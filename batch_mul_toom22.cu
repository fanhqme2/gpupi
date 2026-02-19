#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include "batch_mul_toom22.h"
#include "batch_mul_direct.h"

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
__global__ void batch_mul_toom22_transform_kernel(
    const uint32_t * A, const uint32_t * B,
    uint32_t * A_out, uint32_t * B_out,
    int N, int L, int L_split, int L_half
) {
    // L_split = ceil(L/2)
    // L_half = L_split + 1 (padded size)
    int L_lower = L_split;  // = ceil(L/2)
    int L_upper = L / 2;    // = floor(L/2)
    const uint32_t * AB;
    uint32_t * AB_out;
    if (blockIdx.y == 0) {
        AB = A;
        AB_out = A_out;
    } else {
        AB = B;
        AB_out = B_out;
    }
    for (int idx0 = blockIdx.x * blockDim.x; idx0 < N; idx0 += blockDim.x * gridDim.x) {
        for (int idx = idx0; idx < N && idx < idx0 + blockDim.x; idx++){
            const uint32_t * a = AB + idx * L;
            uint32_t * a0 = AB_out + idx * L_half;
            uint32_t * a1 = AB_out + (N + idx) * L_half;
            for (int i = threadIdx.x; i < L_half; i += blockDim.x){
                if (i < L_lower) {
                    a1[i] = a[i];
                } else {
                    a1[i] = 0;
                }
            }
            for (int i = threadIdx.x; i < L_half; i += blockDim.x){    
                if (i < L_upper) {
                    a0[i] = a[L_lower + i];
                } else {
                    a0[i] = 0;
                }
            }
        }
    }
    __syncthreads();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t * a0 = AB_out + idx * L_half;
        uint32_t * a1 = AB_out + (N + idx) * L_half;
        uint32_t * a_sum = AB_out + (2 * N + idx) * L_half;

        // Compute a0 + a1 and b0 + b1 with carry propagation

        a_sum[0] = add_cc(a0[0], a1[0]);
        for (int i = 1; i < L_half; i ++){
            a_sum[i] = addc_cc(a0[i], a1[i]);
        }
    }
}

// Reconstruct kernel: combines partial results
// L_split = ceil(L/2) is the actual split point (shift amount)
// L_half = L_split + 1 is the buffer size
__global__ void batch_mul_toom22_reconstruct_kernel(
    uint32_t * ret,
    const uint32_t * C,
    int N, int L, int L_split, int L_half
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t * r = ret + idx * (L * 2);
        int c_size = L_half * 2;
        
        const uint32_t * c0 = C + idx * c_size;
        const uint32_t * c1 = C + (N + idx) * c_size;
        const uint32_t * c2 = C + (2 * N + idx) * c_size;
        
        int result_size = L * 2;
        
        // Compute c2' = c2 - c0 - c1
        uint32_t c2_adjusted[BATCH_MUL_TOOM22_L_MAX + 2];
        
        for (int i = 0; i < c_size; i++) {
            c2_adjusted[i] = c2[i];
        }
        
        c2_adjusted[0] = sub_cc(c2_adjusted[0], c0[0]);
        for (int i = 1; i < c_size; i++) {
            c2_adjusted[i] = subc_cc(c2_adjusted[i], c0[i]);
        }
        
        c2_adjusted[0] = sub_cc(c2_adjusted[0], c1[0]);
        for (int i = 1; i < c_size; i++) {
            c2_adjusted[i] = subc_cc(c2_adjusted[i], c1[i]);
        }

        for (int i = 0; i < L_split * 2; i ++){
            r[i] = c1[i];
        }
        for (int i = L_split * 2; i < result_size; i ++){
            r[i] = c0[i - L_split * 2];
        }

        r[L_split] = add_cc(r[L_split], c2_adjusted[0]);
        for (int i = L_split + 1; i < L_split + c_size; i++) {
            r[i] = addc_cc(r[i], c2_adjusted[i - L_split]);
        }
        for (int i = L_split + c_size; i < result_size; i++) {
            r[i] = addc_cc(r[i], 0);
        }
    }
}

// Forward declaration
static void batch_mul_toom22_internal(uint32_t * A, uint32_t * B, uint32_t * ret, 
    uint32_t * workspace, int N, int L);

// Internal recursive function
static void batch_mul_toom22_internal(uint32_t * A, uint32_t * B, uint32_t * ret, 
    uint32_t * workspace, int N, int L) {
    
    const int threads_per_block = BATCH_MUL_TOOM22_THREADS_PER_BLOCK;
    
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
    
    uint32_t * A_combined = workspace;
    uint32_t * B_combined = A_combined + (size_t)3 * N * L_half;
    uint32_t * C_combined = B_combined + (size_t)3 * N * L_half;
    uint32_t * next_workspace = C_combined + (size_t)3 * N * c_size;
    
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 170 * 8) num_blocks = 170 * 8;
    
    batch_mul_toom22_transform_kernel<<<dim3(num_blocks, 2, 1), threads_per_block>>>(
        A, B, A_combined, B_combined, N, L, L_split, L_half
    );
    
    // Single recursive call with 3N instances
    batch_mul_toom22_internal(A_combined, B_combined, C_combined, next_workspace, 3 * N, L_half);
    
    batch_mul_toom22_reconstruct_kernel<<<num_blocks, threads_per_block>>>(
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
    
    size_t current = (size_t)3 * N * L_half + (size_t)3 * N * L_half + (size_t)3 * N * c_size;
    size_t recursive = workspace_size_words_internal(3 * N, L_half);
    
    return current + recursive;
}

// Compute workspace size for a chunk of size chunk_N (used at top level)
static size_t workspace_size_words_chunk(int chunk_N, int L) {
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
    
    size_t current = (size_t)3 * chunk_N * L_half + (size_t)3 * chunk_N * L_half + (size_t)3 * chunk_N * c_size;
    size_t recursive = workspace_size_words_internal(3 * chunk_N, L_half);
    
    return current + recursive;
}

size_t batch_mul_toom22_workspace_size(int N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    int N_max;
    if (L > 250) {
        N_max = 170 * 128;
    } else if (L > 126) {
        N_max = 170 * 128 * 2;
    } else {
        N_max = 170 * 128 * 4;
    }
    
    int chunk_N = (N < N_max) ? N : N_max;
    
    return workspace_size_words_chunk(chunk_N, L) * sizeof(uint32_t);
}

void batch_mul_toom22(uint32_t * A, uint32_t * B, uint32_t * ret, uint32_t * workspace, int N, int L) {
    if (L > BATCH_MUL_TOOM22_L_MAX) {
        return;
    }
    
    int N_max;
    if (L > 250) {
        N_max = 170 * 128;
    } else if (L > 126) {
        N_max = 170 * 128 * 2;
    } else {
        N_max = 170 * 128 * 4;
    }

    for (int offset = 0; offset < N; offset += N_max) {
        int chunk_N = (offset + N_max <= N) ? N_max : (N - offset);
        
        uint32_t * A_chunk = A + (size_t)offset * L;
        uint32_t * B_chunk = B + (size_t)offset * L;
        uint32_t * ret_chunk = ret + (size_t)offset * (L * 2);
        batch_mul_toom22_internal(A_chunk, B_chunk, ret_chunk, workspace, chunk_N, L);
    }
}
