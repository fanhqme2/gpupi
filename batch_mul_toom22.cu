#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include "batch_mul_toom22.h"
#include "batch_mul_direct.h"

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Transform kernel: decomposes inputs and prepares 3N interleaved arrays
// L_split = ceil(L/2) is the actual split point
// L_half = L_split + 1 is the buffer size for padded storage
__global__ void batch_mul_toom22_transform_kernel(
    const uint32_t * A, const uint32_t * B,
    uint32_t * A_out, uint32_t * B_out,
    int N, int L, int L_split, int L_half
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        const uint32_t * a = A + idx * L;
        const uint32_t * b = B + idx * L;
        
        uint32_t * a0 = A_out + idx * L_half;
        uint32_t * a1 = A_out + (N + idx) * L_half;
        uint32_t * a_sum = A_out + (2 * N + idx) * L_half;
        uint32_t * b0 = B_out + idx * L_half;
        uint32_t * b1 = B_out + (N + idx) * L_half;
        uint32_t * b_sum = B_out + (2 * N + idx) * L_half;
        
        // L_split = ceil(L/2)
        // L_half = L_split + 1 (padded size)
        int L_lower = L_split;  // = ceil(L/2)
        int L_upper = L / 2;    // = floor(L/2)
        
        // Copy a1 (lower L_lower words), pad with zeros to L_half
        for (int i = 0; i < L_lower; i++) {
            a1[i] = a[i];
            b1[i] = b[i];
        }
        for (int i = L_lower; i < L_half; i++) {
            a1[i] = 0;
            b1[i] = 0;
        }
        
        // Copy a0 (upper L_upper words), pad with zeros to L_half
        for (int i = 0; i < L_upper; i++) {
            a0[i] = a[L_lower + i];
            b0[i] = b[L_lower + i];
        }
        for (int i = L_upper; i < L_half; i++) {
            a0[i] = 0;
            b0[i] = 0;
        }
        
        // Compute a0 + a1 and b0 + b1 with carry propagation
        uint32_t carry_a = 0;
        uint32_t carry_b = 0;
        for (int i = 0; i < L_half; i++) {
            uint64_t sum_a = (uint64_t)a0[i] + (uint64_t)a1[i] + carry_a;
            a_sum[i] = (uint32_t)sum_a;
            carry_a = (uint32_t)(sum_a >> 32);
            
            uint64_t sum_b = (uint64_t)b0[i] + (uint64_t)b1[i] + carry_b;
            b_sum[i] = (uint32_t)sum_b;
            carry_b = (uint32_t)(sum_b >> 32);
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
        uint32_t c2_adjusted[514]; // Max c_size for L=512: (257)*2 = 514
        
        for (int i = 0; i < c_size; i++) {
            c2_adjusted[i] = c2[i];
        }
        
        uint32_t borrow = 0;
        for (int i = 0; i < c_size; i++) {
            uint64_t diff = (uint64_t)c2_adjusted[i] - (uint64_t)c0[i] - borrow;
            c2_adjusted[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
        
        borrow = 0;
        for (int i = 0; i < c_size; i++) {
            uint64_t diff = (uint64_t)c2_adjusted[i] - (uint64_t)c1[i] - borrow;
            c2_adjusted[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
        
        // Initialize result to zero
        for (int i = 0; i < result_size; i++) {
            r[i] = 0;
        }
        
        // Add c1 to result[0 : c_size-1]
        int words_to_copy = (c_size < result_size) ? c_size : result_size;
        for (int i = 0; i < words_to_copy; i++) {
            r[i] = c1[i];
        }
        
        // Add c2_adjusted to result at offset L_split (NOT L_half)
        uint32_t carry = 0;
        for (int i = 0; i < c_size && (L_split + i) < result_size; i++) {
            uint64_t sum = (uint64_t)r[L_split + i] + (uint64_t)c2_adjusted[i] + carry;
            r[L_split + i] = (uint32_t)sum;
            carry = (uint32_t)(sum >> 32);
        }
        int pos = L_split + c_size;
        while (carry && pos < result_size) {
            uint64_t sum = (uint64_t)r[pos] + carry;
            r[pos] = (uint32_t)sum;
            carry = (uint32_t)(sum >> 32);
            pos++;
        }
        
        // Add c0 to result at offset 2*L_split (NOT 2*L_half)
        int c0_offset = 2 * L_split;
        int c0_words = result_size - c0_offset;
        if (c0_words > c_size) c0_words = c_size;
        if (c0_words > 0) {
            carry = 0;
            for (int i = 0; i < c0_words; i++) {
                uint64_t sum = (uint64_t)r[c0_offset + i] + (uint64_t)c0[i] + carry;
                r[c0_offset + i] = (uint32_t)sum;
                carry = (uint32_t)(sum >> 32);
            }
            pos = c0_offset + c0_words;
            while (carry && pos < result_size) {
                uint64_t sum = (uint64_t)r[pos] + carry;
                r[pos] = (uint32_t)sum;
                carry = (uint32_t)(sum >> 32);
                pos++;
            }
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
    int c_size = L_half * 2;
    
    uint32_t * A_combined = workspace;
    uint32_t * B_combined = A_combined + (size_t)3 * N * L_half;
    uint32_t * C_combined = B_combined + (size_t)3 * N * L_half;
    uint32_t * next_workspace = C_combined + (size_t)3 * N * c_size;
    
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 170 * 16) num_blocks = 170 * 16;
    
    batch_mul_toom22_transform_kernel<<<num_blocks, threads_per_block>>>(
        A, B, A_combined, B_combined, N, L, L_split, L_half
    );
    cudaDeviceSynchronize();
    
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
    
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        batch_mul_direct(A, B, ret, N, L);
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
    
    const int threads_per_block = BATCH_MUL_TOOM22_THREADS_PER_BLOCK;
    
    for (int offset = 0; offset < N; offset += N_max) {
        int chunk_N = (offset + N_max <= N) ? N_max : (N - offset);
        
        uint32_t * A_chunk = A + (size_t)offset * L;
        uint32_t * B_chunk = B + (size_t)offset * L;
        uint32_t * ret_chunk = ret + (size_t)offset * (L * 2);
        
        if (chunk_N == N && offset == 0) {
            batch_mul_toom22_internal(A, B, ret, workspace, N, L);
        } else {
            int L_split = ceil_div(L, 2);
            int L_half = L_split + 1;
            int c_size = L_half * 2;
            
            uint32_t * A_combined = workspace;
            uint32_t * B_combined = A_combined + (size_t)3 * chunk_N * L_half;
            uint32_t * C_combined = B_combined + (size_t)3 * chunk_N * L_half;
            uint32_t * next_workspace = C_combined + (size_t)3 * chunk_N * c_size;
            
            int num_blocks = (chunk_N + threads_per_block - 1) / threads_per_block;
            if (num_blocks > 170 * 16) num_blocks = 170 * 16;
            
            batch_mul_toom22_transform_kernel<<<num_blocks, threads_per_block>>>(
                A_chunk, B_chunk, A_combined, B_combined, chunk_N, L, L_split, L_half
            );
            
            batch_mul_toom22_internal(A_combined, B_combined, C_combined, next_workspace, 3 * chunk_N, L_half);
            
            batch_mul_toom22_reconstruct_kernel<<<num_blocks, threads_per_block>>>(
                ret_chunk, C_combined, chunk_N, L, L_split, L_half
            );
        }
    }
}
