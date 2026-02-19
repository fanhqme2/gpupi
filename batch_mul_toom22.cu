#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "batch_mul_toom22.h"
#include "batch_mul_direct.h"

// Helper to compute ceil(a / b) for positive integers
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Transform kernel: decomposes inputs and prepares 3N interleaved arrays
// Output layout: [a0_0, a0_1, ..., a0_{N-1}, a1_0, ..., a1_{N-1}, a_sum_0, ..., a_sum_{N-1}]
//                [b0_0, b0_1, ..., b0_{N-1}, b1_0, ..., b1_{N-1}, b_sum_0, ..., b_sum_{N-1}]
// Each thread handles one instance
__global__ void batch_mul_toom22_transform_kernel(
    const uint32_t * A, const uint32_t * B,
    uint32_t * A_out, uint32_t * B_out,
    int N, int L, int L_half
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        const uint32_t * a = A + idx * L;
        const uint32_t * b = B + idx * L;
        
        // Output pointers for this instance
        // a0 at position idx, a1 at position N + idx, a_sum at position 2N + idx
        uint32_t * a0 = A_out + idx * L_half;
        uint32_t * a1 = A_out + (N + idx) * L_half;
        uint32_t * a_sum = A_out + (2 * N + idx) * L_half;
        uint32_t * b0 = B_out + idx * L_half;
        uint32_t * b1 = B_out + (N + idx) * L_half;
        uint32_t * b_sum = B_out + (2 * N + idx) * L_half;
        
        // L_half = ceil(L/2) + 1
        // Lower half size: ceil(L/2)
        int L_lower = ceil_div(L, 2);
        int L_upper = L / 2;  // floor(L/2)
        
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
        // a0 starts at offset L_lower
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
        // carry is discarded - L_half is chosen so that sum fits
    }
}

// Reconstruct kernel: combines partial results to produce final output
// Input C layout: [c0_0, ..., c0_{N-1}, c1_0, ..., c1_{N-1}, c2_0, ..., c2_{N-1}]
// where each cX_i is 2*L_half words
// result = ((c0 << L_half*2) + c1) + ((c2 - c0 - c1) << L_half)
__global__ void batch_mul_toom22_reconstruct_kernel(
    uint32_t * ret,
    const uint32_t * C,
    int N, int L, int L_half
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t * r = ret + idx * (L * 2);
        int c_size = L_half * 2;
        
        const uint32_t * c0 = C + idx * c_size;
        const uint32_t * c1 = C + (N + idx) * c_size;
        const uint32_t * c2 = C + (2 * N + idx) * c_size;
        
        int result_size = L * 2;
        
        // First, compute c2' = c2 - c0 - c1 using subtract-with-borrow
        // Store in local array
        uint32_t c2_adjusted[514]; // Max c_size for L=512: (257)*2 = 514
        
        // Copy c2 to local array
        for (int i = 0; i < c_size; i++) {
            c2_adjusted[i] = c2[i];
        }
        
        // Subtract c0 from c2 (c2 = c2 - c0)
        uint32_t borrow = 0;
        for (int i = 0; i < c_size; i++) {
            uint64_t diff = (uint64_t)c2_adjusted[i] - (uint64_t)c0[i] - borrow;
            c2_adjusted[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
        
        // Subtract c1 from c2 (c2 = c2 - c1)
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
        
        // Copy c1 to result[0 : 2*L_half-1] (lower part)
        int c1_words = (c_size < result_size) ? c_size : result_size;
        for (int i = 0; i < c1_words; i++) {
            r[i] = c1[i];
        }
        
        // Copy c0 to result[L_half*2 : L_half*2 + 2*L_half-1] (upper part)
        // But we only copy 2*L - 2*L_half words from c0 (as per spec)
        int c0_offset = L_half * 2;
        int c0_words_to_copy = result_size - c0_offset;
        if (c0_words_to_copy > c_size) c0_words_to_copy = c_size;
        if (c0_words_to_copy > 0) {
            for (int i = 0; i < c0_words_to_copy; i++) {
                r[c0_offset + i] = c0[i];
            }
        }
        
        // Add adjusted c2 to result at offset L_half
        uint32_t carry = 0;
        for (int i = 0; i < c_size && (L_half + i) < result_size; i++) {
            uint64_t sum = (uint64_t)r[L_half + i] + (uint64_t)c2_adjusted[i] + carry;
            r[L_half + i] = (uint32_t)sum;
            carry = (uint32_t)(sum >> 32);
        }
        // Propagate carry
        int pos = L_half + c_size;
        while (carry && pos < result_size) {
            uint64_t sum = (uint64_t)r[pos] + carry;
            r[pos] = (uint32_t)sum;
            carry = (uint32_t)(sum >> 32);
            pos++;
        }
    }
}

// Forward declaration for recursive call
static void batch_mul_toom22_internal(uint32_t * A, uint32_t * B, uint32_t * ret, 
    uint32_t * workspace, int N, int L);

// Internal recursive function - no N_max chunking, goes directly to algorithm
static void batch_mul_toom22_internal(uint32_t * A, uint32_t * B, uint32_t * ret, 
    uint32_t * workspace, int N, int L) {
    
    const int threads_per_block = BATCH_MUL_TOOM22_THREADS_PER_BLOCK;
    
    // Base case: use direct multiplication
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        int num_blocks = (N + threads_per_block - 1) / threads_per_block;
        if (num_blocks >= 170 * 16) {
            num_blocks = 170 * 16;
        }
        batch_mul_direct(A, B, ret, N, L);
        return;
    }
    
    // Recursive case: Toom-Cook 2-way
    int L_half = ceil_div(L, 2) + 1;
    int c_size = L_half * 2;
    
    // Workspace layout:
    // [A_combined][B_combined][C_combined]
    // A_combined: 3N * L_half words (a0, a1, a_sum interleaved)
    // B_combined: 3N * L_half words (b0, b1, b_sum interleaved)
    // C_combined: 3N * c_size words (c0, c1, c2 interleaved)
    // Then recursive workspace follows
    
    uint32_t * A_combined = workspace;
    uint32_t * B_combined = A_combined + (size_t)3 * N * L_half;
    uint32_t * C_combined = B_combined + (size_t)3 * N * L_half;
    uint32_t * next_workspace = C_combined + (size_t)3 * N * c_size;
    
    // Launch transform kernel
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 170 * 16) num_blocks = 170 * 16;
    
    batch_mul_toom22_transform_kernel<<<num_blocks, threads_per_block>>>(
        A, B, A_combined, B_combined, N, L, L_half
    );
    
    // Single recursive call with 3N instances
    batch_mul_toom22_internal(A_combined, B_combined, C_combined, next_workspace, 3 * N, L_half);
    
    // Launch reconstruct kernel
    batch_mul_toom22_reconstruct_kernel<<<num_blocks, threads_per_block>>>(
        ret, C_combined, N, L, L_half
    );
}

// Compute total workspace size recursively for internal use
// This is for the full N (no chunking)
static size_t workspace_size_words_internal(int N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    int L_half = ceil_div(L, 2) + 1;
    int c_size = L_half * 2;
    
    // Current level: A_combined + B_combined + C_combined
    size_t current = (size_t)3 * N * L_half + (size_t)3 * N * L_half + (size_t)3 * N * c_size;
    
    // Recursive workspace (for 3N instances at L_half)
    size_t recursive = workspace_size_words_internal(3 * N, L_half);
    
    return current + recursive;
}

// Compute workspace size for a chunk of size chunk_N (used at top level)
static size_t workspace_size_words_chunk(int chunk_N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    int L_half = ceil_div(L, 2) + 1;
    int c_size = L_half * 2;
    
    // Current level workspace for this chunk
    size_t current = (size_t)3 * chunk_N * L_half + (size_t)3 * chunk_N * L_half + (size_t)3 * chunk_N * c_size;
    
    // Recursive workspace (for 3*chunk_N instances at L_half)
    size_t recursive = workspace_size_words_internal(3 * chunk_N, L_half);
    
    return current + recursive;
}

size_t batch_mul_toom22_workspace_size(int N, int L) {
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        return 0;
    }
    
    // Determine N_max based on L
    int N_max;
    if (L > 250) {
        N_max = 170 * 128;
    } else if (L > 126) {
        N_max = 170 * 128 * 2;
    } else {
        N_max = 170 * 128 * 4;
    }
    
    // Use the actual chunk size we'll process
    int chunk_N = (N < N_max) ? N : N_max;
    
    return workspace_size_words_chunk(chunk_N, L) * sizeof(uint32_t);
}

void batch_mul_toom22(uint32_t * A, uint32_t * B, uint32_t * ret, uint32_t * workspace, int N, int L) {
    // Validate L
    if (L > BATCH_MUL_TOOM22_L_MAX) {
        return;
    }
    
    // Base case: use direct multiplication
    if (L <= BATCH_MUL_DIRECT_L_MAX) {
        batch_mul_direct(A, B, ret, N, L);
        return;
    }
    
    // Determine N_max based on L (top-level chunking only)
    int N_max;
    if (L > 250) {
        N_max = 170 * 128;
    } else if (L > 126) {
        N_max = 170 * 128 * 2;
    } else {
        N_max = 170 * 128 * 4;
    }
    
    const int threads_per_block = BATCH_MUL_TOOM22_THREADS_PER_BLOCK;
    
    // Process in chunks of N_max to limit workspace usage (only at top level)
    for (int offset = 0; offset < N; offset += N_max) {
        int chunk_N = (offset + N_max <= N) ? N_max : (N - offset);
        
        uint32_t * A_chunk = A + (size_t)offset * L;
        uint32_t * B_chunk = B + (size_t)offset * L;
        uint32_t * ret_chunk = ret + (size_t)offset * (L * 2);
        
        // For single chunk that fits in N_max, call internal directly
        if (chunk_N == N && offset == 0) {
            batch_mul_toom22_internal(A, B, ret, workspace, N, L);
        } else {
            // For chunked processing, we need to compute workspace per chunk
            // The provided workspace should be sized for N_max
            int L_half = ceil_div(L, 2) + 1;
            int c_size = L_half * 2;
            
            uint32_t * A_combined = workspace;
            uint32_t * B_combined = A_combined + (size_t)3 * chunk_N * L_half;
            uint32_t * C_combined = B_combined + (size_t)3 * chunk_N * L_half;
            uint32_t * next_workspace = C_combined + (size_t)3 * chunk_N * c_size;
            
            // Launch transform kernel
            int num_blocks = (chunk_N + threads_per_block - 1) / threads_per_block;
            if (num_blocks > 170 * 16) num_blocks = 170 * 16;
            
            batch_mul_toom22_transform_kernel<<<num_blocks, threads_per_block>>>(
                A_chunk, B_chunk, A_combined, B_combined, chunk_N, L, L_half
            );
            
            // Single call to internal with 3*chunk_N instances (no N_max inside)
            batch_mul_toom22_internal(A_combined, B_combined, C_combined, next_workspace, 3 * chunk_N, L_half);
            
            // Launch reconstruct kernel
            batch_mul_toom22_reconstruct_kernel<<<num_blocks, threads_per_block>>>(
                ret_chunk, C_combined, chunk_N, L, L_half
            );
        }
    }
}
