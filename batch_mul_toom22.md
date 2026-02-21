Implementation of Toom-Cook 2-way multiplication algorithm.

# Input and Output

Takes two arrays of large integers A, B as input. Each array contains N numbers stored continuously as L uint32_t words.
The output array is of size N * (2 * L) words.
Also, a workspace pointer is needed.
A dedicated function `batch_mul_toom22_workspace_size()` is needed to compute the workspace size.

# File Structures

batch_mul_toom22.h                  header, defines the entry function
batch_mul_toom22.cu                 implementation code
batch_mul_toom22_test.cu            test case program
batch_mul_toom22_test_compile.sh    compilation command

# The Algorithm

We can always assume that L <= BATCH_MUL_TOOM22_L_MAX
Given N and L, we first check if L <= BATCH_MUL_DIRECT_L_MAX. If so, call the direct method `batch_mul_direct` defined in batch_mul_direct.h and implemented in batch_mul_direct.cu.
We don't need any workspace in this case.
Otherwise, a recursive algorithm is used.

We first determine:
- `L_split = ceil(L/2)` - the actual split point for decomposing numbers
- `L_half = L_split + 1 = ceil(L/2) + 1` - buffer size for padded storage

The reduction produces 3N instances of L_half sized multiplication.
To do the reduction, for each number a in A, we decompose a into two halves, a0 (higher) and a1 (lower). Similarly, we decompose b into b0 and b1. Then we have:
- c0 = a0 * b0
- c1 = a1 * b1  
- c2 = (a0 + a1) * (b0 + b1)

L_half is chosen so that both (a0+a1) and (b0+b1) fit into L_half words (L_split + 1 = ceil(L/2) + 1).

Additionally, when L_half would be <= BATCH_MUL_DIRECT_L_MAX (i.e., when the next recursive level will use the direct method), L_half is rounded up to the next multiple of 4. This ensures better memory alignment and improves performance of the direct multiplication kernel. The actual split point L_split remains unchanged - only the buffer size L_half is adjusted.

The arrays of the reduced operands and results should be allocated in the workspace (incrementing the workspace pointer before passing to subsequent recursive calls).

# Low-level Arithmetic Primitives

The implementation uses inline PTX assembly for efficient carry/borrow propagation:

- `add_cc(a, b)` - Add with carry-out, sets carry flag
- `addc_cc(a, b)` - Add with carry-in and carry-out, uses and sets carry flag
- `addc(a, b)` - Add with carry-in, uses carry flag but no output carry
- `sub_cc(a, b)` - Subtract with borrow-out, sets carry flag
- `subc_cc(a, b)` - Subtract with borrow-in and borrow-out, uses and sets carry flag
- `subc(a, b)` - Subtract with borrow-in, uses carry flag but no output borrow

These primitives enable efficient multi-precision addition and subtraction with proper carry/borrow chain handling.

# Transform Kernel

The transform is done by `batch_mul_toom22_transform_kernel` that:
1. Copies and pads a0, a1 (or b0, b1) to L_half length
2. Computes a0 + a1, b0 + b1 using the add-with-carry primitives

The kernel uses a 2D grid where blockIdx.y selects between A (y=0) and B (y=1) arrays.
Shared memory is used to store decomposed parts and their sums:
- `a0[L_BLOCK][BLOCK_SIZE+1]` - higher half
- `a1[L_BLOCK][BLOCK_SIZE+1]` - lower half  
- `a_sum[L_BLOCK][BLOCK_SIZE+1]` - sum of halves

Each block processes BLOCK_SIZE problem instances in batch, working in chunks of L_BLOCK elements at a time for efficient memory access.

The transform produces interleaved output arrays of size 3N * L_half:
- A_combined = [a0_0, ..., a0_{N-1}, a1_0, ..., a1_{N-1}, a_sum_0, ..., a_sum_{N-1}]
- B_combined = [b0_0, ..., b0_{N-1}, b1_0, ..., b1_{N-1}, b_sum_0, ..., b_sum_{N-1}]

# Recursive Multiplication

After transformation, we recursively call `batch_mul_toom22_internal()` once with 3N instances to get the C array:
- C_combined = [c0_0, ..., c0_{N-1}, c1_0, ..., c1_{N-1}, c2_0, ..., c2_{N-1}]
where each cX_i has 2*L_half words.

# Reconstruct Kernel

The recovery is done by `batch_mul_toom22_reconstruct_kernel` which computes:
```
a * b = ((c0 << 2*L_split) + c1) + ((c2 - c0 - c1) << L_split)
```
Note: the shift amount is L_split = ceil(L/2), not L_half.

The kernel:
1. Computes `((c0 << 2*L_split) + c1)` by copying c0 and c1 into the result array (only copying 2*L - 2*L_split words from c0, which is safe as c0 has at least two leading zero words)
2. Subtracts c0 and c1 from c2 using subtract-with-borrow primitives for efficient propagation
3. Adds the subtracted result to the output at offset L_split

Shared memory arrays are used:
- `part1` - stores combined c0/c1 result
- `part2` - stores c2 and intermediate subtraction results
- `part3` - stores c0 for subtraction
- `part4` - stores c1 for subtraction

# Block Size and Parallelization

## Kernel Launch Configuration

Transform kernel:
- Grid: dim3(num_blocks, 2, 1) where num_blocks = min(ceil(N/32), 170*8)
- Block: 32 threads
- Template parameters: L_BLOCK=32, BLOCK_SIZE=32

Reconstruct kernel:
- Grid: num_blocks_2 = min(ceil(N/16), 170*16)
- Block: 16 threads
- Template parameters: L_BLOCK=16, BLOCK_SIZE=16

## Chunking for Large N

To limit workspace usage, for large N we break them into sequential processing of N_max instances at once.
N_max is determined as follows:
- if L > 250, N_max = 170 * 128
- otherwise, if L > 126, N_max = 170 * 128 * 2
- otherwise, N_max = 170 * 128 * 4

The main entry function `batch_mul_toom22()` iterates through chunks of size N_max, calling the internal recursive function for each chunk.

# Workspace Size Calculation

`batch_mul_toom22_workspace_size()` computes the total workspace needed recursively:
1. If L <= BATCH_MUL_DIRECT_L_MAX, returns 0
2. Otherwise, computes current level needs:
   - A_combined: 3*N*L_half words
   - B_combined: 3*N*L_half words
   - C_combined: 3*N*(L_half*2) words
3. Adds recursive workspace for 3*N instances of size L_half

The size is calculated for a single chunk (chunk_N = min(N, N_max)) and returned in bytes.

# Testing

The testing method should be similar to batch_mul_direct_test.cu but with the following modifications:
- L should be between BATCH_MUL_DIRECT_L_MAX+1 and BATCH_MUL_TOOM22_L_MAX
- For small size, N*L <= 16384
- Benchmarking should go for L = 481, 241, 121, 65 with N*L <= 1e8

The compilation script should compile the test program (using necessary .cu files as input, link against GMP library), and look similar to batch_mul_direct_test_compile.sh.
