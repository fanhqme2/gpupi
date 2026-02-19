Implementation of Toom-Cook 2-way multiplication algorithm.

# Input and Output

Takes two array of large integers A, B as input. Each array contains N numbers stored continously as L uint32_t words.
The output array is of size N * (2 * L) words.
Also, a workspace pointer is needed.
A dedicated function is needed to compute the workspace size.

# File Structures

batch_mul_toom22.h                  header, defines the entry function
batch_mul_toom22.cu                 implementation code
batch_mul_toom22_test.cu            test case program
batch_mul_toom22_test_compile.sh    compilation command

# The Algorithm

We can always assume that L <= BATCH_MUL_TOOM22_L_MAX
Given N and L, we first check if L <= BATCH_MUL_DIRECT_L_MAX. If so, call the direct method batch_mul_direct defined in batch_mul_direct.h and implemented in batch_mul_direct.cu.
We don't need any workspace in this case.
Otherwise, a recursive algorithm is used.

We first determine L_split = ceil(L/2) and L_half = L_split + 1 = ceil(L/2) + 1. 
The reduction produces 3N instances of L_half sized multiplication.
To do the reduction, for each number a in A, we decompose a into two halfs, a0 (higher) and a1 (lower). Similiarly, we decompose b into b0 and b1. Then we have
c0 = a0 * b0
c1 = a1 * b1
c2 = (a0 + a1) * (b0 + b1)
L_half is chosen so that both (a0+a1) and (b0+b1) fits into L_half words (L_split + 1 = ceil(L/2) + 1).

Additionally, when L_half would be <= BATCH_MUL_DIRECT_L_MAX (i.e., when the next recursive level will use the direct method), L_half is rounded up to the next multiple of 4. This ensures better memory alignment and improves performance of the direct multiplication kernel. The actual split point L_split remains unchanged - only the buffer size L_half is adjusted.
The arrays of the reduced operands and results should be allocated in the workspace (incrementing the workspace pointer before passing to subsequent recursive calls).

This reduction should be done by a kernel batch_mul_toom22_transform_kernel that:
1. copies and pads a0, b0, a1, b1 to L_half length
2. computes a0 + a1, b0 + b1.
Each thread works on a different problem instance, and the addition is done serially by one thread so that we can use the add-with-carry instruction for efficient carry propagation.

The transform produces interleaved output arrays of size 3N * L_half:
- A_combined = [a0_0, ..., a0_{N-1}, a1_0, ..., a1_{N-1}, a_sum_0, ..., a_sum_{N-1}]
- B_combined = [b0_0, ..., b0_{N-1}, b1_0, ..., b1_{N-1}, b_sum_0, ..., b_sum_{N-1}]

Then we recursively call the function once with 3N instances to get the C array:
- C_combined = [c0_0, ..., c0_{N-1}, c1_0, ..., c1_{N-1}, c2_0, ..., c2_{N-1}]
where each cX_i has 2*L_half words.

Finally, we recover the results by:
a * b = ((c0 << 2*L_split) + c1) + ((c2 - c0 - c1) << L_split)
Note: the shift amount is L_split = ceil(L/2), not L_half.
The recovery should be done by a kernel called batch_mul_toom22_reconstruct_kernel.
It computes ((c0 << 2*L_split) + c1) by copying c0 and c1 into the result array (but notice that we only copy 2*L - 2*L_split words from c0. This is safe as c0 would have at least two leading zero words).
Then c2 is subtracted by c0, and then c1. Use subtract-with-borrow instruction for efficient propagation.
Finally the subtracted c2 is added to the result array at offset L_split.
Again, each thread works on a different problem instance so that synchronization is trivial.

# Block Size and Parallization

Because we don't want to use too much workspace, for large N, we break them into sequential processing of N_max instances at once.
N_max is deterimined as follows:
- if L > 250, N_max = 170 * 128
- otherwise, if L > 126, N_max = 170 * 128 * 2
- othrewise, N_max = 170 * 128 * 4
The number of threads per block should be a tunnable constant, which defaults to 16.

# Testing

The testing method should be similiar to batch_mul_direct_test.cu but with the following modifications:
- L should be between BATCH_MUL_DIRECT_L_MAX+1 and BATCH_MUL_TOOM22_L_MAX
- For small size, N*L <= 16384
- Benchmarking should goes for L = 481, 241, 121, 65 with N*L <= 1e8

The compilation script should compile the test program (using necessary .cu files as input, link against GMP library), and look similiar to batch_mul_direct_test_compile.sh.
