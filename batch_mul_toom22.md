Implementation blueprint for `batch_mul_toom22.cu` (current version).

# High-Level Algorithm Overview

This implementation is based on Toom-Cook 2-way multiplication (Karatsuba).

For each large integer product `A * B`, it recursively splits numbers into two halves:

- `A = A0 * x + A1`
- `B = B0 * x + B1`

where `x = 2^(32 * L_split)` in word-space.

Instead of 4 half-size multiplications, it computes 3:

- `C0 = A0 * B0`
- `C1 = A1 * B1`
- `C2 = (A0 + A1) * (B0 + B1)`

Then it reconstructs:

`A * B = C0 * x^2 + (C2 - C0 - C1) * x + C1`

This reduces multiplication count asymptotically (vs. schoolbook) and is the core reason for the speedup. In this codebase, recursion is combined with two optimized non-recursive base kernels (`directlv1`, `directlv2`) to avoid overhead at smaller sizes.

# Interface

The entry point is:

`void batch_mul_toom22(uint32_t *A, uint32_t *B, uint32_t *ret, uint32_t *workspace, int N, int L)`

- `A`, `B`: `N` big integers, each with `L` words (`uint32_t`).
- `ret`: output buffer with `N * (2 * L)` words.
- `workspace`: scratch buffer (may be unused for small `L`).
- `batch_mul_toom22_workspace_size(N, L)` returns required bytes.

# File Layout

- `batch_mul_toom22.h`: public API
- `batch_mul_toom22.cu`: implementation
- `batch_mul_toom22_test.cu`: correctness + benchmark
- `batch_mul_toom22_test_compile.sh`: build script

# High-Level Dispatch

Given `(N, L)`:

1. If `L <= BATCH_MUL_DIRECT_L_MAX`, call `batch_mul_direct`.
2. Else compute:
   - `L_split = ceil(L / 2)`
   - `L_half = L_split + 1`
3. If `L_half <= BATCH_MUL_DIRECT_L_MAX`, run `batch_mul_toom22_directlv1_kernel<32>` with block `(32, 3, 1)`.
4. Else if `L <= BATCH_MUL_TOOM22_LV2_MAX` (`252`), run `batch_mul_toom22_directlv2_kernel<32>` with block `(32, 9, 1)`.
5. Else run recursive path:
   - transform (`A,B -> A_combined,B_combined`)
   - recursive multiply on `3N` instances of size `L_half`
   - reconstruct to size `2L`

`L_half` is rounded up to a multiple of 4 only when `L_half <= BATCH_MUL_DIRECT_L_MAX` (to help direct kernel alignment).

# Chunking Policy

`batch_mul_toom22()` processes input in chunks:

- If `L > BATCH_MUL_TOOM22_LV2_MAX`: `N_max = min(N, 170 * 128)`
- Else: `N_max = N` (no chunking needed; direct kernels need no workspace)

# Arithmetic Primitives

Carry/borrow helpers from `batch_mul_addsub_warp.h` are used heavily:

- `batch_mul_add_64_single_warp`
- `batch_mul_add_64_all_warp`
- `batch_mul_add_64_grouped_warp`
- `batch_mul_sub_128_single_warp`

They use warp shuffles, so every lane in a participating warp must execute the same helper call at the same time.

# Recursive Path Details

## Transform Kernel (`batch_mul_toom22_transform_kernel`)

Inputs are split into:

- low half (`a1`, `b1`) of length `L_split`
- high half (`a0`, `b0`) of length `L - L_split`

Outputs are packed as 3 groups (length `L_half` each):

- group 0: high half
- group 1: low half
- group 2: sum (high + low) with full carry propagation

Layout:

- `A_out = [A0, A1, A2]` where each group has `N * L_half` words
- `B_out = [B0, B1, B2]` similarly

## Reconstruct Kernel (`batch_mul_toom22_reconstruct_kernel`)

Given:

- `c0 = a0*b0`
- `c1 = a1*b1`
- `c2 = (a0+a1)*(b0+b1)`

it computes:

`a*b = c1 + ((c2 - c0 - c1) << L_split) + (c0 << (2*L_split))`

The kernel performs:

1. Two subtract rounds on `c2` (subtract `c0`, then `c1`) with borrow propagation.
2. Carry-aware stitched additions into the final output windows.
3. Final writeback to `ret`.

# Direct Level 1 Kernel (`batch_mul_toom22_directlv1_kernel`)

Used when one Toom-2 reduction is enough (next level fits direct multiply).

For each instance:

1. Split at `L_split = ceil(L/2)` into two parts.
2. Build 3 evaluation points per operand:
   - one point for each split part
   - one point for their sum
3. Compute 3 pointwise products.
4. Interpolate with subtract/add stages and place into final `2L` result.

# Direct Level 2 Kernel (`batch_mul_toom22_directlv2_kernel`)

Used for `L <= 252` when directlv1 is not yet applicable.

## Data Decomposition

It uses quarter split:

- `L_split = ceil(L / 4)`
- `L_quad = L_split + 1`

Each operand is represented with 9 evaluation polynomials (`a[0..8]`, `b[0..8]`):

- base parts:
  - `0`: part0
  - `1`: part1
  - `3`: part2
  - `4`: part3
- fused during load:
  - `2 = 0 + 1`
  - `5 = 3 + 4`
- second transform round:
  - `6 = 0 + 3`
  - `7 = 1 + 4`
  - `8 = 2 + 5`

The current code assumes `blockDim.y == 9` for this kernel.

## Important Optimization Notes

1. The first transform pair (`2`, `5`) is fused into the load stage.
2. `8` is computed as `2 + 5` (not `0 + 1 + 3 + 4`).
3. No transform lookup table is used now; source/destination indices are derived directly from `threadIdx.y`.

## Interpolation Structure

After 9 pointwise products (`r[0..8]`):

1. `r[2], r[5], r[8] -= (r[0],r[3],r[6]) + (r[1],r[4],r[7])`
2. `r[6], r[7], r[8] -= (r[0],r[1],r[2]) + (r[3],r[4],r[5])`
3. Collapse each 3-lane group with grouped carry-aware additions.
4. Add the collapsed `r[6]` branch back into the final accumulator window.
5. Store `r[0]` as final result for this instance.

# Workspace Sizing

`workspace_size_words_internal(N, L)`:

1. Returns 0 when `L <= BATCH_MUL_DIRECT_L_MAX`.
2. Computes `L_split`, `L_half` (with same alignment rule), `c_size = 2 * L_half`.
3. If `L > BATCH_MUL_TOOM22_LV2_MAX`, adds:
   - `3 * N * L_half * 2` words (`A_combined + B_combined`)
   - `3 * N * c_size` words (`C_combined`)
4. Adds recursive workspace for `(3N, L_half)`.

Public API `batch_mul_toom22_workspace_size(N, L)` applies this to one chunk (`chunk_N = get_N_max(N, L)`) and returns bytes.

# Current Test/Baseline Setup

`batch_mul_toom22_test.cu` does:

- randomized correctness across small/medium/large `(N, L)` ranges
- benchmark at `L = 65, 121, 241, 481` with `N*L ~= 1e8`

Latest recorded benchmark comment in test file:

- `L=65`: `2.716 ms`
- `L=121`: `2.364 ms`
- `L=241`: `4.192 ms`
- `L=481`: `8.714 ms`
