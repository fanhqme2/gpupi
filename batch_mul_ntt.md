# `batch_mul_ntt.cu` structure notes

This document summarizes the current structure of the CUDA NTT multiplier in [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu). The goal is to make future edits safer by describing the main execution paths, data layout, and the current build/test setup around the file.

## 1. High-level contract

- Public declarations live in [batch_mul_ntt.h](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.h#L5).
- `batch_mul_ntt()` multiplies `N` independent pairs of big integers, where:
  - `A` contains `N * L_a` 32-bit words.
  - `B` contains `N * L_b` 32-bit words.
  - `ret` receives `N * (L_a + L_b)` 32-bit words.
  - all pointers are device pointers.
- Coefficients are treated as base `2^32` digits. The NTT length is `K = 2^k`, where `K >= L_a + L_b`.
- For large sizes the code may process the batch in chunks `N_batch`; the caller still sees one logical `N`.

## 2. Arithmetic model

### 2.1 Modulus representation

- NTT-domain values are stored as `uint3`, effectively a 96-bit residue.
- `add_mod`, `sub_mod`, and `mul_mod` implement arithmetic modulo the NTT prime in a custom 3-limb representation at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L18), [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L41), and [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L56).
- The code relies on helpers from [batch_mul_addsub_asm.h](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_addsub_asm.h) for carry-chain arithmetic.

### 2.2 Precomputed roots

- Three constants define the primitive roots and `1/2` in the field near the top of [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L6).
- `init_ntt_precomputed_tables()` fills:
  - `roots_table_lv1[65536]`
  - `roots_table_lv2[65536]`
  - `inv2n_table[33]`
  at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L150).
- The two root tables split a twiddle index into high and low 16-bit parts. Most kernels reconstruct a twiddle as:
  - `roots_table_lv2[idx & 0xffff]`
  - optionally multiplied by `roots_table_lv1[idx >> 16]`
- `roots_table_lv1` and `roots_table_lv2` are filled in bit-reversed order with `fill_in_power_table_bitrev16`, while `inv2n_table` is filled linearly with powers of `1/2`.

## 3. Major execution paths in `batch_mul_ntt()`

The top-level dispatcher is [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1642).

### 3.1 Small transforms stay entirely local

- If `k <= 9`, `mul_fft_local<512>` is used.
- If `k == 10`, `mul_fft_local<1024>` is used.
- If `k == 11`, `mul_fft_local<2048>` is used.
- If `k == 12`, `mul_fft_local_spill<4096>` is used.
- These kernels perform forward transform, pointwise multiplication, inverse transform, and carry recovery inside one kernel; see [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1310) and [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1458).
- `mul_fft_local_spill` uses workspace to save one transformed operand because a 4096-point local kernel no longer fits comfortably in shared memory.

### 3.2 Large transforms use a staged global-memory pipeline

For `k >= 13`, the code:

1. Picks `N_batch = get_N_max(...)` to cap the active batch size; see [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1622).
2. Splits workspace into:
   - `parts_a = reinterpret_cast<uint3*>(workspace)`
   - `parts_b = parts_a + N * K`
3. Runs a forward transform over `2*N` sequences by storing `A` and `B` back-to-back in the same `parts` area.
4. Applies fused pointwise multiply plus division by `K` using `pointwise_mul`; see [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L509) and [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1755).
5. Runs the inverse transform only on the product data.
6. Reconstructs base-`2^32` output words and propagates carries.

## 4. Forward transform kernel families

### 4.1 Generic radix-2 stage

- `fft_level_forward` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L156) is the simple butterfly form.
- It is currently not used by the main dispatch path; the higher-radix and final-local kernels are preferred.

### 4.2 Radix-16 path

- `fft_level_forward_radix16` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L179) fuses 4 radix-2 levels at once.
- `fft_level_forward_radix16_initial` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L272) also performs zero-padding and loads from `A`/`B` directly, so the first forward stage can skip a separate input marshaling kernel.
- The main loop advances `i += 3` after this kernel because the `for` loop also increments `i`, so one radix-16 launch consumes 4 logical FFT levels; see [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1736).

### 4.3 Radix-32 path

- `fft_level_forward_radix32` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L348) fuses 5 radix-2 levels using shared memory `local_coefs[32][32]`.
- `fft_level_forward_radix32_initial` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L412) combines input load, zero-padding, and the first 32-point local FFT.
- Dispatch prefers radix-32 when:
  - `k >= 22`, or
  - `i != 0 && k - i == 16`
  at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1719).
- Otherwise it uses radix-16.

### 4.4 Final local forward stage

- Once the remaining span is at most 4096 points, `fft_level_forward_final<1024/2048/4096>` finishes the forward transform inside shared memory; see [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L444).
- This kernel is launched on `N * 2` transforms because `parts_a` and `parts_b` still share contiguous storage.

## 5. Pointwise multiply

- `pointwise_mul` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L509) multiplies `parts_a[j] * parts_b[j] * inv2n_table[k]`.
- This fuses normalization by `K` into the spectral product, so the inverse FFT does not need a separate scaling pass.

## 6. Inverse transform kernel families

### 6.1 Final local inverse stage

- The inverse begins with `fft_level_backward_final<...>` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L895).
- It handles the deepest local portion first. `i0` is chosen so the remaining suffix is one local kernel launch:
  - aligned to 4 levels for the radix-16 path
  - aligned to 5 levels for the radix-32 path when `k >= 24`
  at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1764).

### 6.2 Radix-16 and radix-32 backward kernels

- `fft_level_backward_radix16` is at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L532).
- `fft_level_backward_radix32` is at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L623).
- These mirror the forward fused kernels, but with inverse twiddle selection and inverse butterfly ordering.

### 6.3 Initial inverse kernels also extract output words

- `fft_level_backward_radix16_initial` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L758)
- `fft_level_backward_radix32_initial` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L679)
- These are not just FFT kernels. They also:
  - convert `uint3` coefficients back into low 32-bit result words
  - fold in the adjacent `.y` and `.z` limbs that belong to neighboring base-`2^32` digits
  - emit leftover high-word spill into `workspace`
- This dual role is important. Any change to coefficient representation or output normalization must keep these kernels consistent with the later carry-reduction passes.

## 7. Carry-reconstruction logic

This is the most delicate part of the file.

### 7.1 Why extra carry work exists

- After inverse NTT, each coefficient still lives in `uint3`.
- The final big-integer word at position `i` depends on:
  - current coefficient `.x`
  - previous coefficient `.y`
  - coefficient two positions back `.z`
- The code also uses a nontrivial carry encoding via `ushort2 {x, y}` and `combine_carry()` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L668).
- The `+2` followed by checking carry-out appears throughout the output path. That convention is part of the carry encoding and must be preserved if this logic is rewritten.

### 7.2 Single-block carry completion

- `add3_single_block` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L938) is used only when `K == 8192`.
- `add2_single_block` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1218) is used for larger transforms when `N >= 85`.
- Both kernels finish carry propagation for one number inside one block, but they expect different intermediate state:
  - `add3_single_block` reads `uint3` coefficients directly from `parts_a`
  - `add2_single_block` reads partially materialized `ret` words plus spill words from `workspace`

### 7.3 Multi-block carry completion for small `N`

- When `N < 85`, the code uses a 3-kernel sequence:
  - `add2_reduce_blocks` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1135)
  - `add2_combine_blocks` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1035)
  - `add2_apply_blocks` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1101)
- This path exists because one large integer may span multiple 2048-word blocks, so carry information must first be reduced per block, then globally combined, then applied.

## 8. Workspace layout and sizing

- Workspace sizing is defined by `batch_mul_ntt_workspace_size()` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1867).
- Cases:
  - `K <= 2048`: no workspace
  - `K == 4096`: `N_batch * K * sizeof(uint3)`
  - otherwise: `N_batch * K * 2 * sizeof(uint3)`
- In the large staged path the workspace is reused for different meanings over time:
  - initially: two `uint3` arrays, `parts_a` and `parts_b`
  - during inverse-output extraction: `parts_b` is reused as temporary `uint32_t` spill storage
  - during block carry reduction: sections are reinterpreted as `ushort2`
- This means aliasing is intentional. Any future change to kernel order or workspace type assumptions must keep these reinterpret-casts compatible.

## 9. Batch-size throttling

- `get_N_max()` at [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1622) reduces the active batch size for large `K`.
- The heuristic is based on `K` and a fixed cache-like budget:
  - default limit near `4,000,000 / K`
  - special case `K == 4096` uses `8,000,000 / K`
- This is a performance heuristic, not an interface requirement.

## 10. Current test harness

The current dedicated harness is [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu).

### 10.1 Table setup

- The test allocates device root tables and `inv2n_table`, calls `init_ntt_precomputed_tables()`, then reuses those tables across all tests and benchmarks; see [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu#L416).

### 10.2 Correctness testing

- `test_configuration()` at [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu#L269) does full correctness checking with GMP:
  - random host inputs
  - device copies
  - call to `batch_mul_ntt()`
  - copy back
  - compare each product against `mpz_mul`
- `run_suite()` at [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu#L329) generates mixed `(N, L_a, L_b)` cases.
- `main()` runs:
  - small tests with `N * (L_a + L_b) <= 16384`
  - medium tests up to `2^20`
  - large tests around `1e7`
  - explicit large-`L` tests with `N in {1,2,3,4,8}`
  at [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu#L428).

### 10.3 Benchmarking

- `benchmark_configuration()` at [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu#L355) measures throughput for equal-length operands.
- It uses cuRAND to fill device buffers, performs three warmup runs, then averages 10 timed iterations using CUDA events.
- Reported metrics:
  - average kernel time
  - multiplies per second
  - effective bandwidth based on `A + B + ret`
- After each benchmark it runs a lightweight modular sanity check instead of a full GMP verification.

### 10.4 Fast modular sanity check

- The quick check code is at [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu#L43) through [batch_mul_ntt_test.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test.cu#L256).
- It reduces `A`, `B`, and `ret` modulo `10007` on the GPU and verifies:
  - `ret mod 10007 == (A mod 10007) * (B mod 10007)`
- This is only a warning-based spot check for benchmarks, not a proof of correctness.

## 11. Current compilation script

- The dedicated compile script is [batch_mul_ntt_test_compile.sh](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt_test_compile.sh).
- Current command:

```bash
nvcc -O3 -o batch_mul_ntt_test batch_mul_ntt_test.cu batch_mul_ntt.cu -lgmp -lcurand
```

- This means the harness currently depends on:
  - CUDA / `nvcc`
  - GMP
  - cuRAND
- The script only builds; it does not run the test binary.

## 12. Modification notes

- If you change twiddle indexing, review both forward and backward kernels. They do not all compute `bitrev_seq_id` the same way.
- If you change coefficient packing or modulus arithmetic, audit:
  - `mul_fft_local`
  - `mul_fft_local_spill`
  - `fft_level_backward_radix16_initial`
  - `fft_level_backward_radix32_initial`
  - `add3_single_block`
  - `add2_*`
- If you change workspace use, re-check all `reinterpret_cast` sites in [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1690), [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1806), and [batch_mul_ntt.cu](/mnt/c/Users/fhq/work/gpupi/batchmp/batch_mul_ntt.cu#L1849).
- If you change batching heuristics, confirm `batch_mul_ntt_workspace_size()` still matches the maximum in-flight memory usage.
- If you change the carry path, keep the GMP-based correctness suite as the primary regression test, because the benchmark quick check is not sufficient to catch subtle carry bugs.
