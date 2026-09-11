# gpupi: Pi on one or multiple GPUs

CUDA Chudnovsky Pi computation with exact 96-bit NTT multiplication. The
original `pi_bs_gpu` computes up to **5 billion fractional decimal digits** on
one 96 GB GPU. The new `pi_mgpu` computes **10 billion, 25 billion, 50 billion, and 100 billion digits**
on the eight-GPU server, with all complete outputs verified against y-cruncher.

## Multi-GPU build and run

```sh
make -j8 CUDA_ARCH=sm_120
./pi_mgpu 10000000000 --benchmark
./pi_mgpu 25000000000 --benchmark
./pi_mgpu 50000000000 --benchmark
./pi_mgpu 100000000000 --benchmark
./pi_mgpu 100000000000 > pi-100b.txt
```

## Multi-GPU results: 10b through 100b

All four sizes have been compared **byte for byte against independently
computed y-cruncher references** on the eight-GPU server (eight 96 GB cards
and approximately 2 TB of host RAM). Timings below are in seconds.

| Fractional digits | Our fastest measured | Release computation | Release complete process | Fastest listed y-cruncher | Best-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10b | [**19.270**](benchmark_mgpu_10b.json) | [19.537](benchmark_mgpu_10b.json) | 23.431 | [44.059](https://www.numberworld.org/y-cruncher/benchmarks/charts/10b.html) | **2.29×** |
| 25b | [**68.738**](validation/tuning/async144-v24.json) | [69.007](benchmark_mgpu_25b.json) | 75.906 | [120.282](https://www.numberworld.org/y-cruncher/benchmarks/charts/25b.html) | **1.75×** |
| 50b | [**191.629**](validation/50b/v6-affinity320.json) | [199.707](benchmark_mgpu_50b.json) | 211.805 | [275.297](https://www.numberworld.org/y-cruncher/benchmarks/charts/50b.html) | **1.44×** |
| 100b | [**591.787**](validation/100b/v3-serial32.json) | [626.418](benchmark_mgpu_100b.json) | 648.648 | [672.558](https://www.numberworld.org/y-cruncher/benchmarks/charts/100b.html) | **1.14×** |

**Fastest measured** includes individual tuning trials; the 25b, 50b, and
100b best trials used profiling and explicit settings. These are best observed
times, with raw records linked in the table. **Release** columns use three-run
medians for the respective 10b/25b and 50b releases, and one final-build run
for 100b, all with automatic defaults, no warmup, and profiling disabled.
The earlier release measurements retain their original source hashes; they
were recorded before the later extensions. Computation and complete-process
medians are calculated independently.

The final 100b release benchmark takes **10 minutes 26 seconds**, giving a
**1.07×** computation-time speedup over the listed y-cruncher leader. Its
separate full-output verification run computed the digits in 615.617 s and
then passed the complete reference comparison. Allocation order affects
memory reuse and runtime, so the fastest tuning result is not a fixed bound.

### Comparison scope

The linked y-cruncher charts were checked on September 11, 2026. Their fastest
listed entries at all four sizes use **dual AMD EPYC 9R14 CPUs on AWS
hpc7a.96xlarge, 740 GB RAM, and y-cruncher v0.8.1**. The 10b/25b/50b charts
report an April 5, 2025 update; the 100b chart reports May 26, 2025. These are
unofficial comparisons across different hardware against those listed
records. Speedup is the listed y-cruncher time divided by our measured time.

Our computation timing includes host allocation/reservation, binary splitting,
square root, division, decimal conversion, and the decimal device-to-host
transfer. CUDA context startup, output writing, verification, and teardown
are excluded. Complete-process timing includes startup and teardown but prints
only digit previews. The [y-cruncher benchmark definition](https://www.numberworld.org/y-cruncher/#Benchmarks)
also includes decimal conversion and excludes writing digits to disk.

### Implementation strategy

- **Parallel binary splitting:** distribute independent exact subtrees across
  the GPUs. At 100b, selected upper parent merges execute sequentially to
  reduce retained host products, while each large multiplication uses all
  eight GPUs.
- **Bounded 96-bit NTT:** retain 32-bit limbs and the original prime
  `2^96 - 2^32 + 1`. Independent products use local GPU transforms; larger
  products use a distributed six-step NTT with peer transposes. Cyclic-tail
  correction and recursive Karatsuba keep every transform within `2^32`
  points, without an additional CRT modulus.
- **Large integers and remaining stages:** use 64-bit host integer lengths
  for 50b and above, and 64-bit term factors above the 32-bit term-index range
  for 100b. Square-root and reciprocal refinement use GPU multiplication;
  decimal conversion splits work into GPU-sized partitions.
- **Memory reuse:** reusable pinned arenas, NUMA-aware host arithmetic, and
  reservation overlapped with subtree computation reduce allocation and
  transfer overhead. Separate registrations of at most 384 GiB avoid the
  driver's large-registration limit. PCIe communication and host-memory
  overhead increasingly limit scaling at larger sizes.

The final verified 100b run peaked at **1.182 TB of host memory**, with sampled
GPU peaks up to **85.884 GiB per card**. The optional lower-memory schedule
completed in 615.676 s with a process host peak of approximately 742 GB;
see [MULTI_GPU.md](MULTI_GPU.md) for its settings and memory measurements.

Verified outputs remain on the server under
`~/work/gpupi/batchmp3/results/` as `pi-10b.txt`, `pi-25b.txt`, `pi-50b.txt`,
and `pi-100b.txt`. [The result manifest](results/manifest.json) records sizes
and fractional-digit SHA-256 hashes. [Validation records](validation/README.md)
include complete comparisons, arithmetic tests, CUDA memory checks, and
source identification.

The release settings are selected automatically on the eight-card RTX Pro
6000 server. See [MULTI_GPU.md](MULTI_GPU.md) for the detailed algorithm,
memory requirements, and tuning controls. Original single-GPU usage and
measurements follow.

## Single-GPU build and run

Requires CUDA 13 and GMP development headers/libraries:

```sh
make -j8
./pi_bs_gpu 5000000000 --benchmark
./pi_bs_gpu 5000000000 > pi.txt
```

The full 5b computation is intended for the **96 GB RTX Pro 6000**.
The RTX 5090 remains suitable for the billion-digit case; the complete 5b
working set exceeds its device memory.

Default output is `3` followed by exactly the requested number of fractional
digits and a newline, without a decimal point. Timing goes to stderr.
`--benchmark` prints timing and previews; `--profile-stages` prints stage
timings; `--full-print` includes the hexadecimal representation.

```sh
python3 benchmark_pi.py --digits 5000000000 --warmups 1 --runs 5
```

`elapsed_ms` includes binary splitting, square root, reciprocal, and decimal
conversion. It excludes initial CUDA setup, final output transfer, hashing,
and printing. `workspace_max` is the arithmetic scratch capacity, **not total
peak device memory**. The 5b implementation uses about 48.5 GiB of arithmetic scratch.

## Measured performance

RTX Pro 6000, CUDA 13.0, default `-O3` build, one warmup and five serial runs:

| 5 billion digits | Time |
|---|---:|
| Median | **20.297523 s** |
| Best | **20.291771 s** |
| Worst | **20.665990 s** |
| Fastest cited CPU chart entry | 22.377 s |

All five trials beat the CPU chart entry; the median is **9.29% lower**.
The latest modular-arithmetic optimization reduces the previous 22.117398 s
median by **8.23%**.
The same build also improves the smaller sizes on the server (one warmup,
three serial trials each): **3.213666 s median at 1b** and **9.600286 s at
2.5b**. All digits at both precisions match their verified reference digests.

Final 5b benchmark and smaller-size regression measurements are recorded in
[benchmark_5b_results.json](benchmark_5b_results.json), with their source commit,
raw trials, timing scope, and full-output validation. The comparison target is
**22.377 seconds**, the fastest entry on the [5b CPU chart](https://www.numberworld.org/y-cruncher/benchmarks/charts/5b.html)
(dual AMD Epyc 9R14, y-cruncher v0.8.1; chart updated May 26, 2025).
This is a computation-time comparison, not an official benchmark submission.
Earlier 2.5b and 1b measurements are preserved in
[benchmark_results.json](benchmark_results.json).

The current build was also checked on the **RTX 5090** (CUDA 13.1, default
`-O3`): one warmup and five 1b trials gave **3.518641 s median**, **3.509651 s
best**, and **3.531918 s worst**. This is about 14.2% below the original
user-reported 4.1 s result. All one billion fractional digits match the verified
reference checksum; raw trials are in `benchmark_5b_results.json`.

## Algorithm changes

- NTT field multiplication uses explicit PTX carry chains for its 192-bit
  product. Canonical addition, subtraction and reduction use shorter corrections.
  Inverse normalization computes `(x+t*p)/2^k` with shifts and carries, where
  `t = -x mod 2^k`, replacing multiplication by an inverse power of two.
- Decimal digit counts, bit counts, signed shifts, and workspace byte counts
  use 64-bit types.
  Limb counts remain 32-bit, with a checked product limit of `2^31` limbs.
- Two exact subtrees use 356,515,840 Chudnovsky terms at 5b precision
  (178,257,920 at 2.5b). Their merge preserves the P/Q/R scaling.
- Paired products share the transform of their common multiplier during binary
  splitting and the forest merge. Compact carry scratch lets the largest
  paired transform fit the 96 GB GPU.
- Guaranteed common factors are cancelled from P, Q and R through 136-term
  blocks. Fused exact division keeps intermediate carries in registers.
  `generate_common_factors.py --check` proves the constants by exhaustive
  residue enumeration.
- Cached products use one shared-memory transform array and fused tiles up to
  4096 coefficients, with aligned forward/inverse schedules at k16, k17, k25.
- Products just beyond a power of two use a smaller cyclic NTT. A separate
  small convolution of reversed high limbs recovers the wrapped tail before
  full-width carry propagation.
- Large NTTs use tuned radix schedules and fused local multiplication/inverse
  stages. Decimal leaves process multiple independent warps per block.
- Broadcast multipliers (the shared powers in decimal conversion) are transformed
  once and reused across rows and workspace batches, including small NTTs.
- The exact Pell identity `P^2 - 10005 Q^2 = 1` simplifies square-root doubling
  to `P_next = 2*P^2 - 1`, eliminating one multiplication per iteration.
- Cyclic-aware batching uses a tuned six-million-coefficient budget. The
  `2^24` and `2^28` schedules align forward and inverse shared-memory tiles.
- Scratch growth releases the old arena before allocating its replacement.

The prime is `p = 2^96 - 2^32 + 1`. For product lengths through `2^31` limbs,
any coefficient is at most `2^30 * (2^32 - 1)^2 < 2^94 < p`. Thus the original
32-bit chunks remain exact at the required size: 30-bit repacking or an extra
CRT modulus is unnecessary. The relevant limit was transform padding and
implementation size arithmetic, rather than 96-bit coefficient capacity.

## Correctness and reproducibility

The independent 5b reference was generated with the sibling y-cruncher
v0.8.7 build 9547, with verification enabled (98.810 s computation):

```sh
mkdir -p .scratch/reference
'../ycruncher/y-cruncher v0.8.7.9547-static/y-cruncher' \
  skip-warnings pause:-2 custom pi -dec:5000000000 -hex:0 -verify:1 \
  -o .scratch/reference
python3 check_pi.py '.scratch/reference/Pi - Dec - Chudnovsky.txt' 5000000000
```

Full server output is checked against the SHA-256 of **all five billion
fractional digits** from that reference. To repeat this without retaining a
5 GB reference file:

```sh
python3 check_pi.py --sha256 \
  435c9b2f0794ae48141c831500c2aa7f4464e68823916012e7128df27cfe0a72 \
  5000000000
```

Both comparator modes stream the output and check its exact length and newline.
The digest mode hashes every fractional digit, including the last digit.
The byte comparator also accepts an explicit command after `--`, including SSH.
The 1b and 2.5b reference digests are retained in the benchmark JSON files.

Component checks:

```sh
./batch_arith_test_compile.sh && ./batch_arith_test
./batch_mul_ntt_test_compile.sh && ./batch_mul_ntt_test --skip-benchmark
./shared_mul_test_compile.sh && ./shared_mul_test
./mod96_test_compile.sh && ./mod96_test
./exactdiv_chain_test_compile.sh && ./exactdiv_chain_test
python3 generate_common_factors.py --check
# Focused broadcast and workspace-batch boundary checks:
./batch_mul_ntt_test --broadcast-only
./batch_bitlength_test_compile.sh && ./batch_bitlength_test
./batch_decimal_naive_test_compile.sh && ./batch_decimal_naive_test
./large_arith_test_compile.sh && ./large_arith_test 29
# On the 96 GB server, also test 30 and 31:
./large_arith_test 31
compute-sanitizer --tool memcheck --error-exitcode 99 ./pi_bs_gpu 250000 --benchmark
```

The large test checks every product limb for maximal inputs, squares,
unequal operands and cyclic tails, plus shifts exceeding 8 billion bits.
The GMP NTT suite covers random values, broadcasts, strides, carry boundaries
and multiple workspace batches. Pi boundary checks and full billion-digit
regressions also pass. See [WORK_LOG.md](WORK_LOG.md) for experiments and
validation milestones.

## Experiment switches

These opt out of individual optimizations for comparisons:

- `GPUPI_LINEAR_NTT=1`: use fully padded linear transforms.
- `GPUPI_SINGLE_TREE=1`: round all terms to one power-of-two tree.
- `GPUPI_OLD_LARGE_SCHEDULE=1`: restore the original large NTT schedule.
- `GPUPI_OLD_DECIMAL=1`: restore the original decimal leaf kernel.
- `GPUPI_OLD_MEDIUM_SCHEDULE=1`: disable the `2^24`/`2^28` schedule changes.
- `GPUPI_OLD_BROADCAST=1`: disable shared transforms for large products.
- `GPUPI_OLD_SMALL_BROADCAST=1`: disable shared transforms for small products.
- `GPUPI_OLD_BATCHING=1`: size cyclic batches from the fully padded transform.
- `GPUPI_CACHE_COEFS=4000000`: override the default 6000000 coefficient budget.
- `GPUPI_OLD_SHARED_PAIR=1`: calculate paired products separately.
- `GPUPI_OLD_P_PAIR=1`: disable the P/PR pairing in binary splitting.
- `GPUPI_OLD_CACHED_ALIGN=1`: disable cached k16/k17 schedule alignment.
- `GPUPI_OLD_CACHED_TILE=1`: disable the cached k25 tile schedule.
- `GPUPI_COMMON_FACTORS=0`: disable extra common-factor cancellation (default 136).
- `GPUPI_OLD_DIVCHAIN=1`: use separate passes for exact division chains.
- `GPUPI_PAIR_CACHE_COEFS=6000000`: override the paired-product batch budget.
- `GPUPI_TRACE=1`: print arithmetic stages immediately for failure diagnosis.

Compile-time comparisons for the modular arithmetic use
`GPUPI_OLD_MUL96`, `GPUPI_OLD_NTT_SCALE`, `GPUPI_OLD_ADDSUB96`, or
`GPUPI_OLD_REDUCE96`. Build into a fresh directory so changed preprocessor
flags take effect, for example:

```sh
mkdir -p .scratch
make -j8 BUILD_DIR=.scratch/old-mod-build TARGET=.scratch/pi-old-mod \
  CPPFLAGS=-DGPUPI_OLD_MUL96
```

The reciprocal and decimal conversion routines are specialized for this Pi
pipeline; they are not general-purpose arbitrary-precision division APIs.
