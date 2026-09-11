# Multi-GPU Pi through 100 billion decimal digits

`pi_mgpu` extends the existing CUDA implementation with host orchestration,
bounded-precision Chudnovsky merges, and an exact distributed NTT. The original
`pi_bs_gpu` executable remains available for the single-GPU 5b workload.

## Build

Use 64-bit Linux, CUDA, GMP with 64-bit limbs and C++ development headers,
and C++17-capable host tools:

```sh
make -j8 pi_mgpu distributed_ntt_test wide_int_test CUDA_ARCH=sm_120
./pi_mgpu --test
./pi_mgpu 10000000000 --benchmark
./pi_mgpu 25000000000 --benchmark
./pi_mgpu 50000000000 --benchmark
./pi_mgpu 100000000000 --benchmark
```

For the supplied server's noninteractive SSH shell, add
`NVCC=/usr/local/cuda/bin/nvcc` to the `make` command if CUDA is absent from
`PATH`.

The test server has eight 96 GB SM120 GPUs, two Xeon 8558P sockets, and 1.8 TiB
RAM. Its CUDA 13.2 compiler must emit native SM120 code because the installed
driver cannot JIT that compiler's newer PTX. `CUDA_ARCH` can be changed for
other supported hardware; use `make clean` or a fresh `BUILD_DIR` when changing
compiler flags or architecture. The largest workload needs substantial host RAM;
GPU memory alone is not the working storage for the complete calculation.
512 GB or more of host RAM is recommended for 25b; 1 TiB provides ample
headroom for 50b. The 100b case uses the supplied machine’s approximately
2 TB of host RAM; measured peaks are reported below. The measured eight-card settings are defaults; no environment
flags are required.

## Arithmetic and work distribution

The prime is `p = 2^96 - 2^32 + 1`. Its maximum power-of-two transform order
is `2^32`. Increasing the transform order past this limit would be incorrect.
GPU convolution inputs use ordinary 32-bit limbs; host integers pack pairs
into 64-bit limbs. The field is large enough to recover
the convolution coefficients exactly, including the cyclic products below:
with at most `2^32` terms, the coefficient bound is
`2^32 * (2^32 - 1)^2 < p`.

The implementation combines three multiplication paths:

1. Independent products that fit a GPU use the original tuned NTT kernels.
   Keeping several such products on different GPUs is faster on this server
   than distributing every multiplication across all eight cards. Dispatch
   tests the combined operand length against twice the selected grain, so
   unequal operands use a single GPU whenever their product fits. Operands
   are ordered by length to bound persistent buffer capacity.
2. Larger products use a six-step NTT. An outer radix across GPUs and a peer
   transpose give each GPU a complete frequency column. Local transforms,
   pointwise multiplication, and inverse transforms run independently. A
   second transpose restores coefficient order. Exact associative carry maps
   handle carries across warp, partition, and device boundaries.
3. A product slightly longer than a power-of-two transform uses cyclic
   convolution plus an exact reversed-tail convolution. For length `L = K+w`,
   the first `w-1` coefficients are corrected by subtracting the reversed
   product of the highest `w-1` input limbs. The same tail supplies coefficients
   above `K`; carry propagation continues across this boundary. When
   `w <= K/4`, one `K` transform and a transform of at most `K/2` replace three
   full Karatsuba transforms. Other shapes retain the Karatsuba fallback.

The 25b mantissas fit this cyclic-tail case at `K = 2^32`. At 50b, a larger
product is split with Karatsuba until each child fits a supported linear or
cyclic transform. No transform exceeds `2^32`; neither another CRT prime nor
coefficient repacking is required. Squaring uses one forward
transform. Peer copies use separate streams; PCIe communication dominates
the distributed transform's arithmetic on this machine.

Measured individual peer transfers were about 29–38 GB/s, versus 56–57 GB/s
for pinned host transfers. `cudaMemcpyPeerAsync` was faster without permanently
enabling peer mappings on this server, so permanent mappings are disabled by
default. These measurements motivated the mix of independent local products
and distributed transforms; the raw transfer and NTT comparisons are retained
in `validation/development/`.

Binary splitting starts with independent exact GPU subtrees. Above those
subtrees, P, Q, and R have separate binary exponents and mantissas retaining
the requested precision plus 512 guard bits. Whole-limb views discard excess
precision without copying buffers. The subtree factors and storage are widened
for the high term indices needed at 50b. The 50b configuration uses 32 exact
subtrees and 3,565,158,400 terms. The 100b configuration uses 56 subtrees and
7,130,316,800 terms. Subtrees crossing 2^32 terms select 64-bit term factors
and two-word scalar multiplication; earlier ranges keep the original fast
specialization. P/Q/R need at most 106/138/188 bits through 2^33 terms, fitting
the existing leaf arrays. Larger subtree ranges are rejected before launch.

At 50b, even a retained mantissa exceeds GMP's signed-int limb-count limit,
so a high-product window alone is insufficient. `wide_int.h` supplies host
integers with 64-bit signed lengths and `size_t` capacities. It uses standard
GMP `mpn` primitives, the registered-memory allocator, and parallel host
operations. Bounded startup/reference operations still use ordinary GMP `mpz`;
large values never enter that API. The system GMP library and ABI are unchanged.
Karatsuba assembly uses parallel zeroing, copying, and exact carry propagation.

Pell square-root products run independently. Reciprocal refinement uses GPU
multiplication and parallel host arithmetic. Decimal conversion first queues
independent large splits, then runs the original converter on GPU-sized
partitions. Direct bit slices avoid copying whole products just to discard
most of their bits.

Large host buffers use CUDA registration after parallel page initialization.
Reusable arenas split and coalesce allocation blocks at 64 KiB granularity.
Initial reservations above 384 GiB are split into separate arenas, each
allocated and registered independently: the measured driver rejects
individual 512 GiB and 640 GiB registrations. The allocator never coalesces
across arenas, so each allocation stays within one CUDA registration range.
This also avoids CUDA's rejection of transfers crossing registered ranges.
Smaller reservations retain their previous path.
The final decimal output also reuses registered host memory, avoiding a large
serial zero-fill and pageable downloads. Initial arena preparation overlaps
the first GPU subtree wave; workers wait for it before allocating their GMP
outputs and downloading results. This hides most of the first wave's GPU work
behind host allocation. The reported time includes the complete reservation.

## Configuration

`CUDA_VISIBLE_DEVICES` selects physical cards; `GPUPI_GPUS` selects how many of
the visible cards to use, up to eight. Local multiplication sizes and subtree
caps adapt to available GPU memory. The distributed engine supports 1, 2, 4,
or 8 cards; the full 10b/25b/50b/100b performance measurements use eight 96 GB cards.

| Control | Default on the measured server |
| --- | --- |
| `GPUPI_MGPU_TREE_BLOCKS` | `8388608` blocks of 17 terms |
| `GPUPI_MGPU_GRAIN` | `536870912` 64-bit limbs |
| `GPUPI_DISTRIBUTED_MIN` | `2147483649` 32-bit product limbs |
| `GPUPI_SERIAL_DISTRIBUTED_MIN` | `268435456` 32-bit product limbs |
| `GPUPI_MGPU_DECIMAL_DIGITS` | `400000000` digits per GPU partition cap |
| `GPUPI_HOST_THREADS` | 32 |
| `GPUPI_HOST_AFFINITY` | 1 from 30b on eight GPUs; 0 otherwise |
| `GPUPI_MGPU_RESERVE` | 48 GiB at 10b; 144 GiB at 25b; 320 GiB at 50b; 640 GiB at 100b |
| `GPUPI_MGPU_SERIAL_MERGES` | 32 from 75b on eight GPUs; 0 otherwise |
| `GPUPI_MGPU_SERIAL_PRODUCTS` | 0 |
| `GPUPI_MGPU_ASYNC_RESERVE` | 1, overlap initial reservation with GPU subtrees |
| `GPUPI_MGPU_PIN_CACHE` | From 75b: half of host RAM, capped at 1 TiB; otherwise one quarter, capped at 512 GiB |

The grain is half the maximum combined operand length for a local product;
it is not a separate limit on each operand.
Reserve and cache controls are in bytes. Reservation is disabled by default
for smaller calculations and when the machine lacks sufficient host RAM.
At 50b and above, host workers use physical cores selected across the CPU sockets
for memory initialization and parallel arithmetic. `GPUPI_HOST_AFFINITY=0`
disables this; each worker restores its previous CPU mask afterward.
From 25b, the last subtree wave is subdivided to keep all eight GPUs busy and
reduce peak live host memory; `GPUPI_MGPU_BALANCE_TREES=0` disables this.
From 75b on eight GPUs, parent merges execute sequentially once at most 32
nodes remain. Each large multiplication still uses all eight GPUs. This
reduces simultaneously retained products and host registration overhead.
`GPUPI_MGPU_SERIAL_PRODUCTS=1` also collects each product within a parent
sequentially. The tested combination `GPUPI_MGPU_SERIAL_MERGES=16
GPUPI_MGPU_SERIAL_PRODUCTS=1` completed 100b in 615.676 s with a 741.434 GB
registered-memory peak, versus 591.787 s and 1,002.397 GB for the selected
schedule. It remains available as a lower-memory option.
`GPUPI_MGPU_PROFILE=1` prints merge, conversion, and allocation timings;
`GPUPI_DNTT_PROFILE=1` adds transform-phase timings. `--benchmark` prints short
digit previews. Ordinary execution writes all digits to stdout and diagnostics
to stderr. The original `pi_bs_gpu` remains preferable for its tuned smaller
single-GPU workloads.

With asynchronous reservation, `binary_split_s` includes waiting for the arena
and overlaps the allocator's `allocation_s`; do not add those two values.
`GPUPI_MGPU_ASYNC_RESERVE=0` selects synchronous reservation for comparison.

## Validation and timing

All four precisions have been compared byte for byte against separately
generated y-cruncher results on the server. Fractional-digit SHA-256 values:

| Digits | SHA-256 |
| --- | --- |
| 10,000,000,000 | `1dabc86199e9f5e065fc2c839ebf77ba1546afaeb25aadbbccd82ffe522f5541` |
| 25,000,000,000 | `164af5c396b3b87755a99ab07cd6141f8811401d4ccf33b54001f3124f9b54bb` |
| 50,000,000,000 | `76312744aa02ead5482ed3e65c3969f39657d6b221d5dcb22e5d2bb65d786e49` |
| 100,000,000,000 | `78d8a7b1d47170bd95749cc4f4a5f1d047cda93d021fb7790c290433b869fe3a` |

These hashes exclude the initial `3` and trailing newline. Program output is
`3` followed directly by the requested fractional digits and a newline.
y-cruncher's reference files include a decimal point, which `check_pi.py`
handles. It can retain the program output only after successful verification:

```sh
python3 check_pi.py --save pi-100b.txt \
  '/dev/shm/gpupi-ref100b/Pi - Dec - Chudnovsky.txt' 100000000000 \
  -- ./pi_mgpu 100000000000
```

Verified outputs are already saved on the server under
`~/work/gpupi/batchmp3/results/` as `pi-10b.txt`, `pi-25b.txt`, `pi-50b.txt`, and `pi-100b.txt`.
Each file is exactly its requested fractional-digit count plus two bytes.

The regression tests include exact GMP products, unequal lengths, squares,
long carries and borrows, high-index Chudnovsky subtrees, reciprocal refinement,
random cyclic products and product windows, and full-output comparisons at
smaller precisions. Full-scale maximal-input checks compare every retained
32-bit limb, including products that exceed GMP's full-result size limit.

```sh
./wide_int_test
./wide_int_test --large  # requires substantial host memory
./distributed_ntt_test --wrap-random 25
./distributed_ntt_test --max 31
./distributed_ntt_test --wrap 32
python3 benchmark_mgpu.py --digits 100000000000 --runs 1 --output timings-100b.json
```

Reported computation time includes host allocation/reservation, arithmetic,
decimal conversion, and the decimal device-to-host transfer. CUDA context
initialization, output-file writing, verification, and teardown are outside
that interval. The benchmark script also records complete process wall time.

The final 100b benchmark used automatic defaults, no warmup, and profiling
disabled. Its single measured run took **626.418 s** for computation and
**648.648 s** for the complete benchmark process. This is a single-run
measurement. [Raw record](benchmark_mgpu_100b.json) source hashes match
[the final source manifest](validation/100b/source-sha256.json).

The final 100b build computed the digits in 615.617 s with profiling enabled,
then passed another complete y-cruncher byte comparison. Its process host
high-water mark was
1,181,742,465,024 bytes (1.182 TB); registered arenas peaked at 1,181.103 GB
and live allocations at 826.824 GB. Minimum sampled system available memory
was 552.511 GB. Per-card sampled peaks were 84.339–85.884 GiB, including the
driver baseline; five-second sampling can miss brief peaks.

The 100b tuning comparisons, each one profiling-enabled full computation:

| Configuration | Computation (s) | Registered arena peak (GB) |
| --- | ---: | ---: |
| Initial single-reservation attempt, fallback allocation | 735.094 | 1433.293 |
| Separate arenas, parallel parent merges | 631.244 | 1231.599 |
| Separate arenas, parent threshold 32 (selected) | 591.787 | 1002.397 |
| Parent threshold 16, sequential product collection | 615.676 | 741.434 |
| Final automatic defaults, full verification | 615.617 | 1181.103 |

Concurrent allocation order affects memory reuse, so the lowest tuning time
and memory peak are not guarantees. The retained initial reservation uses
separate arenas of at most 384 GiB: a single 640 GiB CUDA registration failed
on this driver. Registering adjacent chunks within one arena was also rejected
because CUDA copies spanning registration boundaries failed. The successful
allocator keeps each returned buffer inside one registered region. See
[100b validation records](validation/100b/README.md) for the probes, complete
comparisons, memory samples, and source snapshots.

The 50b release measurements, before the 100b extension, on September 11, 2026: three serial runs,
no warmup, automatic defaults, and profiling disabled.

| Fractional digits | Computation trials (s) | Median computation (s) | Median complete process (s) |
| --- | --- | ---: | ---: |
| 50b | 201.142, 199.305, 199.707 | **199.707** | **211.805** |

The [50b benchmark record](benchmark_mgpu_50b.json) includes every trial,
hardware details, timing scope, and source hashes matching
[validation/50b/source-sha256.json](validation/50b/source-sha256.json).

Original 10b/25b release measurements on the supplied eight-GPU server,
September 11, 2026 (before the 50b host-integer extension):
three serial runs per size, no warmup, default settings, profiling disabled.

| Fractional digits | Computation trials (s) | Median computation (s) | Median complete process (s) |
| --- | --- | ---: | ---: |
| 10b | 19.665, 19.537, 19.270 | 19.537 | 23.431 |
| 25b | 69.007, 68.798, 70.814 | 69.007 | 75.906 |

Raw records: [10b](benchmark_mgpu_10b.json) and
[25b](benchmark_mgpu_25b.json). Both contain source hashes, hardware details,
environment settings, and all trial logs. The full-output validation runs are
in [validation/](validation/README.md).

The original 10b/25b release's profiled full comparisons measured registered-arena peaks of
68.045 GB at 10b and 184.671 GB at 25b; peak live arena allocations were
59.960 GB and 147.452 GB. These are host allocator statistics, not whole-system
or GPU memory measurements. The original single-GPU 5b program also matched
every reference digit in a fresh 19.621 s regression run.

After the 50b extension, final full comparisons passed at all three sizes:
10b in 20.370 s, 25b in 69.778 s, and 50b in 205.403 s with profiling enabled.
The 50b run's registered-arena peak was 449.128 GB, and peak live arena
allocation was 345.270 GB. Concurrent allocation order affects fragmentation:
the fastest tuning run used 362.784 GB registered and 309.562 GB live. Allow
headroom beyond either allocator figure for the process, driver, reference
files, and operating system.


The 50b validation includes an actual integer exceeding `INT_MAX` 64-bit
limbs, exact signed arithmetic comparisons with GMP, high-index GPU subtrees,
and complete reference-digit comparison. Large multiplication microbenchmarks
also exercise results exceeding `UINT_MAX` 64-bit limbs and check three modular
fingerprints. A tested five-product Toom split was slower than Karatsuba at
50b operand sizes (about 13.4 s versus 8.5 s) and was removed.
