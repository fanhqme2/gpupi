# gpupi

CUDA implementation of Chudnovsky binary splitting for computing up to **5,000,000,000 fractional decimal digits of pi**, with exact 96-bit NTT multiplication.

The 1b case runs on the 32 GB RTX 5090. The 2.5b and 5b cases target the 96 GB RTX Pro 6000.

September 2026 RTX 5090 medians; y-cruncher figures are the previously recorded comparison.

| digits | gpupi | y-cruncher |
|---|---|---|
|1,000,000,000 | 3509.724 ms | 5149 ms |
|500,000,000 | 1633.047 ms | 2499 ms |
|250,000,000 | 801.277 ms | 1288 ms |


## Build

This repo needs CUDA 13

Compile the main binary:

```bash
make pi_bs_gpu
```

This produces `./pi_bs_gpu`.

## Usage

Default mode writes only the decimal digits of pi to `stdout` and writes the timing/stat line to `stderr`.

```bash
./pi_bs_gpu <digits>
```

Example:

```bash
./pi_bs_gpu 1000 > pi_1000.txt
```

Other modes:

```bash
./pi_bs_gpu <digits> --benchmark
./pi_bs_gpu <digits> --profile-stages
./pi_bs_gpu <digits> --full-print
```

- `--benchmark`: prints a short benchmark summary plus `RET`/`DEC` previews.
- `--profile-stages`: prints per-stage timings.
- `--full-print`: prints both hexadecimal and decimal representation

## Tests

Each CUDA component has a small standalone test and compile script, for example:

```bash
./batch_arith_test_compile.sh && ./batch_arith_test
./mod96_test_compile.sh && ./mod96_test
./shared_mul_test_compile.sh && ./shared_mul_test
./exactdiv_chain_test_compile.sh && ./exactdiv_chain_test
python3 generate_common_factors.py --check
./batch_mul_ntt_test_compile.sh && ./batch_mul_ntt_test --skip-benchmark
./batch_decimal_naive_test_compile.sh && ./batch_decimal_naive_test
./convert_decimal_test_compile.sh && ./convert_decimal_test --skip-full
```

## Performance

The RTX 5090 results above use CUDA 13.1 and the default `-O3` build, with
one warmup and five serial measurements at each precision. The 1b median is
14.7% lower than the previous 4115.181 ms result. Raw trials and
full-output checks are recorded in [benchmark_results.json](benchmark_results.json).

The same production code was previously measured on the RTX Pro 6000 server
(CUDA 13.0, default `-O3`):

| digits | median | best | measured runs |
|---|---:|---:|---:|
| 1,000,000,000 | 3213.666 ms | 3171.524 ms | 3 |
| 2,500,000,000 | 9600.286 ms | 9395.885 ms | 3 |
| 5,000,000,000 | 20297.523 ms | 20291.771 ms | 5 |

Each set has one warmup. All five 5b trials are below the
[22.377-second CPU chart entry](https://www.numberworld.org/y-cruncher/benchmarks/charts/5b.html).
This is a computation-time comparison, not an official benchmark submission.

The implementation shares forward transforms between products, recovers
wrapped tails for smaller cyclic NTTs, cancels common factors through 136-term
blocks, and uses exact Pell doubling for the square root. The modular kernels
use PTX carry chains, shift-based inverse normalization and shorter canonical
corrections. Digit counts, bit counts, shifts and workspace offsets cover 5b.

All 1b, 2.5b and 5b fractional digits from this production code match independently
verified reference checksums. The port is additionally checked at 1b on the RTX
5090. GMP tests cover modular arithmetic, shared products and division chains;
the 96 GB server also passes complete maximal-product checks at NTT size `2^31`.

Repeat the 1b benchmark and full-output check without storing a large file:

```bash
python3 benchmark_pi.py --digits 1000000000 --warmups 1 --runs 5
python3 check_pi.py --sha256 \
  ef0a513ee398d5dbe1ed6994c222e03a16fcfe91f5de685691953d35961798e5 \
  1000000000
```

On a 96 GB GPU, use `--digits 5000000000` for the benchmark. The complete
5b fractional digest is
`435c9b2f0794ae48141c831500c2aa7f4464e68823916012e7128df27cfe0a72`.

`elapsed_ms` includes computation and decimal conversion, but excludes initial
CUDA setup and final output transfer/printing. `workspace_max` is the arithmetic
workspace capacity, not total peak GPU memory.

## Limitations

Some of the algorithms (inversion, decimal conversion) are specifically designed for Pi computation and will not generalize to other general multi-precision computation.

Product lengths are checked through `2^31` 32-bit limbs. At this limit, each
coefficient is at most `2^30 * (2^32 - 1)^2 < 2^94`, below the modulus
`2^96 - 2^32 + 1`; 30-bit repacking or an extra CRT modulus is unnecessary.
The full 2.5b/5b working sets exceed the RTX 5090 device memory.
