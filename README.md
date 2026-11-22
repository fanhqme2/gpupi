# gpupi

gpupi computes up to **5,000,000,000 fractional decimal digits of pi on a 32 GB RTX 5090** using CUDA, Chudnovsky binary splitting and exact 96-bit number-theoretic transforms (NTTs).

## RTX 5090 performance

| Fractional digits | Median (ms) | Best (ms) |
|---:|---:|---:|
| 250,000,000 | 620.085 | 619.684 |
| 500,000,000 | 1,304.498 | 1,299.332 |
| 1,000,000,000 | 2,643.161 | 2,638.889 |
| 2,500,000,000 | 8,245.625 | 8,243.835 |
| 5,000,000,000 | 20,777.980 | 20,752.328 |

Measured November 22, 2026 with CUDA 13.1, driver 610.88 and the default `-O3` build. Each size had one warmup and three measured runs in separate processes, using `./pi_bs_gpu <digits> --benchmark`. The timer includes computation and decimal conversion. CUDA startup, final host transfers and printing occur outside it. The complete 1b, 2.5b and 5b outputs matched the independent reference hashes. [Raw measurements](benchmark_results.json)

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

## Large computations

The 2.5b and 5b runs evaluate the Chudnovsky series in batches of up to 71,303,168 terms. Each batch retains only the limbs needed at its position in the series. Later batches need fewer limbs because each preceding term attenuates their contribution by at least 14.18 decimal digits. During a merge, two products can share a forward transform, and the sum of two products can use a single inverse transform.

For requests above one billion digits on GPUs with at most 40 GiB, multiplications that exceed the 6 GiB transform scratch budget use smaller exact products. Their limbs are added at the appropriate offsets with carry propagation. The limit applies to individual transforms. The reported `workspace_max` also includes reserved scratch, and the full computation uses additional device memory for operands and decimal digits.

To repeat the 5b benchmark and verify the entire output without storing a multi-gigabyte file, run

```bash
python3 benchmark_pi.py --digits 5000000000 --warmups 1 --runs 3
python3 check_pi.py --sha256 \
  435c9b2f0794ae48141c831500c2aa7f4464e68823916012e7128df27cfe0a72 \
  5000000000
```

## Earlier measurements

The September 2026 build on the 96 GB RTX Pro 6000 with CUDA 13.0 recorded medians of 3,213.666 ms for 1b, 9,600.286 ms for 2.5b and 20,297.523 ms for 5b. The September RTX 5090 build took 3,509.724 ms for 1b. Earlier y-cruncher comparison times were 1,288 ms for 250m, 2,499 ms for 500m and 5,149 ms for 1b. The GPU trials and the November optimization profiles are in the [benchmark history](benchmark_results.json).

## Limitations

The reciprocal and decimal conversion routines are specialized for pi. Each NTT product supports up to `2^31` 32-bit limbs. At that limit the maximum coefficient is `2^30 * (2^32 - 1)^2 < 2^94`, within the modulus `2^96 - 2^32 + 1`.
