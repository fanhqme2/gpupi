# gpupi

CUDA implementation of Chudnovsky binary splitting for computing (up to) 1,000,000,000 decimal digits of pi, with 96-bit NTT multiplication algorithm.
With RTX 5090, it breaks several records as listed in https://www.numberworld.org/y-cruncher/#FastestTimes

September 2026 gpupi medians; y-cruncher figures are the previously recorded comparison.

| digits | gpupi | y-cruncher |
|---|---|---|
|1,000,000,000 | 4115 ms | 5149 ms |
|500,000,000 | 1935 ms | 2499 ms |
|250,000,000 | 947 ms | 1288 ms |


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
./batch_mul_ntt_test_compile.sh && ./batch_mul_ntt_test --skip-benchmark
./batch_decimal_naive_test_compile.sh && ./batch_decimal_naive_test
./convert_decimal_test_compile.sh && ./convert_decimal_test --skip-full
```

## Performance

On an RTX 5090 with CUDA 13.1 and the default `-O3` build, the billion-digit
benchmark improved from 4830.578 ms to 4115.181 ms (14.8% less time). These are
five-run medians with one warmup per executable and alternating run order.

The implementation fuses local NTT transforms and pointwise multiplication,
reuses forward transforms for squares, switches from naive multiplication to
NTT above 512 product limbs, and reuses scratch buffers during decimal conversion.
It also fixes a reciprocal allocation leak and insufficient precision at small
digit counts.

All billion fractional digits matched a local reference. GMP component tests,
30 digit-count boundary cases, and CUDA memory checks on smaller runs passed.

Measured on this machine with:

```bash
./pi_bs_gpu 1000000000 --benchmark
```

Median benchmark result:

```text
target_digits=1000000000 convert_digits=1000341504 RET_len=103812502 elapsed_ms=4115.181 workspace_max=6144MB
RET = 3243f6a88...9218acb27b5a248c hash = 987023668
DEC = 3141592653589793238462643383279502884197169399375105820974944592...1990548327874671398682093196353628204612755715171395115275045519
```

`elapsed_ms` includes computation and decimal conversion, but excludes initial
CUDA setup and final output transfer/printing. `workspace_max` is the arithmetic
workspace capacity, not total peak GPU memory.

## Limitations

Some of the algorithms (inversion, decimal conversion) are specifically designed for Pi computation and will not generalize to other general multi-precision computation.

The NTT kernel works only up to 2^28 limbs of product length, which is enough for computing 1b digits of pi, but no more.
