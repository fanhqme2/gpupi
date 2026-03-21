# gpupi

CUDA/GMP implementation of Chudnovsky binary splitting for computing decimal digits of pi.

## Build

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
- `--full-print`: prints the full `RET = ...` and `DEC = ...` output format used for debugging.

## Tests

Each CUDA component has a small standalone test and compile script, for example:

```bash
./batch_arith_test_compile.sh && ./batch_arith_test
./batch_decimal_naive_test_compile.sh && ./batch_decimal_naive_test
./convert_decimal_test_compile.sh && ./convert_decimal_test --skip-full
```

## Performance

Measured on this machine with:

```bash
./pi_bs_gpu 1000000000 --benchmark
```

Observed result:

```text
target_digits=1000000000 compute_digits=1000341504 RET_len=103847955 elapsed_ms=4723.391 workspace_max=6144MB
```

So the current end-to-end 1e9-digit run, including decimal conversion, is about `4.72 s`.
