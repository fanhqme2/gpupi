# Multi-GPU validation artifacts

The benchmark JSON files in the project root contain individual
serial trials, complete process wall times, source SHA-256 digests, GPU details,
environment overrides, and raw program logs. The computation timing scope is
defined in each JSON and in `MULTI_GPU.md`.

- `pi-10b.log` and `pi-25b.log`: complete byte comparisons with independently
  generated y-cruncher references on the eight-GPU server.
- `pi-5b-single.log`: full original single-GPU regression against the reference.
- `arithmetic.log` and `forced-paths.log`: GMP arithmetic comparisons, including
  high-index subtrees, unequal operands, high products, and reciprocal refinement.
- `ntt32-window.log`: every retained limb of maximal-value products and squares
  using the full 2^32-point transform and cyclic-tail correction.
- `memcheck-final.log`: forced distributed/cyclic arithmetic tests under CUDA
  Compute Sanitizer on the local RTX 5090, with zero memory errors.
- `memcheck-reservation.log` and `pi-10m-reservation.log`: overlapping host
  reservation with GPU subtrees, under Compute Sanitizer, with a full digit hash
  comparison and zero memory errors.
- `cyclic-windows-local.log`: random cyclic products and windows, including
  windows wholly within the corrected high tail.
- `pi-1b-local.log`: all one billion digits checked against a verified hash.
- `build-final.log`, `build-v25.log`, `source-sha256.json`, and hardware text files: build and
  reproducibility details. Source hashes describe the shared arithmetic sources;
  local and server executables use their respective CUDA compilers.

`development/` preserves selected earlier full comparisons and microbenchmarks.
Those exploratory single trials are not the final repeated performance results.
The 185 GB of saved decimal output remains on the server; see
`../results/manifest.json` for paths, sizes, and fractional-digit hashes.

`50b/` records the host-integer extension: large limb-count tests, GMP
comparisons, CUDA memcheck, complete 50b reference comparisons, multiplication
microbenchmarks, and reservation/CPU-affinity trials. The older root-level
10b/25b records describe the preceding release and retain its source hashes.

- `50b/pi-50b-final.log`: all 50 billion digits compared against y-cruncher
  using the final automatic settings.
- `50b/pi-10b-regression.log` and `50b/pi-25b-regression.log`: full reference
  comparisons after the host-integer extension.
- `50b/wide-size-final.log`: an actual integer above `INT_MAX` 64-bit limbs,
  with exact shifts, additions, subtractions, and modular checks.
- `50b/host-final.log` and `50b/arithmetic-final.log`: host and GPU arithmetic
  checks, including the high term indices required at 50b.
- `50b/memcheck-v6.log`: CUDA Compute Sanitizer reports zero memory errors.
- `50b/source-sha256.json`, `50b/server-executables.sha256`, and
  `50b/build-final.log`: final source, binary, and build identification.
- `../benchmark_mgpu_50b.json`: three serial 50b trials with the final defaults,
  profiling disabled, and no warmup. Its source hashes match `50b/source-sha256.json`.

The `reserve*.json`, `v6-affinity*.json`, and `verify-v*.log` records are tuning
trials. The Toom microbenchmark describes a tested candidate that was removed;
the final implementation uses Karatsuba above the transform-size limit.

`100b/` records the 64-bit term-factor extension, the full 100b comparison,
memory measurements, and subsequent allocator/scheduling comparisons. Its
[README](100b/README.md) explains the individual files. The 10b/25b/50b release
records retain their historical source and artifact hashes; the original
snapshots are preserved in `.scratch/` on the server.

The final 100b build passed a complete byte comparison in
`100b/pi-100b-final.log`. `../benchmark_mgpu_100b.json` records one final
profiling-disabled run with automatic defaults and hashes matching
`100b/source-sha256.json`. `100b/memory-final.json` records resource use
during the complete verification run.
