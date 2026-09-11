# 100-billion-digit validation

The 100b extension widens global term indices and denominator factors, while
retaining the original leaf storage and bounded NTT engine. The 100b run uses
7,130,316,800 Chudnovsky terms.

- `arithmetic-v1-local.log` and `arithmetic-v1.log`: exact host/GPU checks.
  Both kinds of leaf are checked against GMP below, across, and above 2^32
  terms. Complete subtrees include block offsets 419,430,368 and 500,000,000.
  Invalid ranges, including `UINT64_MAX`, are rejected before allocation.
- `memcheck-local.log`: Compute Sanitizer with zero memory errors.
- `pi-1b-local.log`: full 1b multi-GPU-program regression on the local card.
- `pi-5b-single.log`: full 5b original-program regression on the server.
  This overlapped CPU reference generation; its timing is not an isolated
  performance comparison.
- `source-v1-sha256.json`: exact arithmetic sources and Makefile for the
  initial 640 GiB reservation configuration.
- `ycruncher100b.log`: independent reference generation, verification enabled.
- `pi-100b-v1.log`: initial full GPU computation and streaming comparison.
- `memory-v1.json`: process host high-water mark and GPU memory samples.
  GPU values include the driver baseline and can miss short allocation peaks.
- `reserve-probe.json`: single 512 GiB and 640 GiB registrations fail on the
  measured driver; 384 GiB succeeds.
- `register-crossing.log`: an 8 KiB transfer crossing two adjacent registered
  ranges fails, while transfers within either range succeed.
- `v2-parallel.log`: the failed contiguous-arena workaround; this candidate
  was replaced with separate registered arenas.
- `reserve640-arenas.log` and `reserve768-arenas.log`: large-pool validation
  using separate arenas, with endpoint transfers checked on all eight GPUs.
- `cuda-reservation-v3.log` and `memcheck-reservation-v3.log`: CUDA memory
  checking and full 10m digit comparison with the revised reservation path.
- `v3-parallel.json` and `v3-serial32.json`: full-size scheduling comparisons.
  Matching `memory-v3-*.json` files record resource use; `.stderr.log` files
  retain live stage output in addition to the JSON copies.

- `v4-products.json` and `memory-v4-products.json`: sequential product
  collection with parent threshold 16, the slower but lower-memory option.
- `cuda-products-v4.log` and `memcheck-products-v4.log`: full 10m comparison
  under CUDA memory checking for the optional product schedule.
- `source-v3-sha256.json` and `source-v4-sha256.json`: tuning source snapshots.

- `pi-100b-final.log`: all 100 billion digits match y-cruncher using final
  automatic defaults; computation 615.617 s with profiling enabled.
- `memory-final.json`: final full-run process high-water mark
  1,181,742,465,024 bytes, with at least 552,510,533,632 bytes of system RAM
  available in the samples. Allocation order affects reuse, so the earlier
  tuning peak is not a fixed memory bound.
- `arithmetic-final.log` and `arithmetic-final-local.log`: final build checks.
- `source-sha256.json`, `build-final.log`, and `server-executables.sha256`:
  final arithmetic sources, build output, and executable identification.
- `../../benchmark_mgpu_100b.json`: one final run, no warmup, profiling
  disabled, and no GPUPI environment overrides. Source hashes match
  `source-sha256.json`. The complete process prints digit previews.

The tuning runs used profiling and explicit scheduling overrides. Their
previews were checked against the reference; full byte comparisons are
identified separately. Reported GB uses decimal units, while reservation
settings use GiB. The optional lower-memory schedule reduced process host
high-water mark to 742,054,227,968 bytes.

Arithmetic timings include registered-memory reservation, all computation,
decimal conversion, and decimal D2H. They exclude CUDA context startup,
printing, reference comparison, and teardown. Saved digit files are retained
on the server, not in this validation directory.
