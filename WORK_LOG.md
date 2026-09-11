# 2.5 billion digit implementation work

All source changes and generated files are confined to batchmp2; batchmp is untouched.
Baseline commit: 9531687.

## Arithmetic range

The NTT prime is p = 2^96 - 2^32 + 1. For a product of at most 2^31
32-bit limbs, the shorter operand has at most 2^30 limbs, so every
coefficient is bounded by 2^30 (2^32 - 1)^2 < 2^94 < p. There is no
coefficient overflow at 2.5 billion decimal digits. Repacking to 30-bit
chunks or adding a CRT modulus would add work without improving exactness
in this range. The old README's 2^28 restriction needs experimental audit,
not a change of modulus.

Initial changes widen bit counts, shifts, square-root workspace byte counts,
and decimal conversion precision. Limb counts remain 32-bit with a checked
2^31 product-length limit. CLI accepts 0 through 2,500,000,000 digits.

## Validation so far

- Generated all 2.5 billion fractional reference digits by running the sibling
  y-cruncher v0.8.7 build 9547, custom Pi, decimal output, verification enabled.
  Output is under .scratch/reference. y-cruncher computation time 44.268 s;
  complete run including output and verification 73.607 s.
- Existing GMP-backed arithmetic and NTT correctness suites pass.
- Every limb of 2^29-element NTT products of maximal 32-bit limbs passes:
  square, separate operands, unequal lengths. These cases exercise worst-case
  coefficient magnitude and long carry propagation. 64-bit bitlength passes.
- Streaming reference comparison matches all 1,000,000 fractional digits.
- Full 1-billion comparison also passes: fractional-digit SHA-256
  ef0a513ee398d5dbe1ed6994c222e03a16fcfe91f5de685691953d35961798e5.
  Local computation + decimal conversion was 4037.178 ms.

- Server maximal-limb tests pass every limb at 2^30 and 2^31 transform
  elements. The exact binary-splitting root actually needs 2^31 elements
  (R*Q operands 578,989,960 and 512,143,239 limbs); this is a transform
  padding cliff, not coefficient overflow. The 96-bit coefficient bound
  still has at least two spare bits over the supported range.
- First complete server 2.5b profile: 22,633.990 ms. Decimal conversion
  1,741.383 ms; arithmetic workspace 48 GiB. Many R*Q products cross a
  power-of-two padding boundary by only a few percent.
- All 2,500,000,000 fractional digits of the initial implementation match
  y-cruncher. SHA-256 of fractional digits:
  5753752acbbf07c3414331c3b9121b16dd910e426a5eb014d0b8f31a78da4059.
  Unprofiled time: 22,278.529 ms.

## Cyclic tail and term-count experiments

A product with L just above power-of-two K can use a K-element cyclic NTT.
The wrapped coefficients are recovered from the low coefficients of the
product of the reversed high (L-K-1) limbs. They are subtracted modulo p
before carries, and appended above K. This preserves exact integer products
while avoiding the 2K padding cliff. GMP tests through K=2^24 pass squares,
maximal limbs, random inputs, broadcasts, strides, and tails of 1, 2, 31,
32, 33, K/32 and K/8 limbs. GPUPI_LINEAR_NTT disables the experiment.

Two subtrees of 2^23 and 2^21 blocks require 178,257,920 terms for 2.5b,
versus 285,212,672 in the original single tree. A root merge uses the exact
existing P/Q/R identities, including the implicit power-of-two scale.
GPUPI_SINGLE_TREE restores the original tree. Smaller full-digit checks
(250,000 and 2,500,000) pass against y-cruncher.
- Cyclic tail alone: 19,362.414 ms, same full binary-result modular hash
  and decimal previews as the verified baseline.
- Two subtrees + cyclic tail: 13,194.519 ms; 25m and 250m complete reference
  comparisons pass. The changed binary result differs only below the
  requested decimal precision, as expected from the changed term count.
- Large transforms (k>=29) now use forward 8+8+5 stages and backward
  5+8+8 with a fused local product/inverse. Local k=29 maximal-limb tests
  pass, reducing multiplication from 193.5 to 119.3 ms.
- First server result with the new large schedule: 12,233.738 ms.
  Repeated timings and full optimized-output verification remain to be done.
- Final large schedule passes every limb on the server at k=31 for
  squares, ordinary/unequal multiplication, and a cyclic-tail case.
  Shifts of 2^33+5 bits and INT64_MIN/INT64_MAX edge cases pass.
- Cyclic carry tests also pass with 257 rows, including broadcast inputs,
  the single-block carry path, and multiple workspace batches.
- Full existing NTT, arithmetic, decimal and bitlength component suites pass.
  26 decimal boundary counts (0 through 10m) and 5 invalid CLI inputs pass.
  CUDA memcheck of a 250,000-digit forest/cyclic run reports zero errors.
- Native sm_120 build trial: 12,412.045 ms versus the default build's
  12,233.738 ms. Keep the original build target pending repeated timings.
- Final five-run server batch, after one warmup: 12,350.459, 12,190.312,
  12,298.887, 12,264.063, 12,268.798 ms. Median 12,268.798 ms;
  all five runs are below 12,500 ms. Raw results: benchmark_results.json.

## Final outcome

The final optimized server executable matched every one of 2,500,000,000
fractional digits byte for byte against y-cruncher. SHA-256:
5753752acbbf07c3414331c3b9121b16dd910e426a5eb014d0b8f31a78da4059.
The full-output verification run reported 12,400.733 ms before transfer/output.
The separate five-run timing median is 12,268.798 ms (best 12,190.312 ms),
with every measured trial below the cited 12.5-second target.

Temporary reference digits, experiment logs, tuning builds and test binaries
have been removed. Sources, reproducible test/benchmark scripts, benchmark
results and Git milestone history remain. The sibling batchmp was never edited.

## Further optimization pass

Starting from verified commit 474d751 (12,268.798 ms median).
The new profile has about 10.18 s of multiplication and 1.63 s of decimal
conversion. Initial cyclic-aware workspace batching / cache-budget trials:
2m coefficients 12,814.281 ms; 3m 12,134.438; 4m 12,282.662;
6m 12,027.092; 8m 12,066.879. All binary-result hashes and previews match.
These are exploratory single trials, not final medians.

## Further optimization: shared operands and Pell recurrence

The follow-up optimization pass first swept cyclic-aware workspace batching:
2M/3M/4M/6M/8M cache coefficients gave exploratory 12814/12134/12283/12027/12067 ms.
These are single trials, not final performance claims. A broadcast operand now
has one persistent full transform per multiplication call; its transform is
reused across all rows and workspace batches. The initial 6M-cache server trial
fell to 11770.582 ms, with unchanged binary hash and decimal previews.

The square-root recurrence satisfies P^2 - 10005 Q^2 = 1 exactly, starting with
P=4001, Q=40. Replacing P^2+10005 Q^2 with 2P^2-1 removes Q^2 and its associated
small multiplication. P remains odd, so the subtraction in the fused shift
kernel never borrows beyond limb zero. Workspace reservations were reduced
accordingly. Combined exploratory server trials: 11520.862, 11558.300,
11834.317 ms (warmup 11723.156); median 11558.300 ms. Full final validation and
controlled performance trials follow after schedule tuning.

Large broadcast-transform reuse passed the complete GMP multiplication suite,
including cyclic tails, identical pointers, broadcast rows, nontrivial strides,
and workspace batch boundaries. Additional exploratory schedule trials (6M
cache, Pell and large broadcasts enabled): k24 forward 5+8: 11568.275 ms;
k24 8+5: 11509.585; k25 inverse 4+8+4: 11658.842; k28 forward 5+8+5:
11722.973; k28 8+5+5: 11483.912. Expanding the cyclic-tail threshold to K/4
gave 11721.361 ms and was not retained. Every trial preserved the full binary
result hash and decimal previews. Schedule decisions require combined trials.

Combined settings (Pell, broadcast reuse through small NTTs, k24 forward 8+5,
k28 forward 8+5+5) were swept again. Cache budgets 6M, 4M, 8M gave pairs
[11459.396,11433.072], [11647.932,11762.009], [11507.802,11811.854] ms.
Disabling only the small-broadcast path at 6M gave [11603.659,11455.423] ms.
Retained 6M, both broadcast paths, and the k24/k28 schedules as defaults;
removed the unsuccessful k25 and wider-tail experiments (available in Git).

The expanded complete GMP suite passes with broadcasts down to k6. Local
analytic full-limb checks pass at k24, k28 and k29, including maximal inputs,
unequal operands, cyclic tails, long carries, and shifts beyond 2^33 bits.
The complete billion-digit output still has the previously verified SHA-256
`ef0a513ee398d5dbe1ed6994c222e03a16fcfe91f5de685691953d35961798e5`.

`check_pi.py --sha256` can now check every output byte against a stored digest
without retaining the multi-gigabyte reference. Existing byte comparison is
preserved. Both modes enforce exact output length and newline. Comparator
checks passed for both valid modes, wrong digest, wrong digit, truncation and
extra output. This makes repeated full-output server validation inexpensive
without transferring 2.5 GB through SSH.

## Follow-up final performance and full-output validation

Final default build (`be9eaf9` arithmetic implementation), CUDA 13.0 on the
RTX Pro 6000 server, no optimization environment overrides:

- Warmup: 11489.126 ms.
- Five serial trials: 11659.386, 11629.848, 11445.973, 11440.569, 11440.968 ms.
- Median 11445.973 ms, best 11440.569 ms, worst 11659.386 ms.
- Median computation time is 6.7% lower than the previous 12268.798 ms median.
- Complete 2.5b output run: 11656.010 ms. Hashing every fractional digit gives
  `5753752acbbf07c3414331c3b9121b16dd910e426a5eb014d0b8f31a78da4059`, exactly
  the y-cruncher reference digest established by the earlier full byte comparison.
  Exact output length and trailing newline also pass.
- 23 Pi boundary counts through 250,000 digits agree with mpmath at extra
  precision; CUDA memcheck at 250,000 digits reports zero errors.
- The local RTX 5090's complete billion-digit output also matches its reference
  checksum. Exploratory warm timings with the new default: 3950.132/3947.197 ms;
  a 4M cache override gives 3927.508/3950.262 ms (no material difference).

The targeted broadcast suite now uses 1025 rows to cross workspace batches
at the larger 6M default, for both ordinary transforms and cyclic tails. The
`--broadcast-only` option isolates these checks and the k6..k24 shared-operand
cases. Timing continues to exclude CUDA startup, output transfer and printing;
it includes the entire arithmetic and decimal conversion computation.

The final focused broadcast suite passes, including all new 1025-row batch
boundary cases. Experiment scripts, logs and temporary test executables were
removed locally after recording these results; the normal build and Pi
executable are retained.

The optimized arithmetic source and executable were deployed to the server
and fully validated there before connectivity failed. Final synchronization of
the Git history/documentation and removal of the server's `.scratch` directory
could not be confirmed: repeated fresh SSH connections timed out during banner
exchange, including a retry with a 30-second connection timeout. Local Git is
clean and all milestones/results are committed. The server executable used for
the recorded trials is the final optimized implementation; repository sync and
remote artifact cleanup remain unconfirmed.


## RTX Pro 6000 billion-digit follow-up

SSH access recovered. The existing final optimized server executable was tested
with one warmup (3767.473 ms) and five serial billion-digit runs: 3866.589,
3780.426, 3740.206, 3776.094, 3749.482 ms. Median 3776.094 ms, best 3740.206 ms,
worst 3866.589 ms. The median is 0.43% above the user's original 3760 ms,
within the variation of these runs. The full-output run took 3839.343 ms;
all one billion fractional digits hash to the verified reference
`ef0a513ee398d5dbe1ed6994c222e03a16fcfe91f5de685691953d35961798e5`, with exact
output length and newline checks passing.

The earlier interrupted archive sync had left remote pi_bs_gpu.cu as exactly
the first 67072 bytes of its 86970-byte committed content. The executable
predated that failed upload and was intact. After checking the exact prefix,
the source was restored atomically from the server's Git commit. Its SHA-256
now matches the local source, the worktree is clean, and git fsck passes.
Final result updates are synchronized through a verified incremental Git
bundle rather than unpacking a live source archive; old remote experiments
are removed after synchronization.

## Five-billion-digit extension: initial range audit

The requested CPU chart's fastest listed 5b computation is 22.377 seconds
(https://www.numberworld.org/y-cruncher/benchmarks/charts/5b.html, fetched
2026-09-10). Server GPU memory is 97887 MiB, initially unused.

Decimal precision, rounded conversion lengths, per-level decimal lengths,
Pell iteration precision and profiling counters now use 64-bit counts. Input
parsing permits up to 5,000,000,000 fractional digits. Limb counts remain
32-bit: about 519 million final limbs, and the largest padded transform is
expected to remain within the already validated 2^31-element limit. The
96-bit coefficient bound remains valid. Leaf factor arithmetic is explicitly
unsigned. The local build and 10,000-digit smoke computation pass; complete
5b validation follows. An independent y-cruncher 5b reference is generating.

The widened implementation completes 5b on the server: first benchmark
25098.055 ms, stage-profile run 25068.271 ms, 48 GiB arithmetic scratch.
An independent y-cruncher v0.8.7 build 9547 run (verification enabled) took
98.810 seconds computation / 158.807 seconds wall time on the local CPU.
The complete five-billion-digit GPU output (25307.137 ms computation) has
SHA-256 `435c9b2f0794ae48141c831500c2aa7f4464e68823916012e7128df27cfe0a72`,
identical to the independently hashed y-cruncher fractional digits. Exact
output length and suffix pass. The 5 GB reference was removed after recording
its digest; subsequent full-output checks reuse this reference checksum.

The stage profile attributes 21284 ms to multiplication and 2775 ms to decimal
conversion. Q/RQ pairs eligible to share a transform account for 7824 ms;
P/PR pairs account for 7393 ms. A shared-operand pair primitive is being tested.

The shared-B NTT primitive passes GMP comparisons through k24, with random and
maximal inputs, two different outputs, cyclic/linear combinations, two wrapped
tails, strides, broadcasts, and 1025-row workspace boundaries. It falls back
when transform lengths differ or caching would require a 2^31 transform.
Initial integration shares Q_odd between Q_even*Q_odd and R_even*Q_odd.

The second integration shares P_even between P_odd*P_even and R_odd*P_even.
The final unused P row remains omitted; its R product is calculated separately.
Ten Pi boundary/smoke counts through 100,000 agree with mpmath after this
reordering. Large server trials will determine whether both integrations stay.

Paired Q/RQ trials: 24291.018 and 24234.590 ms. Adding paired P/PR gave
23686.531 ms at the 6M pair cache budget. Pair budgets 3M/4M/2M/1M gave
23878.281/23744.711/24931.338/26130.713 ms; smaller budgets were not retained.
Loading cached multipliers only at the pointwise stage halves the local shared
memory requirement. This permits 4096-coefficient fused tiles and a cached
k25 schedule aligned with the inverse. GMP checks through k25 pass; server
trials improve to 23126.279 and 23095.359 ms.

Cached products now reserve compact carry summaries instead of an unused full
second-transform scratch buffer. This allows shared 2^31-element transforms
in about 48.5 GiB scratch, and enables Q/RQ sharing in the forest merge. The
GMP paired suite and complete maximal-limb checks at k28 pass with this layout.

The compact shared scratch layout passes full k31 checks on the server, including
both shared-product outputs and long shifts. Subsequent 5b trials are 22908.902
and 23027.811 ms, using the same reference previews.

A further exact reduction is being tested at 34-, 68-, and 136-term blocks.
For primes outside 2,3,5,23,29, every term of R has a P prefix and Q suffix.
Counting their linear factors modulo each prime and taking the minimum over
all split points and all starting residues gives a guaranteed common divisor
of P, Q and R. Higher prime powers only strengthen this bound. Subtracting the
already-cancelled child factors gives extra divisors; the generator records
this conservative proof. Small Pi comparisons pass through 100,000 digits for
all three cutoffs. The existing exact-divider's artificial 128-limb ceiling
was raised to 1024 for these blocks, with larger GMP regression cases added.

Cancellation cutoff trials: 34 terms 22987.871 ms, 68 terms 23072.986 ms,
136 terms 22473.881 ms. The 136-term variant is within 0.1 seconds of the CPU
chart target. A 272-term variant adds 22 packed divisors; the generator now
checks all CUDA constants directly with --check. Its smoke result passes and
the scalar exact-divider bound is 2048 limbs to cover its larger rows.

The 272-term cutoff regressed to 23190.467/23115.852 ms, so the 136-term
approach remains preferred. The full arithmetic API suite and enlarged GMP
exact-division suite pass; CUDA memcheck at one million digits reports zero
errors even with the 272-term cancellation enabled. Next experiments align
cached forward tiles at k16/k17/k27 with their inverse tiles; the default
schedule is preserved unless these focused comparisons show a benefit.

Focused cached-schedule measurements (ms per paired multiplication batch):
k16 old 14.559/14.547, aligned 13.782/13.756; k17 old 15.658/15.643,
aligned 15.171/15.155; k27 old 52.012/50.996, alternative 53.113/52.600.
Retain k16/k17 alignment, reject k27. Full 5b runs at the 136-term cutoff
with both alignments: 22447.393 and 22336.057 ms. The best is below the CPU
chart, but the median is 22391.725 ms. CUDA eager module loading was also
checked (NVIDIA documents its influence on timed initialization); it did not
help here: 22533.686/22494.609 ms. Normal module-loading behavior is retained.

A fused exact-divider now pipelines the 4-, 11-, or 22-divisor chains through
register carries, avoiding global writes/reads of intermediate quotients.
Its GMP suite passes chain lengths 0,1,2,4,11,22,32, rows 1,33,65, and lengths
64,129,512,2048, including output padding and invalid arguments. Pi checks
through 100,000 digits pass at all active cancellation cutoffs.

Fusing the division chains improved the 136-term trials to 22112.420 and
22020.840 ms, both below the 22377 ms CPU chart entry. The fused 272-term
variant gave 22064.168/22388.457 ms and is not retained. The final defaults
select cancellation through 136 terms and cached k16/k17 alignment. The
unsuccessful 272-term table and k27 schedule experiment are removed; their
committed history preserves the comparison. Final full-output and repeated
benchmark validation follows on these defaults.

### Final 5b validation and benchmark (d9ba4ea)

The default CUDA 13.0 -O3 server build completes all five billion digits.
One warmup: 22164.463 ms. Five serial trials: 22117.398, 22350.424,
22124.688, 22047.592, 21998.420 ms. Median 22117.398 ms, best 21998.420,
worst 22350.424. Every measured trial beats the fastest cited CPU chart
entry, 22377 ms (dual AMD Epyc 9R14). The median is 1.16% below it.
Timing includes arithmetic and decimal conversion and excludes initial CUDA
setup, final host transfer, hashing and printing; this is not an official
benchmark submission.

The final complete 5b output matches the independently generated y-cruncher
SHA-256 435c9b2f0794ae48141c831500c2aa7f4464e68823916012e7128df27cfe0a72.
The comparator checks all 5,000,000,000 fractional bytes, exact output length,
and newline. That full-output run computed in 21975.834 ms.
Full 1b and 2.5b outputs also match their previously verified SHA-256 digests.
Their full-output computation times were 3502.571 and 10391.416 ms.

Smaller-size performance with the same final defaults:
- 1b warmup 3527.265 ms; trials 3568.156, 3561.273, 3545.772 ms;
  median 3561.273 ms (previous measured median 3776.094 ms).
- 2.5b warmup 10389.528 ms; trials 10369.610, 10282.135, 10392.036 ms;
  median 10369.610 ms (previous measured median 11445.973 ms).

Final local validation adds 23 mpmath boundary comparisons from zero through
100,000 digits, rejects four invalid precision arguments, and runs CUDA
memcheck at one million digits with zero errors. Common-factor proofs pass
for all retained blocks. The component GMP, k31 maximal-product, carry and
large-shift checks are recorded above and in benchmark_5b_results.json.

The final arithmetic commit is installed and built on the server. Completed
experiment binaries, profiles and the multi-GB reference output are removed;
only normal build outputs remain after recording these results. No sibling
batchmp files were changed. README documents the 5b commands and all retained
comparison switches; benchmark_5b_results.json records raw trials and hashes.

### Further NTT optimization experiments

Fresh 5b profile at 675ba70: 22016.732 ms total, 18315.250 ms multiplication,
2707.179 ms decimal conversion. A 96-bit multiply experiment uses explicit
PTX multiply-add carry chains for the six-limb product, preserving the existing
modulus reduction. The modular primitives are factored into ntt_mod96.cuh for
independent GMP testing. The test checks 131072 input pairs for multiplication,
addition and subtraction, including values close to the modulus. It exposed
and helped fix an initially omitted carry into the sixth product limb.
Both variants now pass. Local throughput trials favor PTX (0.421–0.448 ms)
over the original (0.499–0.509 ms); full server measurements will decide.

PTX full-server trials: warmup 21679.756 ms; measured 21321.809,
21368.951, 21589.230 ms (median 21368.951). The complete 5b SHA-256
matches the independent reference; that check computed in 21223.672 ms.
PTX is now the default, with GPUPI_OLD_MUL96 as a compile-time comparison.

The next experiment precomputes inverse root tables in an additional half of
each root allocation (1.5 MiB extra total). Backward butterflies use positive
bit-reversed indices into inverse powers, avoiding per-butterfly bit reversal
and negation. Correction to the initial hypothesis: reversing a negated
reversed index preserves its leading-bit position, so the old lookup also
skips the second root multiplication for small indices. Mixed forward/inverse kernels
select the appropriate half for each phase. GPUPI_OLD_INVERSE_ROOTS restores
the old lookup at compile time. Table allocation sizes are named in the API.

Inverse-table trials were 21248.670 and 21472.605 ms, median 21360.6375,
indistinguishable from PTX alone. The extra tables are removed. Instead, exact
inverse normalization now computes (x+t*p)/2^k using shifts and carries,
where t=-x mod 2^k and p=2^96-2^32+1. This replaces a full modular multiply
for each coefficient. GMP covers k=0..32; the shared multiplication suite
through k25 passes. GPUPI_OLD_NTT_SCALE restores multiplication by the scalar
at compile time for comparisons.

The next modular primitive experiment simplifies canonical addition and
subtraction. For canonical inputs, the wrapped addition correction cannot
overflow a second time; an unwrapped sum is at least p exactly when its top
two limbs are all ones and its low limb is nonzero. Both add/subtract corrections
are a single low-limb +/- (2^32-1), with carry propagation. GMP passes all
131072 cases; scaling boundary inputs now exercise every exponent 0..32.
GPUPI_OLD_ADDSUB96 retains the old code as a compile-time comparison.

Shift-normalization server trials: 21218.953 and 21142.461 ms, median
21180.707 ms (warmup 21057.100). A further reduction experiment replaces
the final modulus comparison/correction with the same canonical interval
check. The second fold adds less than 2^64, so a wrapped result cannot also
lie in the >= p interval. GMP checks pass; GPUPI_OLD_REDUCE96 selects the
previous correction for compile-time comparisons.

Shorter add/subtract corrections: server trials 20771.098/20594.504 ms,
median 20682.801 (warmup 20561.977). The complete shared multiplication suite
passes for both the addition and final-reduction changes.

The shorter final reduction improves the server trials to 20338.570 and
20219.432 ms (median 20279.001; warmup 20165.027). Before final validation,
one compiler experiment marks the read-only root parameters const/restrict;
they never alias writable transform data. This may improve read-only caching
and load scheduling without changing the transform arithmetic.

The reduction version matches every fractional digit of the 5b reference
(full-output computation 20123.324 ms). Read-only/nonaliasing root trials are
20279.445 and 20158.801 ms, median 20219.123; warmup 20240.506. This small
difference is within run variation, but the annotations correctly express the
existing no-alias invariant and are retained. The focused shared GMP suite
passes. Final repeated benchmarking and all-size output checks follow with
PTX multiplication, shift normalization and shorter canonical corrections.

### Final modular-arithmetic optimization results (2123917)

All final server checks pass with default CUDA 13.0 -O3 compilation:
- 5b warmup 20298.111 ms; five trials 20296.213, 20297.523, 20665.990,
  20381.312, 20291.771 ms. Median 20297.523, best 20291.771, worst 20665.990.
  The median is 8.23% lower than the previous 22117.398 ms result and 9.29%
  below the cited 22377 ms CPU chart entry. This pass did not reach 20 seconds.
- 1b warmup 3206.405 ms; trials 3388.219, 3171.524, 3213.666 ms;
  median 3213.666 ms (previous 3561.273).
- 2.5b warmup 9444.552 ms; trials 9600.286, 9395.885, 9677.284 ms;
  median 9600.286 ms (previous 10369.610).

The complete fractional SHA-256 matches at 5b, 1b and 2.5b, with exact output
length and newline checks. Their full-output computation times were
20362.848, 3309.063 and 9542.782 ms respectively. Reference digests are
unchanged and recorded in benchmark_5b_results.json.

The final local full NTT correctness suite passes, covering random sizes,
identical operands, broadcast, strides, carries and multiple batches. The
shared-product GMP suite passes through k25. There are 21 passing small Pi
boundary checks after the arithmetic changes and zero CUDA memcheck errors
at one million digits on the final local build. Direct GMP testing of 131072
modular primitive cases passes on both GPUs, including every scaling exponent
0..32 at each canonical boundary. On the server, k31 checks every limb of
maximal squares, products, unequal and cyclic products, both shared outputs,
and shifts above 2^33 bits; all pass.

Retained changes are PTX carry-chain multiplication, exact shift normalization,
shorter canonical modular corrections and read-only/nonaliasing root pointers.
The inverse-root table experiment was removed after showing no meaningful
gain. Binary splitting, sqrt and decimal-conversion algorithms are unchanged
in this optimization pass. The existing timing scope is preserved. Results,
comparison switches and reproduction commands are documented; experiment
outputs and binaries are cleaned from both workspaces after recording them.

### RTX 5090 1b check after modular optimizations (8c7f85a)

Local RTX 5090, driver 591.86, CUDA 13.1 nvcc V13.1.115, default -O3.
The existing build is up to date. One warmup: 3524.498 ms. Five serial trials:
3509.651, 3514.000, 3531.918, 3518.641, 3520.995 ms. Median 3518.641,
best 3509.651, worst 3531.918 ms; about 14.2% below the original user-reported
4100 ms. The full-output run took 3533.709 ms and matches the SHA-256 of all
one billion fractional digits, with exact length/newline checks. Timing scope
is unchanged: arithmetic and decimal conversion, excluding initial CUDA setup
and final transfer/hash/printing. Results recorded; temporary logs removed.

## Multi-GPU extension (September 11, 2026)

Implemented `pi_mgpu` alongside the original executable. The driver schedules
exact GPU subtrees, then merges independently scaled P/Q/R mantissas with 512
guard bits. High-index leaf factors use 64-bit arithmetic and wider P/Q
storage; nonzero-offset subtrees also need larger temporary backing arenas
before common-factor cancellation. Small exact GMP subtree checks and CUDA
memcheck caught and verified the fixes for those high-index storage bounds.

The 96-bit prime supports transforms only through 2^32. A six-step distributed
NTT reaches that limit across eight GPUs. Cyclic convolution with an exact
reversed-tail correction handles the 25b high products without adding CRT
primes. Other shapes retain Karatsuba. Exact cross-device carry maps and
high-product output windows preserve correctness without constructing a GMP
integer larger than its signed-int limb-count limit. See `MULTI_GPU.md` for
the coefficient bound and algorithm description.

Host-memory work proved critical. Parallel page initialization followed by
CUDA registration was much faster than cudaHostAlloc. With eight contexts,
registration throughput was about 10–11 GB/s versus about 2.5 GB/s for
cudaHostAlloc; with one context, registration reached 43–47 GB/s. Reusable
arenas and direct high-limb downloads reduce both registration and transfers.
Fixed-size power-of-two allocation blocks were subsequently replaced by an
optional 64 KiB size granularity, with adjacent-block coalescing.

The GPU topology is PCIe across two NUMA sockets, without NVLink. Individual
256 MiB peer transfers measured roughly 29–38 GB/s; pinned H2D/D2H measured
56–57 GB/s. Concurrent transfer experiments favored separate copy streams.
Explicitly enabling permanent peer mappings made copies slower on this
server: a 2^31-point product dropped from about 1.07 s to 0.51 s when CUDA's
peer-copy API handled the transfers without those mappings. Both variants
passed every limb of maximal-input products. An SM copy kernel was much
slower (about 16.3 s) and was removed.

The full Pi pipeline keeps independent smaller multiplications on individual
GPUs and uses the distributed engine for larger products. Sequential
reciprocal and decimal-prefix stages can use a lower distributed threshold.
Decimal splits queue independent products, copy only required bit ranges, and
reuse each worker's power table and device buffers. CPU arithmetic and shifts
are parallelized; CPU affinity helped a controlled memory microbenchmark but
had no clear end-to-end benefit in the first comparison.

Both requested full results match separately generated y-cruncher references
byte for byte. Fractional-digit SHA-256:

- 10b: `1dabc86199e9f5e065fc2c839ebf77ba1546afaeb25aadbbccd82ffe522f5541`
- 25b: `164af5c396b3b87755a99ab07cd6141f8811401d4ccf33b54001f3124f9b54bb`

Verified output files are retained on the server at `results/pi-10b.txt` and
`results/pi-25b.txt`. They contain an initial `3`, exactly the requested
fractional digits without a decimal point, and one final newline.

Selected development runs (one trial each; not final repeated benchmarks):

| Version | Digits | Computation time | Full comparison |
| --- | ---: | ---: | --- |
| v5, distributed products throughout | 10b | 162.442 s | byte match |
| v7, parallel host arithmetic/views | 10b | 144.015 s | byte match |
| v9, hybrid independent/distributed products | 10b | 86.636 s | byte match |
| v11, registered memory and parallel decimal splits | 10b | 40.403 s | byte match |
| v13, reusable arenas | 10b | 36.291 s | byte match |
| v16, high-limb downloads and larger subtrees | 10b | 29.502 s | byte match |
| v9, first complete 25b run | 25b | 621.540 s | byte match |
| v12, cyclic tails and faster registration | 25b | 189.336 s | byte match |
| v16, high-limb downloads and peer-copy changes | 25b | 115.614 s | byte match |
| v18, allocation sizing and sequential-stage scheduling | 25b | 94.204 s | byte match |
| v23, dispatch by combined operand length | 25b | 74.260 s | byte match |

The interval includes host arena reservation, arithmetic, decimal conversion,
and decimal D2H. It excludes initial CUDA contexts, output writing, verification,
and final teardown. `benchmark_mgpu.py` additionally measures complete process
wall time and records source digests and environment settings. Ground-truth
reference generation overlapped early development work; its timing is not a
controlled CPU-versus-GPU comparison.

A six-configuration 10b sweep favored 2^23-block subtree caps, 32 host
arithmetic threads, and a 2^28-word distributed threshold in sequential stages
(26.995 s in that sweep). Balancing the final 25b subtree waves lowered peak
memory, allowing a smaller initial arena. Reusing registered memory for decimal
output removed an unnecessary zero-fill and pageable transfer. The final large
arithmetic improvement dispatches unequal local products by their combined
length instead of rejecting either operand above the grain. This reduced the
fully verified 25b run from about 88 s to 74.260 s; its peak live arena allocation
was 149.990 GB. The updated 10b run matched all digits in 24.785 s.

The original single-GPU 5b executable was also rechecked against every reference
digit in 19.621 s. The latest local arithmetic tests, forced distributed/cyclic
CUDA memcheck (zero errors), and full one-billion-digit hash comparison pass.
A final scheduling change overlaps initial host arena preparation with the
first GPU subtree wave, waiting before any subtree result is allocated or
downloaded. At 10b this removed about five seconds of exposed startup time.
The selected initial reservations are 48 GiB for 10b and 144 GiB for 25b,
with asynchronous reservation enabled by default. The candidate trials reached
19.608 s and 68.738 s respectively. CUDA memcheck and a complete 10m hash
comparison cover the asynchronous path; a full local 1b comparison also passed.

The last focused comparisons retained balanced 25b subtrees and left explicit
CPU affinity disabled: neither alternative showed a clear enough advantage.
Synchronous reservation trials at 10b took 25.010 s (none), 24.148 s (48 GiB),
24.443 s (56 GiB), and a 25.141 s median (64 GiB). At 25b, 144/160/168 GiB
synchronous reservations took 72.772/73.411/74.844 s; the 176 GiB median was
76.311 s. Smaller reservations sometimes need additional allocations during
merges; testing the full pipeline was necessary to assess their overlap.
Asynchronous 64/176 GiB trials took 20.141/70.919 s before selecting 48/144 GiB.
Raw exploratory records are retained under `validation/tuning/`.

The default final build again matched every 10b and 25b fractional digit
against the independent reference, in 20.179 s and 72.266 s with profiling
enabled. Its profiled registered-arena peaks were 68.045 GB and 184.671 GB;
peak live arena allocations were 59.960 GB and 147.452 GB. These peaks are
allocator measurements, not whole-system or GPU memory usage.

Final repeated timings (profiling disabled) are recorded below.

| Digits | Three serial computation trials (s) | Median (s) | Median complete process (s) |
| --- | --- | ---: | ---: |
| 10b | 19.665, 19.537, 19.270 | 19.537 | 23.431 |
| 25b | 69.007, 68.798, 70.814 | 69.007 | 75.906 |

The final source files and Makefile match between the local workspace and
server. Both benchmark records have empty override environments and match
`validation/source-sha256.json`. Full outputs remain saved on the server;
`results/manifest.json` records their exact lengths and fractional-digit hashes.
Final arithmetic checks, full comparisons, and the original 5b regression all
pass. No GPU experiments remain running.

## 50-billion-digit extension (September 11, 2026)

The retained precision itself now exceeds GMP's signed 32-bit limb count.
Added `wide_int.h`, a host integer with 64-bit signed lengths and `size_t`
capacities. It uses standard GMP `mpn` operations and the existing registered
allocator; bounded reference/startup operations use ordinary GMP `mpz`. No
GMP fork or ABI modification is needed. Host arithmetic is generic over the
old and new integer types; large product assembly now zeros, copies, and
propagates carries in parallel.

The GPU representation remains base 2^32 with the same 96-bit prime. Products
larger than the supported transform use Karatsuba children; eligible children
use cyclic convolution plus tail correction. Full cyclic outputs can now
exceed GMP's `mpz` size limit because the destination is a wide host integer.
The 50b run uses 32 exact GPU subtrees and 3,565,158,400 terms. Exact leaf and
subtree tests now include block offset 205,000,000.

The first full run matched all 50,000,000,000 fractional digits against an
independently generated y-cruncher reference. Computation took 218.626 s,
including host arena preparation and decimal transfer. A 384 GiB initial
reservation registered 412.317 GB, with 310.497 GB peak live arena allocation.
The verified result is saved at `results/pi-50b.txt` on the server, exactly
50,000,000,002 bytes, with fractional-digit SHA-256:

`76312744aa02ead5482ed3e65c3969f39657d6b221d5dcb22e5d2bb65d786e49`

Tests include actual integers beyond `INT_MAX` 64-bit limbs, signed arithmetic
against GMP, parallel carries and shifts, modular checks of products beyond
`UINT_MAX` limbs, GPU arithmetic tests, CUDA memcheck with zero errors, and a
complete local one-billion-digit comparison. The large host-size test passed
again after parallel modular checks were added.

A five-product Toom split was implemented and passed exact GMP tests, including
parallel exact division by three. At 5.2 billion 32-bit words per operand,
Karatsuba took 8.611/8.414 s and Toom took 13.637/13.355 s. Toom was removed.
The retained microbenchmark accepts `--bench-mul 12..33`, with optional
`GPUPI_BENCH_WORDS` and `GPUPI_BENCH_RESERVE` overrides. Its product timing
excludes initial reservation and input initialization.

Initial reservation trials with exact-sized host buffers: 256 GiB took
208.667 s (466.255 GB total registered, 345.966 GB peak live); 320 GiB took
217.354 s. Enabling physical-core affinity across both CPU sockets at 320 GiB
reduced this to 201.385 s (408.341 GB registered, 324.306 GB peak live).

Reviewing host allocation revealed that a one-limb carry could force a whole
large buffer to be copied. Wide host allocations now retain eight spare limbs,
which avoid these small growth reallocations. The local one-billion-digit
comparison improved from 5.525 s to 5.403 s and still matches the reference.

With spare limbs retained, another complete 50b comparison passed in
214.757 s using 256 GiB initially, but fragmentation expanded registered
storage to 542.146 GB. Affinity reduced that configuration to 207.055 s.
With affinity and 384 GiB initially, the run took 204.282 s and stayed in one
412.317 GB arena. The combined 320 GiB/affinity configuration was faster at
191.629 s: 362.784 GB registered, 309.562 GB peak live, and two raw allocations.
The final default reserves 320 GiB and enables physical-core affinity for
eight-GPU calculations from 30b upward. Smaller calculations retain their
previous affinity and reservation settings. Environment overrides remain
available. These exploratory timings precede the final repeated measurements.

The final automatic settings passed another complete 50b reference comparison
in 205.403 s with profiling enabled. Registered-arena peak was 449.128 GB,
with 345.270 GB peak live allocation; concurrent allocation order explains
some memory and timing variation between trials. Final full 10b and 25b
regressions also matched every reference digit in 20.370 s and 69.778 s.
The actual large-limb test and host/GPU arithmetic tests passed again on the
server. All 69 arithmetic source/Makefile hashes match the local workspace.

Final repeated 50b timings, with automatic defaults, profiling disabled, no
warmup, and serial execution:

| Trial | Computation (s) | Complete process (s) |
| --- | ---: | ---: |
| 1 | 201.142 | 213.149 |
| 2 | 199.305 | 211.634 |
| 3 | 199.707 | 211.805 |
| Median | **199.707** | **211.805** |

`benchmark_mgpu_50b.json` retains the complete logs, hardware information,
empty override environment, and final source hashes. The computation interval
includes initial registered-memory reservation, binary splitting, square root,
division, decimal conversion, and decimal D2H. The complete-process interval
also includes CUDA startup and teardown. Neither benchmark interval includes
full-file writing or reference comparison. The saved 50b file has already
passed full byte comparison; all 10b/25b regressions passed as well.

## 100-billion-digit extension (September 11, 2026)

The CLI now accepts up to 100b. GPU leaves select a new 64-bit term/scalar
specialization when a subtree crosses 2^32 terms. Earlier ranges keep the
32-bit specialization. The global block offset is also 64-bit. Two-word
scalar multiplication retains both halves of the denominator factors.
Through 2^33 terms, uncancelled P/Q/R require at most 106/138/188 bits, fitting
the existing 128/160/192-bit arrays. A range check rejects larger subtree
indices before allocation or launch. NTT orders and GPU subtree caps remain
bounded as before; the 100b tree has 7,130,316,800 terms.

Exact tests cover both head and interior leaves across the 2^32 boundary,
complete subtrees around that boundary and at block offsets 419,430,368 and
500,000,000, and rejection of overflowing/out-of-range offsets. Local and
server arithmetic checks pass; local CUDA memcheck reports zero errors.
The complete local 1b comparison and server original-program 5b regression
also pass. The 5b regression overlapped reference generation and is a
correctness check, not an isolated performance measurement.

The initial 100b candidate reserves 640 GiB and increases the arena reuse
cache to half of host RAM, capped at 1 TiB. A separate process-memory/GPU-memory
sampler records actual resource use. Reference generation precedes the full
GPU run to avoid contention during the measurement.

The first full 100b run matched every digit against y-cruncher and saved
`results/pi-100b.txt`, exactly 100,000,000,002 bytes. Fractional-digit SHA-256:

`78d8a7b1d47170bd95749cc4f4a5f1d047cda93d021fb7790c290433b869fe3a`

Computation took 735.094 s: binary splitting 512.286 s, square root 12.393 s,
division 161.259 s, and decimal conversion 45.667 s. Registered arenas peaked
at 1,433.293 GB, with 883.256 GB peak live allocation. The observed process
RSS high-water mark was 1,433.910 GB; minimum system-available RAM was
289.981 GB. Sampled GPU peaks were 85,847–88,393 MiB per card, including the
driver baseline. Thus 100b works on this machine, but its initial memory peak
exceeded the simple extrapolation from 50b.

The allocator reported one failed raw allocation. Isolated eight-context
tests reproduced `cudaHostRegister` out-of-memory errors for individual
640 GiB and 512 GiB regions; 384 GiB succeeded. Registration failure made
the first run lose its intended initial arena and subsequently accumulate
fragmented smaller arenas. The allocator now registers large contiguous
arenas in ranges of at most 384 GiB, rolls back completed ranges on failure,
and unregisters each range during release. Requests at or below 384 GiB keep
their existing behavior. The large allocation probe checks registered pointer
attributes and host/device transfers at both ends of every range on all GPUs.

An optional `GPUPI_MGPU_SERIAL_MERGES` threshold selects deferred parent
merges in the last tree levels; zero retains the previous parallel schedule.
The GPU multiplication engine is unchanged. This is being compared because
large distributed transforms already use all GPUs, while concurrent parents
retain more temporary host products. The local full 1b comparison also passes
with threshold 8.

The first registration workaround succeeded for 640 GiB and 768 GiB, including
endpoint transfers on all GPUs, but the full pipeline exposed an additional
CUDA restriction: one transfer spanning two registered ranges fails with
`invalid argument`. A focused 386 GiB test reproduced this: transfers inside
either range passed, while an 8 KiB transfer crossing the 384 GiB boundary
failed. The v2 pipeline stopped after its first merge level; the verified v1
output remains valid and unchanged.

The retained v3 approach uses separately allocated and registered arenas of
at most 384 GiB. Arena ownership already prevents coalescing between regions,
so no returned allocation can span CUDA registration boundaries. The optional
buddy allocator uses power-of-two arenas up to 256 GiB. Large-pool probes check
range endpoints on every GPU and verify that multiple pool allocations stay
inside their owning registered regions. Local arithmetic, reserved-memory
digit comparison, and a full 1b comparison with deferred parent merges pass.
The benchmark runner now also writes live stderr logs for long-run progress.

Separate-arena probes passed at 640 GiB (83.157 s) and 768 GiB (99.657 s),
including endpoint transfers on every GPU and pool ownership checks. The v3
parallel-parent 100b run then completed in 631.244 s (653.272 s complete
benchmark process), with no allocation failures. Binary splitting took
437.543 s, division 147.683 s, and decimal conversion 33.579 s. Registered
arenas peaked at 1,231.599 GB, with 905.756 GB peak live allocation.

The threshold-32 parent scheduling comparison slowed the 28-to-14-node level
from 17.852 s to 29.282 s. The next candidate therefore uses threshold 16,
preserving concurrency in that earlier level. It also adds an optional
`GPUPI_MGPU_SERIAL_PRODUCTS` path that collects each parent's Q/R/P products
before planning the next one. This bounds retained Karatsuba temporaries
inside even the final single parent. Full local 1b comparison passes with
this path, as does reserved-memory 10m comparison under CUDA memcheck.

The v3 threshold-32 comparison finished in 591.787 s (618.224 s complete
benchmark process), versus 631.244 s for parallel parents. Its registered
peak was 1,002.397 GB and peak live allocation 819.123 GB, with no failed
allocations. Binary splitting took 426.760 s, division 120.048 s, and decimal
conversion 33.174 s. Both v3 digit previews match the full reference. Observed
process high-water marks were 1,232.233 GB for parallel parents and 1,003.045 GB
for threshold 32. The v4 candidate uses parent threshold 16 plus sequential
product collection.

The selected default serializes parent merges at 32 remaining nodes from
75b on eight GPUs, while products within each parent retain parallel
planning. Environment overrides remain available. This changes no default
scheduling below 75b. The final source snapshot contains 70 source/build
files; full 100b reference comparison and a single profiling-disabled
benchmark are the final release checks.

The v4 product-collection candidate completed in 615.676 s (635.596 s
complete process), with registered peak 741.434 GB and live peak 665.253 GB.
Its preview matches y-cruncher. It is 4.0% slower than the selected v3
threshold-32 schedule but reduces registered memory by 26.0%; both overrides
are retained for users preferring that tradeoff. No allocation failures
occurred in either candidate.

Final automatic-default full verification passed: all 100,000,000,000
fractional digits match y-cruncher, SHA-256
`78d8a7b1d47170bd95749cc4f4a5f1d047cda93d021fb7790c290433b869fe3a`.
Computation took 615.617 s with profiling enabled. Registered peak was
1,181.103 GB, live peak 826.824 GB, and process host high-water mark
1,181,742,465,024 bytes. Allocation order changes reuse and fragmentation;
the earlier 1,002.397 GB peak is not a fixed bound for this schedule.
Minimum sampled system available RAM was 552,510,533,632 bytes. Sampled
per-card peaks ranged from 86,363 to 87,945 MiB (84.339–85.884 GiB), including
driver memory. No host allocation fallback occurred. The saved output from
v1 remains identical and was not rewritten during this verification.

The final profiling-disabled benchmark, automatic defaults and no warmup,
completed in 626.418221 s (648.647541 s complete process). This is one
measured run. Its 70 source hashes match the final snapshot, environment
overrides are empty, and both previews match y-cruncher. The final release
retains the lowest-time scheduling candidate while documenting run-to-run
allocation variability and the optional lower-memory schedule.
