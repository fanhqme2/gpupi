#!/usr/bin/env python3
"""Run serial warmups and trials; report every GPU computation time and median."""
import argparse
import json
import re
import statistics
import subprocess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--exe', default='./pi_bs_gpu')
ap.add_argument('--digits', type=int, default=2_500_000_000)
ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--warmups', type=int, default=1)
args = ap.parse_args()
if args.runs < 1 or args.warmups < 0:
    ap.error('runs must be positive and warmups nonnegative')
times = []
for i in range(args.runs + args.warmups):
    p = subprocess.run([args.exe, str(args.digits), '--benchmark'], check=True, text=True, capture_output=True)
    match = re.search(r'elapsed_ms=([0-9.]+)', p.stdout)
    if not match:
        raise RuntimeError(p.stdout + p.stderr)
    ms = float(match[1])
    print(json.dumps({'trial': i - args.warmups + 1, 'warmup': i < args.warmups, 'elapsed_ms': ms}), flush=True)
    if i >= args.warmups:
        times.append(ms)
print(json.dumps({'digits': args.digits, 'times_ms': times, 'median_ms': statistics.median(times),
                  'best_ms': min(times), 'worst_ms': max(times)}), flush=True)
