#!/usr/bin/env python3
"""Run serial multi-GPU Pi trials and retain timing scope, settings, and raw logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--digits', type=int, required=True)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--warmups', type=int, default=0)
    ap.add_argument('--exe', default='./pi_mgpu')
    ap.add_argument('--output', type=Path, default=Path('benchmark_mgpu_results.json'))
    args = ap.parse_args()
    if args.runs < 1 or args.warmups < 0:
        ap.error('runs must be positive and warmups nonnegative')
    command = [args.exe, str(args.digits), '--benchmark']
    record = {
        'command': command,
        'environment': {k: v for k, v in os.environ.items() if k.startswith('GPUPI_') or k == 'CUDA_VISIBLE_DEVICES'},
        'timing_scope': 'Program elapsed_ms includes host arena reservation, binary splitting, sqrt, division, decimal conversion, and decimal D2H. Excludes CUDA context setup, output printing, verification, and teardown. process_wall_s includes the whole subprocess.',
        'source_sha256': {},
        'trials': [],
    }
    names = ['Makefile'] + sorted(str(p) for p in Path('.').iterdir()
                                 if p.suffix in {'.cu', '.cuh', '.h', '.cpp'})
    for name in names:
        path = Path(name)
        if path.exists():
            record['source_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,pci.bus_id', '--format=csv'], capture_output=True, text=True)
    record['gpus'] = gpu.stdout.strip()
    for trial in range(args.warmups + args.runs):
        stderr_path = args.output.with_name(f'{args.output.stem}.trial-{trial}.stderr.log')
        # Keep stage progress readable during long trials, as well as in JSON.
        with stderr_path.open('w') as log:
            begin = time.monotonic()
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True)
            wall = time.monotonic() - begin
        result.stderr = stderr_path.read_text()
        if result.returncode:
            raise RuntimeError(result.stderr)
        match = re.search(r'elapsed_ms=([0-9.]+)', result.stderr)
        if match is None:
            raise RuntimeError('Program did not report elapsed_ms: ' + result.stderr)
        data = {'warmup': trial < args.warmups, 'elapsed_ms': float(match[1]),
                'process_wall_s': wall, 'stdout': result.stdout, 'stderr': result.stderr,
                'stderr_log': str(stderr_path)}
        record['trials'].append(data)
        measured = [v['elapsed_ms'] for v in record['trials'] if not v['warmup']]
        if measured:
            record['median_ms'] = statistics.median(measured)
            record['best_ms'] = min(measured)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(f"trial={trial} warmup={data['warmup']} elapsed_ms={data['elapsed_ms']:.3f} process_wall_s={wall:.3f}", flush=True)


if __name__ == '__main__':
    main()
