#!/usr/bin/env python3
"""Sample a Pi process's host RSS/high-water mark and per-GPU memory usage."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def fields(path):
    return {line.split(':')[0]: line.split(':', 1)[1].strip()
            for line in path.read_text().splitlines() if ':' in line}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--digits', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    pid = None
    while pid is None:
        for path in Path('/proc').glob('[0-9]*'):
            try:
                command = (path / 'cmdline').read_bytes().split(b'\0')
                if (len(command) >= 2 and Path(command[0].decode()).name == 'pi_mgpu'
                        and command[1].decode() == args.digits):
                    pid = int(path.name)
                    break
            except (OSError, UnicodeError):
                pass
        if time.monotonic() - start > 14400:
            raise RuntimeError('Timed out waiting for Pi process')
        if pid is None:
            time.sleep(2)
    record = {'pid': pid, 'digits': int(args.digits), 'interval_s': 5,
              'scope': 'Host VmHWM is the process kernel high-water mark observed while running. GPU memory is sampled across all compute processes on each card, and can miss brief peaks.',
              'samples': []}
    start = time.monotonic()
    while True:
        try:
            status = fields(Path(f'/proc/{pid}/status'))
            if 'VmRSS' not in status:
                break
            ram = fields(Path('/proc/meminfo'))
        except FileNotFoundError:
            break
        query = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True)
        record['samples'].append({
            'elapsed_s': time.monotonic() - start,
            'rss_kib': int(status['VmRSS'].split()[0]),
            'hwm_kib': int(status['VmHWM'].split()[0]),
            'available_kib': int(ram['MemAvailable'].split()[0]),
            'gpu_used_mib': [int(v) for v in query.stdout.split()],
        })
        time.sleep(5)
    if not record['samples']:
        raise RuntimeError('Process exited before any memory sample was recorded')
    record['max_host_hwm_bytes'] = max(v['hwm_kib'] for v in record['samples']) * 1024
    record['min_system_available_bytes'] = min(v['available_kib'] for v in record['samples']) * 1024
    record['sampled_gpu_peak_mib'] = [max(v['gpu_used_mib'][i] for v in record['samples'])
                                      for i in range(len(record['samples'][0]['gpu_used_mib']))]
    args.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
