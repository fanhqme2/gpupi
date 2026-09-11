#!/usr/bin/env python3
"""Compare every decimal digit against a y-cruncher text reference, streaming.

Example: python3 check_pi.py .scratch/reference/'Pi - Dec - Chudnovsky.txt' 2500000000 -- ./pi_bs_gpu 2500000000
The command may instead be ssh ... ./pi_bs_gpu 2500000000.
"""
import argparse
import contextlib
import hashlib
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sha256', action='store_true',
                    help='interpret reference as a previously verified fractional-digit SHA-256')
    ap.add_argument('--save', type=Path,
                    help='retain the verified program output in this file')
    ap.add_argument('reference')
    ap.add_argument('digits', type=int)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.digits < 0:
        ap.error('digits must be nonnegative')
    if args.sha256 and (len(args.reference) != 64 or
                       any(c not in '0123456789abcdefABCDEF' for c in args.reference)):
        ap.error('expected a 64-character hexadecimal SHA-256')
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        command = ['./pi_bs_gpu', str(args.digits)]
    digest = hashlib.sha256()
    saved = args.save.with_name(args.save.name + '.partial') if args.save else None
    output = saved.open('wb') if saved else None
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        reference_context = contextlib.nullcontext() if args.sha256 else open(args.reference, 'rb')
        with reference_context as reference:
            # y-cruncher writes a plain 3.xxx text file; program writes 3xxx.
            if reference is not None:
                assert reference.read(2) == b'3.'
            assert proc.stdout.read(1) == b'3'
            if output is not None:
                output.write(b'3')
            offset = 0
            while offset < args.digits:
                size = min(4 * 1024 * 1024, args.digits - offset)
                expected = reference.read(size) if reference is not None else None
                actual = proc.stdout.read(size)
                if len(actual) != size:
                    raise AssertionError('Program output is shorter than requested precision')
                if expected is not None and len(expected) != size:
                    raise AssertionError('Reference is shorter than requested precision')
                if expected is not None and actual != expected:
                    index = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
                    raise AssertionError(f'Mismatch at fractional digit {offset + index + 1}: got {actual[index:index+32]!r}, expected {expected[index:index+32]!r}')
                digest.update(actual)
                if output is not None:
                    output.write(actual)
                offset += size
            assert proc.stdout.read() == b'\n', 'Unexpected output suffix'
        if proc.wait() != 0:
            raise RuntimeError(f'Program exited {proc.returncode}')
        if args.sha256 and digest.hexdigest() != args.reference.lower():
            raise AssertionError(f'SHA-256 mismatch: {digest.hexdigest()}')
        if output is not None:
            output.write(b'\n')
            output.close()
            saved.replace(args.save)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        if output is not None:
            output.close()
        if saved is not None:
            saved.unlink(missing_ok=True)
    mode = 'SHA-256 MATCH' if args.sha256 else 'BYTE MATCH'
    print(f'{mode}: all {args.digits:,} fractional digits; SHA-256={digest.hexdigest()}', flush=True)


if __name__ == '__main__':
    main()
