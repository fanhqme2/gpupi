#!/usr/bin/env python3

import subprocess
import sys


def main():
    digits = 1_000_000_000
    if len(sys.argv) >= 2:
        digits = int(sys.argv[1])

    with open("../pi_1000000000.txt", "r", encoding="utf-8") as fin:
        expected = fin.read(digits + 2).replace(".", "")[:digits + 1]

    proc = subprocess.run(
        ["./pi_bs_gpu", str(digits)],
        check=True,
        capture_output=True,
        text=True,
    )

    got = proc.stdout.rstrip("\n")
    if got != expected:
        sys.stderr.write(proc.stderr)
        mismatch = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), min(len(got), len(expected)))
        raise AssertionError(
            f"mismatch at index {mismatch}: got={got[mismatch:mismatch+32]!r} expected={expected[mismatch:mismatch+32]!r}"
        )

    if len(got) != len(expected):
        raise AssertionError(f"length mismatch: got={len(got)} expected={len(expected)}")

    sys.stderr.write(proc.stderr)
    print(f"matched digits={digits}")


if __name__ == "__main__":
    main()
