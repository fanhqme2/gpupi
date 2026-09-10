#!/usr/bin/env python3
"""Derive guaranteed extra odd common factors for Chudnovsky blocks.

Outside primes 2,3,5,23,29, leaf P has factors (6i+1)(2i+1)(6i+5),
Q has (i+1)^3, and R contains Q's prime contribution. Every block-R term
contains a P prefix and Q suffix, times a power of two. The minimum valuation
over all split points therefore divides P, Q and R. Minimize over every start
residue modulo p; ignoring higher p powers is conservative. Subtract factors
already cancelled from the two children, starting with the original 17-block
cancellation. All candidate primes are at most the block length: a larger
prime cannot divide every Q block. --check verifies the constants in the CUDA
source by exhaustive residue enumeration, independently of GPU computation.
"""
import argparse
import math
from pathlib import Path
import re


def derive(max_terms):
    primes = [p for p in range(7, max_terms + 1)
              if p not in (23, 29) and
              all(p % d for d in range(2, math.isqrt(p) + 1))]
    previous = {7: 6, 11: 3, 13: 2, 17: 3}
    length = 34
    while length <= max_terms:
        extra, current = {}, {}
        for p in primes:
            bound = 3 * length
            for start in range(p):
                value = sum(3 * ((start + i + 1) % p == 0) for i in range(length))
                low = value
                for i in range(length):
                    x = start + i
                    value += sum((m * x + a) % p == 0 for m, a in ((6, 1), (2, 1), (6, 5)))
                    value -= 3 * ((x + 1) % p == 0)
                    low = min(low, value)
                bound = min(bound, low)
            exponent = max(0, bound - 2 * previous.get(p, 0))
            current[p] = 2 * previous.get(p, 0) + exponent
            if exponent:
                extra[p] = exponent
        packed, value = [], 1
        for p, exponent in extra.items():
            for _ in range(exponent):
                if value * p > 2**32 - 1:
                    packed.append(value)
                    value = 1
                value *= p
        if value > 1:
            packed.append(value)
        yield length, extra, packed
        previous = current
        length *= 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-terms', type=int, choices=(34, 68, 136), default=136)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).with_name('pi_bs_gpu.cu').read_text() if args.check else ''
    for length, factors, packed in derive(args.max_terms):
        if args.check:
            match = re.search(r'kCommon' + str(length) + r'\[\]\s*=\s*\{([^}]+)\}', source)
            assert match, length
            actual = [int(v) for v in re.findall(r'(\d+)u', match.group(1))]
            assert actual == packed, (length, actual, packed)
            print(f'PROVED block={length}, {len(packed)} packed divisors')
        else:
            print(length, factors, packed)
