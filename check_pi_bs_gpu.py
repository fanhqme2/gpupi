#!/usr/bin/env python3

import re
import subprocess
import sys

import gmpy2
mpz = gmpy2.mpz

MAX_INDEX_EXCLUSIVE = 71428572

def reduce_gcd_quotient(a, b):
    for factor in (3, 5, 23, 29):
        if a % factor == 0 and b % factor == 0:
            a //= factor
            b //= factor
    return a, b

def binary_split(i, j, is_initial = False):
    if j == i + 1:
        p1 = 6 * i + 1
        p2 = 2 * i + 1
        p3 = 6 * i + 5
        q1 = i + 1
        q2 = i + 1
        q3 = i + 1
        q4 = 640320 >> 6
        q5 = 40020 >> 2
        q6 = 426880 >> 7

        r1 = 545140134 * i + 13591409

        if p1 % 5 == 0 and q1 % 5 == 0:
            p1 //= 5
            q1 //= 5
        if p1 % 5 == 0 and q2 % 5 == 0:
            p1 //= 5
            q2 //= 5
        if p1 % 5 == 0 and q3 % 5 == 0:
            p1 //= 5
            q3 //= 5
        p1, q4 = reduce_gcd_quotient(p1, q4)
        p1, q5 = reduce_gcd_quotient(p1, q5)
        p2, q4 = reduce_gcd_quotient(p2, q4)
        p2, q5 = reduce_gcd_quotient(p2, q5)
        p3, q4 = reduce_gcd_quotient(p3, q4)
        p3, q5 = reduce_gcd_quotient(p3, q5)


        p = mpz(p1) * p2 * p3
        q = mpz(q1) * q2 * q3 * q4 * q5 * q6
        r = mpz(r1) * q1 * q2 * q3 * q4 * q5

        if is_initial:
            return q, r
        else:
            return p, q, r
    k = (i + j) // 2
    p1, q1, r1 = binary_split(i, k)
    if is_initial:
        q2, r2 = binary_split(k, j, True)
    else:
        p2, q2, r2 = binary_split(k, j)
    
    q = q1 * q2
    r1 = r1 * q2
    r2 = p1 * r2

    r1 = r1 << (15 * (j - k))

    if (k - i) & 1:
        r = r1 - r2
    else:
        r = r1 + r2
    if is_initial:
        return q, r
    p = p1 * p2
    return p, q, r


def int_to_hex(value):
    if value == 0:
        return "0"
    return format(int(value), "x")


def limb_length_from_hex(hex_value):
    if hex_value == "0":
        return 0
    return (len(hex_value) + 7) // 8


def limb_length(value):
    return 0 if value == 0 else (int(value).bit_length() + 31) // 32


def int_to_limbs(value):
    limbs = []
    value = int(value)
    while value:
        limbs.append(value & 0xffffffff)
        value >>= 32
    return limbs


def parse_cuda_output(text):
    matches = dict(re.findall(r"([PQR])\s*=\s*([0-9a-fA-F]+)", text))
    missing = [name for name in ("P", "Q", "R") if name not in matches]
    if missing:
        raise ValueError("missing values in CUDA output: %s" % ", ".join(missing))
    return {name: mpz(matches[name], 16) for name in ("P", "Q", "R")}


def print_limb_mismatch(name, expected, actual, window=2):
    expected_limbs = int_to_limbs(expected)
    actual_limbs = int_to_limbs(actual)
    if len(expected_limbs) != len(actual_limbs):
        print(
            "%s length mismatch: expected %d limbs, actual %d limbs"
            % (name, len(expected_limbs), len(actual_limbs))
        )

    limit = min(len(expected_limbs), len(actual_limbs))
    mismatch = None
    for i in range(limit):
        if expected_limbs[i] != actual_limbs[i]:
            mismatch = i
            break
    if mismatch is None:
        if len(expected_limbs) != len(actual_limbs):
            mismatch = limit
        else:
            print("%s matches" % name)
            return

    print("%s mismatch at limb %d" % (name, mismatch))
    start = max(0, mismatch - window)
    end = min(max(len(expected_limbs), len(actual_limbs)), mismatch + window + 1)
    for i in range(start, end):
        exp = expected_limbs[i] if i < len(expected_limbs) else None
        act = actual_limbs[i] if i < len(actual_limbs) else None
        exp_text = "--------" if exp is None else "%08x" % exp
        act_text = "--------" if act is None else "%08x" % act
        marker = "<--" if i == mismatch else "   "
        print("%s limb %d: expected %s actual %s" % (marker, i, exp_text, act_text))


def compare_value(name, expected, actual):
    if expected == actual:
        print("%s matches" % name)
        return True
    print_limb_mismatch(name, expected, actual)
    return False

def main():
    N = int(sys.argv[1])
    binary = sys.argv[2] if len(sys.argv) >= 3 else "./pi_bs_gpu"

    p, q, r = binary_split(0, N * 17)
    expected = {"P": p, "Q": q, "R": r}

    result = subprocess.run(
        [binary, str(N)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = parse_cuda_output(result.stdout)

    ok = True
    for name in ("P", "Q", "R"):
        ok = compare_value(name, expected[name], actual[name]) and ok

    if ok:
        print("All values match for N=%d" % N)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
