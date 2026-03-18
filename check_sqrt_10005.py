#!/usr/bin/env python3

import re
import subprocess
import sys

import gmpy2

mpz = gmpy2.mpz


def sqrt_10005_reference(target_digits):
    p = mpz(4001)
    q = mpz(40)
    valid_digits = 7.5
    while valid_digits < target_digits:
        p, q = p * p + 10005 * q * q, 2 * p * q
        valid_digits *= 2
    return p, q


def parse_cuda_output(text):
    matches = dict(re.findall(r"([PQ])\s*=\s*([0-9a-fA-F]+)", text))
    missing = [name for name in ("P", "Q") if name not in matches]
    if missing:
        raise ValueError("missing values in CUDA output: %s" % ", ".join(missing))
    return {name: mpz(matches[name], 16) for name in ("P", "Q")}


def int_to_hex(value):
    if value == 0:
        return "0"
    return format(int(value), "x")


def int_to_limbs(value):
    hex_value = int_to_hex(value)
    if hex_value == "0":
        return []
    padded = hex_value.zfill(((len(hex_value) + 7) // 8) * 8)
    return [int(padded[i - 8:i], 16) for i in range(len(padded), 0, -8)]


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


def describe_accuracy(target_digits, p, q):
    delta = p * p - mpz(10005) * q * q
    abs_delta = abs(delta)
    if abs_delta == 0:
        print("exact equality: P^2 = 10005 * Q^2")
        return

    numer_bits = abs_delta.bit_length()
    denom_bits = (2 * mpz(10005) * q * q).bit_length()
    approx_digits = int((denom_bits - numer_bits) * 0.3010299956639812)
    print("derived decimal digits ~= %d for target_digits=%d" % (approx_digits, target_digits))


def main():
    target_digits = int(sys.argv[1])
    binary = sys.argv[2] if len(sys.argv) >= 3 else "./pi_bs_gpu"

    result = subprocess.run(
        [binary, str(target_digits)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = parse_cuda_output(result.stdout)

    p, q = sqrt_10005_reference(target_digits)
    expected = {"P": p, "Q": q}

    ok = compare_value("P", expected["P"], actual["P"]) and compare_value("Q", expected["Q"], actual["Q"])
    describe_accuracy(target_digits, actual["P"], actual["Q"])

    if ok:
        print("All values match for target_digits=%d" % target_digits)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
