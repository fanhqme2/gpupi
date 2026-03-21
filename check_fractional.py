#!/usr/bin/env python3

import re
import subprocess
import sys

import gmpy2

mpz = gmpy2.mpz

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
    
    if j - i == 17:
        q = q // 130016887243243
        r = r // 130016887243243

    if is_initial:
        return q, r
    p = p1 * p2

    if j - i == 17:
        p = p // 130016887243243

    return p, q, r

def sqrt_10005(target_digits):
    p = mpz(4001)
    q = mpz(40)
    valid_digits = 7.5
    while valid_digits < target_digits:
        p, q = p * p + 10005 * q * q, 2 * p * q
        valid_digits *= 2
    return p, q

def truncate_limbs_quotient(p, q, prec_limbs):
    p_nlimbs = (p.bit_length() + 31) // 32
    q_nlimbs = (q.bit_length() + 31) // 32
    extra_limbs = min(p_nlimbs, q_nlimbs) - prec_limbs
    if extra_limbs > 0:
        p >>= extra_limbs * 32
        q >>= extra_limbs * 32
    return p, q

def recursive_inverse_old(x, L, is_initial = False):
    if L <= 256:
        total = mpz(1) << L
        y = total // x
        res = total - y * x
        return y, res

    m = (L - 2) // 4
    m -= m & 31
    x1 = x >> m
    x2 = x & ((mpz(1) << m) - 1)

    y1, res1 = recursive_inverse_old(x1, L - m * 2)
    x2y1 = x2 * y1
    res1_m = res1 << m
    if res1_m >= x2y1:
        res2 = res1_m - x2y1
        y2 = (res2 * y1) >> (L - m * 2)
        res2 = (res2 << m) - y2 * x
        y1y2 = (y1 << m) + y2
    else:
        res2 = x2y1 - res1_m
        y2 = ((res2 * y1) >> (L - m * 2)) + 1
        res2 = y2 * x - (res2 << m)
        y1y2 = (y1 << m) - y2

    if is_initial:
        return y1y2
    
    assert res2 >= 0 and res2 < x
    # while res2 < 0:
    #     res2 += x
    #     y2 -= 1
    # while res2 >= x:
    #     res2 -= x
    #     y2 += 1
    return y1y2, res2

def recursive_inverse(x, L):
    Ls = [L]
    while Ls[-1] > 256:
        m = (Ls[-1] - 2) // 4
        m -= m & 31
        Ls.append(Ls[-1] - m * 2)
    def get_chunk(x, l): # Can be implemented as slicing as ((L - l) // 2) is always multiple of 32
        return x >> ((L - l) // 2)
    L0 = Ls[-1]
    total = mpz(1) << L0
    y, res = gmpy2.f_divmod(total, get_chunk(x, L0))

    for L1, L2 in zip(reversed(Ls), reversed(Ls[:-1])):
        # go from L1 to L2
        m = (L2 - L1) // 2
        x2 = get_chunk(x, L2) & ((mpz(1) << m) - 1)  # Can be implemented as slicing as m is multiple of 32
        x2y1 = x2 * y
        res1_m = res << m
        if res1_m >= x2y1:
            res2 = res1_m - x2y1
            y2 = (res2 * y) >> (L2 - m * 2)
            if L2 != L:
                res = (res2 << m) - y2 * get_chunk(x, L2)
            y = (y << m) + y2
        else:
            res2 = x2y1 - res1_m
            y2 = ((res2 * y) >> (L2 - m * 2)) + 1
            if L2 != L:
                res = y2 * get_chunk(x, L2) - (res2 << m)
            y = (y << m) - y2
    
    return y

def round_up_decimal_digits(target_digits):
    if target_digits <= 0:
        return 0
    digits = 477
    while digits < target_digits:
        digits *= 2
    return digits


def main():
    target_digits = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    
    proc = subprocess.run(
        ["./pi_bs_gpu", str(target_digits), "--full-print"],
        check=True,
        capture_output=True,
        text=True,
    )
    info_match = re.search(r"^target_digits=(\d+) compute_digits=(\d+) RET_len=(\d+) elapsed_ms=", proc.stderr, re.MULTILINE)
    ret_match = re.search(r"^RET = ([0-9a-fA-F]+)$", proc.stdout, re.MULTILINE)
    dec_match = re.search(r"^DEC = ([0-9]+)$", proc.stdout, re.MULTILINE)
    if info_match is None or ret_match is None or dec_match is None:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError("failed to parse RET/DEC from pi_bs_gpu output")
    gpu_compute_digits = int(info_match.group(2))
    gpu_ret = mpz(ret_match.group(1), 16)
    gpu_dec = dec_match.group(1)

    compute_digits = round_up_decimal_digits(target_digits)
    assert gpu_compute_digits == compute_digits

    prec_limbs = (int(compute_digits * 3.322) + 31) // 32 + 1
    n = 1
    while n * 17 * 14.3 < compute_digits:
        n *= 2

    q, r = binary_split(0, n * 17, True)
    r >>= 15 * (n * 17) - 8

    p_15, q_15 = sqrt_10005(compute_digits)
    q, r = truncate_limbs_quotient(q, r, prec_limbs)

    p_final = p_15 * q
    q_final = r * q_15

    p_final, q_final = truncate_limbs_quotient(p_final, q_final, prec_limbs)
    
    q_bits_count = q_final.bit_length()
    inv_q_final = recursive_inverse(q_final, q_bits_count * 2 - 1)

    ret_final = (p_final * inv_q_final) >> (q_bits_count * 2 - 1 - prec_limbs * 32)

    assert gpu_ret == ret_final
    expected_dec = str((ret_final * mpz(10) ** compute_digits) >> (prec_limbs * 32))
    assert gpu_dec == expected_dec[:target_digits + 1]
    print(f"matched digits={target_digits}")

    # #verifying the results with the known value of pi
    # ret = (ret_final * mpz(10) ** target_digits) >> (prec_limbs * 32)
    # ret = str(ret)
    # ret_gt = open('../pi_1000000000.txt', 'r').read(target_digits + 2).replace('.', '')
    # print(ret[:20] + '...' + ret[-20:])
    # assert ret == ret_gt
    

if __name__ == "__main__":
    main()
