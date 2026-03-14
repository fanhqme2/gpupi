'''
Chudnovsky
1 / pi = 12 sum((-1) ** k * frac(6 * k) * (545140134 * k + 13591409) / frac(3 * k) / frac(k) ** 3 / (640320 ** (3 * k + 3 / 2))) for k)

P(i) / Q(i) = (6 * i + 1) * (6 * i + 2) * (6 * i + 3) * (6 * i + 4) * (6 * i + 5) * (6 * i + 6) / (3 * i + 1) / (3 * i + 2) / (3 * i + 3) / (i + 1) / (i + 1) / (i + 1) / (640320 ** 3)
R(i) = 545140134 * i + 13591409

P(i, j) = P(i) * P(i + 1) * ... * P(j - 1)
Q(i, j) = Q(i) * Q(i + 1) * ... * Q(j - 1)
R(i, j) = P(i, j - 1) * R(j - 1) * Q(j - 1, j) - ... +- P(i, i) * R(i) * Q(i, j)

'''

import time
import gmpy2
import math
import sys

mpz = gmpy2.mpz

n_terms = 714
if len(sys.argv) > 1:
    n_terms = int(sys.argv[1])

target_digits = n_terms * 14
target_digits = min(target_digits, 10 ** 9)

prec_bits = int(target_digits * 3.322) + 32
prec_bits += (-prec_bits) & 31

def trim_prec_quotient(*args, prec_bits):
    x_bits = min(x.bit_length() for x in args)
    if x_bits > prec_bits:
        crop = x_bits - prec_bits
        crop -= crop & 31
        if crop >= 32:
            return [x >> crop for x in args]
    return args

def reduce_gcd_quotient(a, b):
    ga = a
    gb = b
    while ga != 0:
        ga, gb = gb % ga, ga
    return a // gb, b // gb

def binary_split(i, j, is_initial = False):
    if j == i + 1:
        # i_mpz = mpz(i)
        # i1 = i_mpz + 1
        # p = (6 * i_mpz + 1) * (2 * i_mpz + 1) * (6 * i_mpz + 5)
        # scale = i1 * i1 * i1
        # q = scale * (640320 * 640320 * 26680)
        # scale_2 = scale * (640320 * 40020)
        # r = (545140134 * i_mpz + 13591409) * scale_2
        # g = gmpy2.gcd(p, scale_2)
        # p //= g
        # q //= g
        # r //= g
        # p0 = p
        # q0 = q
        # r0 = r

        p1 = 6 * i + 1
        p2 = 2 * i + 1
        p3 = 6 * i + 5
        q1 = i + 1
        q2 = i + 1
        q3 = i + 1
        q4 = 640320
        q5 = 40020
        q6 = 426880
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
        if i == 0:
            q2, r2 = trim_prec_quotient(q2, r2, prec_bits = (prec_bits - (15 * (j - k))))
    else:
        p2, q2, r2 = binary_split(k, j)
    
    q = q1 * q2
    r1 = r1 * q2
    r2 = p1 * r2
    if (k - i) & 1:
        r = r1 - r2
    else:
        r = r1 + r2
    if is_initial:
        return q, r
    p = p1 * p2

    # if j - i >= 64 and j - i < 128:
    #     g = gmpy2.gcd(p, q)
    #     g = gmpy2.gcd(g, r)
    #     p //= g
    #     q //= g
    #     r //= g

    return p, q, r

print('digits', target_digits)

t0 = time.time()
q, r = binary_split(0, n_terms, is_initial = True)

print('binary splitting time %.3fs' % (time.time() - t0))
print('len(hex(q))', q.num_digits(16) + 2)
print('len(hex(r))', r.num_digits(16) + 2)

q, r = trim_prec_quotient(q, r, prec_bits = prec_bits)

p_15 = mpz(4001)
q_15 = mpz(40)
valid_terms = 7.5

while valid_terms < target_digits:
    valid_terms *= 2
    # (p_15 / q_15 + 10005 * q_15 / p_15) / 2
    p_15, q_15 = (
        p_15 * p_15 + 10005 * q_15 * q_15,
        p_15 * q_15 * 2,
    )

print('sqrt time %.3fs' % (time.time() - t0))

print('len(hex(p_15))', p_15.num_digits(16) + 2)
print('len(hex(q_15))', q_15.num_digits(16) + 2)

p_final = p_15 * q
q_final = r * q_15

print('q and p mul time %.3fs' % (time.time() - t0))

p_final, q_final = trim_prec_quotient(p_final, q_final, prec_bits = prec_bits)

def recursive_inverse(x, L, is_initial = False):
    if L <= 256:
        y = mpz(1) << (L // 2 + 1)
        res = (mpz(1) << L) - x * y
        while abs(res) >= x:
            dy = (res * y) >> L
            y += dy
            res -= dy * x
        if res < 0:
            res += x
            y += 1
        return y, res

    m = (L - 2) // 4
    m -= m & 31
    x1 = x >> m
    x2 = x & ((mpz(1) << m) - 1)

    y1, res1 = recursive_inverse(x1, L - m * 2)
    res2 = (res1 << m) - x2 * y1
    y2 = (res2 * y1) >> (L - m * 2)
    if is_initial:
        return (y1 << m) + y2
    res2 = (res2 << m) - y2 * x
    assert res2 >= 0 and res2 < x
    # while res2 < 0:
    #     res2 += x
    #     y2 -= 1
    # while res2 >= x:
    #     res2 -= x
    #     y2 += 1
    return (y1 << m) + y2, res2

q_bits_count = q_final.bit_length()
inv_q_final = recursive_inverse(q_final, q_bits_count * 2 - 1, is_initial = True)
print('inverse time %.3fs' % (time.time() - t0))
prod_final = (p_final * inv_q_final) >> (q_bits_count * 2 - 1 - prec_bits)
print('div time %.3fs' % (time.time() - t0))

powers_of_5 = [mpz(5)]
max_power = 1
while max_power * 2 <= target_digits:
    powers_of_5.append(powers_of_5[-1] ** 2)
    max_power *= 2

print('power_5 time %.3fs' % (time.time() - t0))

def recursive_decimal(p, B, L):
    # p >> B  for L decimal digits
    # we are extracting digits of Pi, and it is guarenteed that there are at most 9 consecutive 9s in the first one billion digits
    if L <= 512:
        # ret = str((p * (10 ** L)) >> B)
        # ret = '0' * (L - len(ret)) + ret
        # return ret
        ret = ''
        while L > 0:
            digits = min(L, 9)
            scale = 10 ** digits
            prod = p * mpz(scale)
            chunk = prod >> B
            p = prod & ((mpz(1) << B) - 1)
            chunk = str(chunk)
            chunk = '0' * (digits - len(chunk)) + chunk
            ret = ret + chunk
            L -= 9
        return ret
    L_half = 1
    power_idx = 0
    while L_half * 2 <= L // 2:
        L_half *= 2
        power_idx += 1
    L_half = L - L_half

    #prod = p * (mpz(5) ** (L - L_half))
    prod = p * powers_of_5[power_idx]
    residual = prod & ((mpz(1) << (B - (L - L_half))) - 1)
    required_bits_lo = int(L_half * 3.322 + 34)
    required_bits_hi = int((L - L_half) * 3.322 + 34)
    return recursive_decimal(p >> (B - required_bits_hi), required_bits_hi, L - L_half) + recursive_decimal(residual >> (B - (L - L_half) - required_bits_lo), required_bits_lo, L_half)

ret_gt = open('../pi_1000000000.txt', 'r').read(target_digits + 2).replace('.', '')
ret = '3' + recursive_decimal(prod_final - (mpz(3) << prec_bits), prec_bits, target_digits)
print('decimal time %.3fs' % (time.time() - t0))
#fout = open('./%d.txt'%target_digits, 'w')
#fout.write(ret)
#fout.close()
assert ret == ret_gt
print(ret[:20] + '...' + ret[-20:])
