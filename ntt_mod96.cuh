#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "batch_mul_addsub_asm.h"

// Arithmetic modulo p = 2^96 - 2^32 + 1. Inputs and outputs are canonical
// residues in [0,p); the correction bounds below rely on that invariant.
__device__ __forceinline__ uint3 add_mod(uint3 a, uint3 b){
    uint32_t c0, c1, c2;
    c0 = add_cc(a.x, b.x);
    c1 = addc_cc(a.y, b.y);
    c2 = addc_cc(a.z, b.z);
    uint32_t carry = addc(0u, 0u);

#ifdef GPUPI_OLD_ADDSUB96
    uint32_t t = add_cc(c0, 0u - (carry + 1u));
    t = addc_cc(c1, carry);
    t = addc_cc(c2, 0u);
    uint32_t carry2 = addc(0u, 0u);
    (void)t;
    carry += carry2;

    uint32_t cs0 = sub_cc(0u, carry);
    uint32_t cs1 = subc(carry, 0u);

    uint32_t out0 = add_cc(c0, cs0);
    uint32_t out1 = addc_cc(c1, cs1);
    uint32_t out2 = addc(c2, 0u);
#else
    // Canonical operands imply that correcting a wrapped sum cannot wrap
    // again. A nonwrapped sum is >= p exactly in this small top interval.
    const uint32_t correction = carry | ((c1 & c2) == 0xffffffffu && c0 != 0u);
    uint32_t out0 = add_cc(c0, 0u-correction);
    uint32_t out1 = addc_cc(c1, 0u);
    uint32_t out2 = addc(c2, 0u);
#endif
    return make_uint3(out0, out1, out2);
}

__device__ __forceinline__ uint3 sub_mod(uint3 a, uint3 b){
    uint32_t c0 = sub_cc(a.x, b.x);
    uint32_t c1 = subc_cc(a.y, b.y);
    uint32_t c2 = subc_cc(a.z, b.z);
    uint32_t borrow = subc(0u, 0u) & 1u;

#ifdef GPUPI_OLD_ADDSUB96
    uint32_t bs0 = sub_cc(0u, borrow);
    uint32_t bs1 = subc(borrow, 0u);

    uint32_t out0 = sub_cc(c0, bs0);
    uint32_t out1 = subc_cc(c1, bs1);
    uint32_t out2 = subc(c2, 0u);
#else
    uint32_t out0 = sub_cc(c0, 0u-borrow);
    uint32_t out1 = subc_cc(c1, 0u);
    uint32_t out2 = subc(c2, 0u);
#endif
    return make_uint3(out0, out1, out2);
}

__device__ __forceinline__ uint3 mul_mod(uint3 a, uint3 b){
    uint32_t a0 = a.x, a1 = a.y, a2 = a.z;
    uint32_t b0 = b.x, b1 = b.y, b2 = b.z;

#ifndef GPUPI_OLD_MUL96
    // Six-limb schoolbook product. Propagate carries through c5, including
    // the carry created before accumulating the final diagonal product.
    uint32_t c0, c1, c2, c3, c4, c5;
    asm volatile (
        "mul.lo.u32 %0, %6, %9;\n\t"
        "mul.hi.u32 %1, %6, %9;\n\t"
        "mad.lo.cc.u32 %1, %6, %10, %1;\n\t"
        "madc.hi.u32 %2, %6, %10, 0;\n\t"
        "mad.lo.cc.u32 %2, %6, %11, %2;\n\t"
        "madc.hi.u32 %3, %6, %11, 0;\n\t"
        "mad.lo.cc.u32 %1, %7, %9, %1;\n\t"
        "madc.hi.cc.u32 %2, %7, %9, %2;\n\t"
        "addc.u32 %3, %3, 0;\n\t"
        "mad.lo.cc.u32 %2, %7, %10, %2;\n\t"
        "madc.hi.cc.u32 %3, %7, %10, %3;\n\t"
        "addc.u32 %4, 0, 0;\n\t"
        "mad.lo.cc.u32 %3, %7, %11, %3;\n\t"
        "madc.hi.u32 %4, %7, %11, %4;\n\t"
        "mad.lo.cc.u32 %2, %8, %9, %2;\n\t"
        "madc.hi.cc.u32 %3, %8, %9, %3;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.u32 %5, 0, 0;\n\t"
        "mad.lo.cc.u32 %3, %8, %10, %3;\n\t"
        "madc.hi.cc.u32 %4, %8, %10, %4;\n\t"
        "addc.u32 %5, %5, 0;\n\t"
        "mad.lo.cc.u32 %4, %8, %11, %4;\n\t"
        "madc.hi.u32 %5, %8, %11, %5;\n\t"
        : "=&r"(c0), "=&r"(c1), "=&r"(c2), "=&r"(c3), "=&r"(c4), "=&r"(c5)
        : "r"(a0), "r"(a1), "r"(a2), "r"(b0), "r"(b1), "r"(b2));

#else
    uint64_t mul00 = (uint64_t)a0 * (uint64_t)b0;
    uint32_t c0 = (uint32_t)mul00;
    uint64_t mul01 = (uint64_t)a0 * (uint64_t)b1 + (mul00 >> 32);
    uint32_t c1 = (uint32_t)mul01;
    uint64_t mul02 = (uint64_t)a0 * (uint64_t)b2 + (mul01 >> 32);
    uint32_t c2 = (uint32_t)mul02;
    uint32_t c3 = (uint32_t)(mul02 >> 32);

    uint64_t mul10 = (uint64_t)a1 * (uint64_t)b0 + (uint64_t)c1;
    c1 = (uint32_t)mul10;
    uint64_t mul11 = (uint64_t)a1 * (uint64_t)b1 + (mul10 >> 32) + (uint64_t)c2;
    c2 = (uint32_t)mul11;
    uint64_t mul12 = (uint64_t)a1 * (uint64_t)b2 + (mul11 >> 32) + (uint64_t)c3;
    c3 = (uint32_t)mul12;
    uint32_t c4 = (uint32_t)(mul12 >> 32);

    uint64_t mul20 = (uint64_t)a2 * (uint64_t)b0 + (uint64_t)c2;
    c2 = (uint32_t)mul20;
    uint64_t mul21 = (uint64_t)a2 * (uint64_t)b1 + (mul20 >> 32) + (uint64_t)c3;
    c3 = (uint32_t)mul21;
    uint64_t mul22 = (uint64_t)a2 * (uint64_t)b2 + (mul21 >> 32) + (uint64_t)c4;
    c4 = (uint32_t)mul22;
    uint32_t c5 = (uint32_t)(mul22 >> 32);

#endif

    uint32_t d0 = sub_cc(0u, c3);
    uint32_t d1 = subc_cc(c3, c4);
    uint32_t d2 = subc_cc(c4, c5);
    uint32_t d3 = subc(c5, 0u);

    d0 = add_cc(d0, c0);
    d1 = addc_cc(d1, c1);
    d2 = addc_cc(d2, c2);
    d3 = addc(d3, 0u);

    uint32_t ds0, ds1;
    ds0 = sub_cc(0u, d3);
    ds1 = subc(d3, 0u);

    d0 = add_cc(d0, ds0);
    d1 = addc_cc(d1, ds1);
    d2 = addc_cc(d2, 0u);
    d3 = addc(0u, 0u);

#ifdef GPUPI_OLD_REDUCE96
    (void)add_cc(d0, 0xffffffffu);
    (void)addc_cc(d1, 0u);
    (void)addc_cc(d2, 0u);
    uint32_t add_one = addc(0u, 0u);
    d3 += add_one;

    ds0 = sub_cc(0u, d3);
    ds1 = subc(d3, 0u);

    uint32_t out0 = add_cc(d0, ds0);
    uint32_t out1 = addc_cc(d1, ds1);
    uint32_t out2 = addc(d2, 0u);
#else
    // The second fold adds less than 2^64. If it wrapped, its low 96 bits
    // are below 2^64 and cannot also require the >= p correction.
    const uint32_t correction = d3 | ((d1 & d2) == 0xffffffffu && d0 != 0u);
    uint32_t out0 = add_cc(d0, 0u-correction);
    uint32_t out1 = addc_cc(d1, 0u);
    uint32_t out2 = addc(d2, 0u);
#endif
    return make_uint3(out0, out1, out2);
}


// Exact modular division by 2^k, for 0 <= k <= 32 and canonical x.
// p == 1 (mod 2^k), so t = -x (mod 2^k) makes x+t*p divisible by 2^k.
// Form x+t*p = x+t-t*2^32+t*2^96 in four limbs, then shift once.
__device__ __forceinline__ uint3 scale_inverse_power2(uint3 x, uint32_t k) {
    if (k == 0) return x;
    const uint32_t mask = 0xffffffffu >> (32u-k);
    const uint32_t t = (0u-x.x) & mask;
    uint32_t c0,c1,c2,c3;
    asm volatile(
        "add.cc.u32 %0, %4, %7;\n\t"
        "addc.cc.u32 %1, %5, 0;\n\t"
        "addc.cc.u32 %2, %6, 0;\n\t"
        "addc.u32 %3, %7, 0;\n\t"
        "sub.cc.u32 %1, %1, %7;\n\t"
        "subc.cc.u32 %2, %2, 0;\n\t"
        "subc.u32 %3, %3, 0;\n\t"
        : "=&r"(c0),"=&r"(c1),"=&r"(c2),"=&r"(c3)
        : "r"(x.x),"r"(x.y),"r"(x.z),"r"(t));
    return make_uint3(__funnelshift_rc(c0,c1,k),__funnelshift_rc(c1,c2,k),__funnelshift_rc(c2,c3,k));
}
