#pragma once

// Add with carry-out (sets carry flag)
__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Add with carry-in and carry-out (uses and sets carry flag)
__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Add with carry-in (uses carry flag, no output carry)
__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Subtract with borrow-out (sets carry flag)
__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Subtract with borrow-in and borrow-out (uses and sets carry flag)
__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Subtract with borrow-in (uses carry flag, no output borrow)
__device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}