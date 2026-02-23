#pragma once
#include "batch_mul_addsub_asm.h"

template<int BLOCK_SIZE>
__device__ __forceinline__ void batch_mul_add_64_single_warp(
    uint32_t & r0_value, uint32_t & r1_value, uint32_t & c0_value, uint32_t & c1_value
){
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    r0_value = add_cc(r0_value, c0_value);
    r1_value = addc_cc(r1_value, c1_value);
    uint32_t carry_state = addc(0, 0);
    add_cc(1, r0_value);
    addc_cc(0, r1_value);
    carry_state = addc(carry_state, carry_state);
    // carry_state:  0   no carry    2  carry    1  depends on previous
    for (int delta = 1; delta < BLOCK_SIZE; delta *= 2){
        uint32_t prev_carry = __shfl_up_sync(warp_mask, carry_state, delta, BLOCK_SIZE);
        if (carry_state == 1){
            carry_state = prev_carry;
        }
    }
    carry_state = __shfl_up_sync(warp_mask, carry_state, 1, BLOCK_SIZE);
    if (threadIdx.x == 0){
        carry_state = 0;
    }
    r0_value = add_cc(r0_value, carry_state >> 1);
    r1_value = addc_cc(r1_value, 0);
}

template<int BLOCK_SIZE, int N_BLOCKS>
__device__ __forceinline__ void batch_mul_add_64_multi_warp(
    uint32_t & r0_value, uint32_t & r1_value, uint32_t & c0_value, uint32_t & c1_value, uint32_t carry_prop[]
){
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    r0_value = add_cc(r0_value, c0_value);
    r1_value = addc_cc(r1_value, c1_value);

    uint32_t carry_state = addc(0, 0);
    add_cc(1, r0_value);
    addc_cc(0, r1_value);
    carry_state = addc(carry_state, carry_state);

    // carry_state:  0   no carry    2  carry    1  depends on previous
    for (int delta = 1; delta < BLOCK_SIZE; delta *= 2){
        uint32_t prev_carry = __shfl_up_sync(warp_mask, carry_state, delta, BLOCK_SIZE);
        if (carry_state == 1){
            carry_state = prev_carry;
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1){
        carry_prop[threadIdx.y] = carry_state;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0){
        if (carry_prop[0] == 1){
            carry_prop[0] = 0;
        }
        for (int i = 1; i < N_BLOCKS - 1; i++){
            if (carry_prop[i] == 1){
                carry_prop[i] = carry_prop[i - 1];
            }
        }
    }

    __syncthreads();
    if (carry_state == 1){
        if (threadIdx.y > 0){
            carry_state = carry_prop[threadIdx.y - 1];
        }else{
            carry_state = 0;
        }
    }
    carry_state = __shfl_up_sync(warp_mask, carry_state, 1, BLOCK_SIZE);
    if (threadIdx.x == 0){
        if (threadIdx.y == 0){
            carry_state = 0;
        }else{
            carry_state = carry_prop[threadIdx.y - 1];
        }
    }
    r0_value = add_cc(r0_value, carry_state >> 1);
    r1_value = addc_cc(r1_value, 0);
}

template<int BLOCK_SIZE, int N_BLOCKS>
__device__ __forceinline__ void batch_mul_sub_64_multi_warp(
    uint32_t & r0_value, uint32_t & r1_value, uint32_t & c0_value, uint32_t & c1_value, uint32_t carry_prop[]
){
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    r0_value = sub_cc(r0_value, c0_value);
    r1_value = subc_cc(r1_value, c1_value);

    uint32_t borrow_state = -subc(0, 0);
    sub_cc(r0_value, 1);
    subc_cc(r1_value, 0);
    borrow_state = (borrow_state << 1) -subc(0, 0);

    // borrow_state:  0   no borrow    2  borrow    1  depends on previous
    for (int delta = 1; delta < BLOCK_SIZE; delta *= 2){
        uint32_t prev_borrow = __shfl_up_sync(warp_mask, borrow_state, delta, BLOCK_SIZE);
        if (borrow_state == 1){
            borrow_state = prev_borrow;
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1){
        carry_prop[threadIdx.y] = borrow_state;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0){
        if (carry_prop[0] == 1){
            carry_prop[0] = 0;
        }
        for (int i = 1; i < N_BLOCKS - 1; i++){
            if (carry_prop[i] == 1){
                carry_prop[i] = carry_prop[i - 1];
            }
        }
    }

    __syncthreads();
    if (borrow_state == 1){
        if (threadIdx.y > 0){
            borrow_state = carry_prop[threadIdx.y - 1];
        }else{
            borrow_state = 0;
        }
    }
    borrow_state = __shfl_up_sync(warp_mask, borrow_state, 1, BLOCK_SIZE);
    if (threadIdx.x == 0){
        if (threadIdx.y == 0){
            borrow_state = 0;
        }else{
            borrow_state = carry_prop[threadIdx.y - 1];
        }
    }
    r0_value = sub_cc(r0_value, borrow_state >> 1);
    r1_value = subc_cc(r1_value, 0);
}