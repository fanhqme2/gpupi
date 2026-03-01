#pragma once
#include "batch_mul_addsub_asm.h"
#include <cuda/warp>

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

template<int BLOCK_SIZE>
__device__ __forceinline__ void batch_mul_add_64_all_warp(
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
        for (int i = 1; i < blockDim.y - 1; i++){
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

template<int BLOCK_SIZE>
__device__ __forceinline__ void batch_mul_sub_128_single_warp(
    uint32_t & r0_value, uint32_t & r1_value, uint32_t & r2_value, uint32_t & r3_value,
    uint32_t & c0_value, uint32_t & c1_value, uint32_t & c2_value, uint32_t & c3_value
){
    uint32_t borrow_state;
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    r0_value = sub_cc(r0_value, c0_value);
    r1_value = subc_cc(r1_value, c1_value);
    r2_value = subc_cc(r2_value, c2_value);
    r3_value = subc_cc(r3_value, c3_value);
    borrow_state = -subc(0, 0); // 0 or 1
    sub_cc(r0_value, 1);
    subc_cc(r1_value, 0);
    subc_cc(r2_value, 0);
    subc_cc(r3_value, 0);
    borrow_state = (borrow_state << 1) -subc(0, 0);
    // borrow_state:  0   no borrow    2  borrow    1  depends on previous
    for (int delta = 1; delta < BLOCK_SIZE; delta *= 2){
        uint32_t prev_borrow = __shfl_up_sync(warp_mask, borrow_state, delta, BLOCK_SIZE);
        if (borrow_state == 1){
            borrow_state = prev_borrow;
        }
    }
    borrow_state = __shfl_up_sync(warp_mask, borrow_state, 1, BLOCK_SIZE);
    if (threadIdx.x == 0){
        borrow_state = 0;
    }
    r0_value = sub_cc(r0_value, borrow_state >> 1);
    r1_value = subc_cc(r1_value, 0);
    r2_value = subc_cc(r2_value, 0);
    r3_value = subc_cc(r3_value, 0);
}

template<int BLOCK_SIZE>
__device__ __forceinline__ void batch_mul_add_64_grouped_warp(
    uint32_t & r0_value, uint32_t & r1_value, uint32_t & c0_value, uint32_t & c1_value,
    int rank, int group_size, bool is_active,
    uint32_t * carry_prop
){
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    uint32_t carry_state;
    if (is_active){
        r0_value = add_cc(r0_value, c0_value);
        r1_value = addc_cc(r1_value, c1_value);

        carry_state = addc(0, 0);
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
            carry_prop[rank] = carry_state;
        }
    }
    __syncthreads();

    if (is_active){
        if (rank == 0){
            if (carry_prop[0] == 1){
                carry_prop[0] = 0;
            }
            for (int i = 1; i < group_size - 1; i++){
                if (carry_prop[i] == 1){
                    carry_prop[i] = carry_prop[i - 1];
                }
            }
        }
    }

    __syncthreads();
    if (is_active){
        if (carry_state == 1){
            if (rank > 0){
                carry_state = carry_prop[rank - 1];
            }else{
                carry_state = 0;
            }
        }
        carry_state = __shfl_up_sync(warp_mask, carry_state, 1, BLOCK_SIZE);
        if (threadIdx.x == 0){
            if (rank == 0){
                carry_state = 0;
            }else{
                carry_state = carry_prop[rank - 1];
            }
        }
        r0_value = add_cc(r0_value, carry_state >> 1);
        r1_value = addc_cc(r1_value, 0);
    }
}

template<int BLOCK_SIZE>
__device__ __forceinline__ void batch_mul_sub_64_grouped_warp(
    uint32_t & r0_value, uint32_t & r1_value, uint32_t & c0_value, uint32_t & c1_value,
    int rank, int group_size, bool is_active,
    uint32_t * carry_prop
){
    const unsigned int warp_mask = (1ull << BLOCK_SIZE) - 1;
    uint32_t borrow_state;
    if (is_active){
        r0_value = sub_cc(r0_value, c0_value);
        r1_value = subc_cc(r1_value, c1_value);

        borrow_state = -subc(0, 0);
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
            carry_prop[rank] = borrow_state;
        }
    }
    __syncthreads();

    if (is_active){
        if (rank == 0){
            if (carry_prop[0] == 1){
                carry_prop[0] = 0;
            }
            for (int i = 1; i < group_size - 1; i++){
                if (carry_prop[i] == 1){
                    carry_prop[i] = carry_prop[i - 1];
                }
            }
        }
    }

    __syncthreads();
    if (is_active){
        if (borrow_state == 1){
            if (rank > 0){
                borrow_state = carry_prop[rank - 1];
            }else{
                borrow_state = 0;
            }
        }
        borrow_state = __shfl_up_sync(warp_mask, borrow_state, 1, BLOCK_SIZE);
        if (threadIdx.x == 0){
            if (rank == 0){
                borrow_state = 0;
            }else{
                borrow_state = carry_prop[rank - 1];
            }
        }
        r0_value = sub_cc(r0_value, borrow_state >> 1);
        r1_value = subc_cc(r1_value, 0);
    }
}

template<int BLOCK_SIZE>
__device__ __forceinline__ void batch_mul_add3_64_grouped_warp(
    uint32_t & r0_value, uint32_t & r1_value, uint32_t & c0_value, uint32_t & c1_value,
    uint32_t & t0_value, uint32_t & t1_value,
    int rank, int group_size, bool is_active,
    ushort2 * carry_prop
){
    ushort2 carry;
    if (is_active){
        r0_value = add_cc(r0_value, c0_value);
        r1_value = addc_cc(r1_value, c1_value);
        carry.x = addc(0, 0);
        r0_value = add_cc(r0_value, t0_value);
        r1_value = addc_cc(r1_value, t1_value);
        carry.x += addc(0, 0);
        add_cc(r0_value, 2);
        addc_cc(r1_value, 0);
        if (addc(0, 0)){
            carry.y = r0_value & 3;
        }else{
            carry.y = 0;
        }
        for (int delta = 1; delta < 32; delta *= 2){
            ushort2 carry_prev = cuda::device::warp_shuffle_up<BLOCK_SIZE, ushort2>(carry, delta);
            if (threadIdx.x >= delta){
                ushort compound = carry.y + carry_prev.x;
                carry.x += compound >> 2;
                if ((compound & 3) == 3){
                    carry.y = carry_prev.y;
                }else{
                    carry.y = 0;
                }
            }
        }
        
        if (threadIdx.x == BLOCK_SIZE - 1){
            carry_prop[rank] = carry;
        }
    }
    __syncthreads();
    if (is_active && rank == 0 && threadIdx.x == 0){
        for (int i = 1; i < group_size - 1; i ++){
            carry_prop[i].x += (carry_prop[i].y + carry_prop[i - 1].x) >> 2;
        }
    }
    __syncthreads();
    if (is_active){
        if (rank > 0){
            ushort compound = carry_prop[rank - 1].x + carry.y;
            carry.x += compound >> 2;
        }
        carry = cuda::device::warp_shuffle_up<BLOCK_SIZE, ushort2>(carry, 1);
        if (threadIdx.x == 0){
            if (rank == 0){
                carry.x = 0;
            }else{
                carry.x = carry_prop[rank - 1].x;
            }
        }
        r0_value = add_cc(r0_value, (uint32_t)carry.x);
        r1_value = addc_cc(r1_value, 0);
    }
}

template<int BLOCK_SIZE>
__device__ __forceinline__ void batch_mul_sub3_64_grouped_warp(
    uint32_t & r0_value, uint32_t & r1_value,
    uint32_t & c0_value, uint32_t & c1_value,
    uint32_t & t0_value, uint32_t & t1_value,
    int rank, int group_size, bool is_active,
    ushort2 * carry_prop
){
    ushort2 borrow;
    if (is_active){
        r0_value = sub_cc(r0_value, c0_value);
        r1_value = subc_cc(r1_value, c1_value);
        borrow.x = -subc(0, 0);
        r0_value = sub_cc(r0_value, t0_value);
        r1_value = subc_cc(r1_value, t1_value);
        borrow.x -= subc(0, 0);
        sub_cc(r0_value, 2);
        subc_cc(r1_value, 0);
        if (subc(0, 0)){
            borrow.y = 3 - (r0_value & 3);
        }else{
            borrow.y = 0;
        }
        for (int delta = 1; delta < 32; delta *= 2){
            ushort2 borrow_prev = cuda::device::warp_shuffle_up<BLOCK_SIZE, ushort2>(borrow, delta);
            if (threadIdx.x >= delta){
                ushort compound = borrow.y + borrow_prev.x;
                borrow.x += compound >> 2;
                if ((compound & 3) == 3){
                    borrow.y = borrow_prev.y;
                }else{
                    borrow.y = 0;
                }
            }
        }
        if (threadIdx.x == BLOCK_SIZE - 1){
            carry_prop[rank] = borrow;
        }
    }
    __syncthreads();
    if (is_active && rank == 0 && threadIdx.x == 0){
        for (int i = 1; i < group_size - 1; i ++){
            carry_prop[i].x += (carry_prop[i].y + carry_prop[i - 1].x) >> 2;
        }
    }
    __syncthreads();
    if (is_active){
        if (rank > 0){
            ushort compound = carry_prop[rank - 1].x + borrow.y;
            borrow.x += compound >> 2;
        }
        borrow = cuda::device::warp_shuffle_up<BLOCK_SIZE, ushort2>(borrow, 1);
        if (threadIdx.x == 0){
            if (rank == 0){
                borrow.x = 0;
            }else{
                borrow = carry_prop[rank - 1];
            }
        }
        r0_value = sub_cc(r0_value, (uint32_t)borrow.x);
        r1_value = subc_cc(r1_value, 0);
    }
}
