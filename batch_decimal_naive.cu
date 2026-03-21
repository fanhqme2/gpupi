#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>

#include "batch_decimal_naive.h"
#include "batch_mul_addsub_warp.h"

namespace {

constexpr int kDecimalNaiveLimbMax = 2047;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 1024;

template<int max_chunk>
__global__ void batch_decimal_naive_kernel(
    const uint32_t * A,
    char * B,
    uint32_t N,
    int L_limb,
    uint32_t stride_A,
    int L_dec,
    uint32_t stride_B
) {
    __shared__ uint32_t prop[max_chunk / 64];
    __shared__ uint32_t out_value;

    const int my_idx = threadIdx.x + threadIdx.y * blockDim.x;
    const int power_10[9] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    for (uint32_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        uint32_t val0 = (my_idx * 2 < L_limb) ? A[(size_t)idx * stride_A + my_idx * 2] : 0u;
        uint32_t val1 = (my_idx * 2 + 1 < L_limb) ? A[(size_t)idx * stride_A + my_idx * 2 + 1] : 0u;
        __syncthreads();

        for (int start_idx = 0; start_idx < L_dec; start_idx += 9) {
            const int limb_base = my_idx * 2;
            const uint64_t mul0 = (uint64_t)val0 * 1000000000ULL;
            const uint64_t mul1 = (uint64_t)val1 * 1000000000ULL;
            uint32_t next_prop = (uint32_t)(mul1 >> 32);
            uint32_t muln1_hi = cuda::device::warp_shuffle_up<32, uint32_t>(next_prop, 1);
            if (threadIdx.x == 31){
                prop[threadIdx.y] = next_prop;
            }
            __syncthreads();
            if (threadIdx.x == 0){
                muln1_hi  = (threadIdx.y > 0) ? prop[threadIdx.y - 1] : 0u;
            }
            __syncthreads();

            uint32_t r0value = (uint32_t)mul0;
            uint32_t r1value = (uint32_t)mul1;
            uint32_t c0value = (uint32_t)muln1_hi;
            uint32_t c1value = (uint32_t)(mul0 >> 32);

            if (max_chunk <= 32){
                batch_mul_add_64_single_warp<32>(r0value, r1value, c0value, c1value);
            }else{
                batch_mul_add_64_all_warp<32>(r0value, r1value, c0value, c1value, prop);
            }

            if (limb_base < L_limb) {
                val0 = r0value;
            }
            if (limb_base + 1 < L_limb) {
                val1 = r1value;
            }
            
            if (limb_base == L_limb) {
                out_value = r0value;
            }
            if (limb_base + 1 == L_limb) {
                out_value = r1value;
            }
            __syncthreads();
            const int out_idx_len = min(9, L_dec - start_idx);
            if (my_idx < out_idx_len){
                const int out_pos = L_dec - start_idx - out_idx_len;
                B[(size_t)idx * stride_B + out_pos + my_idx] = out_value / power_10[my_idx + (9 - out_idx_len)] % 10u + '0';
            }
            __syncthreads();
        }
    }
}

template<int max_warps>
__global__ void batch_decimal_naive_kernel2(
    const uint32_t * A,
    char * B,
    uint32_t N,
    int L_limb,
    uint32_t stride_A,
    int L_dec,
    uint32_t stride_B
) {
    __shared__ uint32_t prop[max_warps * 2];
    __shared__ uint32_t read_buf[32];
    const int my_idx = threadIdx.x + threadIdx.y * blockDim.x;
    for (uint32_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        uint32_t val0 = 0;
        uint32_t val1 = 0;
        uint32_t val0_2 = 0;
        uint32_t val1_2 = 0;
        uint32_t carry = 0;
        uint32_t carry_2 = 0;
        uint32_t ret = 0;
        uint32_t ret_2 = 0;

        for (int i = 0; i < ((L_limb + 2) >> 1) + ((L_dec + 17) / 18); i ++){
            uint32_t send_val0 = ((i - 1 - my_idx) * 2 < L_limb) ? val0_2 : 0;
            uint32_t send_val1 = ((i - 1 - my_idx) * 2 + 1 < L_limb) ? val1_2 : 0;
            val0 = cuda::device::warp_shuffle_up(send_val0, 1);
            val1 = cuda::device::warp_shuffle_up(send_val1, 1);

            if (blockDim.y > 1){
                if (threadIdx.x == 31){
                    prop[threadIdx.y * 2] = send_val0;
                    prop[threadIdx.y * 2 + 1] = send_val1;
                }
                __syncthreads();
                if (threadIdx.x == 0 && threadIdx.y > 0){
                    val0 = (threadIdx.y > 0) ? prop[(threadIdx.y - 1) * 2] : 0u;
                    val1 = (threadIdx.y > 0) ? prop[(threadIdx.y - 1) * 2 + 1] : 0u;
                }
                __syncthreads();
            }
            if (threadIdx.y == 0){
                if ((i & 15) == 0){
                    read_buf[threadIdx.x] = (i * 2 + threadIdx.x < L_limb) ? A[(size_t)idx * stride_A + i * 2 + threadIdx.x] : 0;
                }
                (void)cuda::device::warp_shuffle_up(0, 0);
                if (threadIdx.x == 0){
                    val0 = read_buf[(i & 15) * 2 + 0];
                    val1 = read_buf[(i & 15) * 2 + 1];
                }
            }

            uint64_t mul0 = (uint64_t)val0 * 1000000000ULL + carry;
            uint64_t mul1 = (uint64_t)val1 * 1000000000ULL + (mul0 >> 32);
            carry = mul1 >> 32;
            val0 = (uint32_t)mul0;
            val1 = (uint32_t)mul1;

            mul0 = (uint64_t)((i - my_idx) * 2 < L_limb ? val0 : 0) * 1000000000ULL + carry_2;
            mul1 = (uint64_t)((i - my_idx) * 2 + 1 < L_limb ? val1 : 0) * 1000000000ULL + (mul0 >> 32);
            carry_2 = mul1 >> 32;
            val0_2 = (uint32_t)mul0;
            val1_2 = (uint32_t)mul1;

            if ((i - my_idx) * 2 == L_limb){
                ret = val0;
                ret_2 = val0_2;
            }
            if ((i - my_idx) * 2 + 1 == L_limb){
                ret = val1;
                ret_2 = val1_2;
            }
        }
        for (int i = 0; i < 9; i ++){
            char digit = '0' + ret % 10u;
            ret /= 10;
            const int out_pos = L_dec - (my_idx * 2 + 1) * 9 + i;
            if (out_pos >= 0 && out_pos < L_dec){
                B[(size_t)idx * stride_B + out_pos] = digit;
            }
        }
        for (int i = 0; i < 9; i ++){
            char digit = '0' + ret_2 % 10u;
            ret_2 /= 10;
            const int out_pos = L_dec - (my_idx * 2 + 2) * 9 + i;
            if (out_pos >= 0 && out_pos < L_dec){
                B[(size_t)idx * stride_B + out_pos] = digit;
            }
        }
    }
}

}  // namespace

void batch_decimal_naive(
    const uint32_t * A,
    char * B,
    uint32_t N,
    int L_limb,
    uint32_t stride_A,
    int L_dec,
    uint32_t stride_B
) {
    if (N == 0 || L_limb <= 0 || L_dec <= 0) {
        return;
    }

    assert(stride_A >= (uint32_t)L_limb);
    assert(stride_B >= (uint32_t)L_dec);
    assert(L_limb <= kDecimalNaiveLimbMax);

    if (L_dec > 18 * 1024){
        const int threads_needed = (L_limb / 2) + 1;
        const int warps_per_block = (threads_needed + kWarpSize - 1) / kWarpSize;
        assert(threads_needed <= kThreadsPerBlock);
        
        const dim3 block(kWarpSize, (unsigned)warps_per_block, 1u);
        const uint32_t num_blocks = std::min<uint32_t>(N, 65535u);

        batch_decimal_naive_kernel<2048><<<num_blocks, block>>>(
            A, B, N, L_limb, stride_A, L_dec, stride_B
        );
    }else{
        const int threads_needed = (L_dec + 17) / 18;
        const int warps_per_block = (threads_needed + kWarpSize - 1) / kWarpSize;
        const dim3 block(kWarpSize, (unsigned)warps_per_block, 1u);
        const uint32_t num_blocks = std::min<uint32_t>(N, 65535u);
        batch_decimal_naive_kernel2<32><<<num_blocks, block>>>(
            A, B, N, L_limb, stride_A, L_dec, stride_B
        );
    }
}
