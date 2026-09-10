#include "batch_mul_ntt.h"
#include "batch_mul_addsub_asm.h"
#include "batch_mul_addsub_warp.h"
#include <cuda/warp>
#include <cstdlib>

const uint32_t arr_root0 = 0x46038aeb;
const uint32_t arr_root1 = 0x55c21a87;
const uint32_t arr_root2 = 0xa257c693;

const uint32_t arr_root65536_0 = 0xea78dff7;
const uint32_t arr_root65536_1 = 0xbcc6f614;
const uint32_t arr_root65536_2 = 0x2c17ad1;

const uint32_t arr_inv2_0 = 0x80000001;
const uint32_t arr_inv2_1 = 0xffffffff;
const uint32_t arr_inv2_2 = 0x7fffffff;

#include "ntt_mod96.cuh"

__device__ uint32_t bitrev16(uint32_t idx){
    return __brev(idx) >> 16;
}

__global__ void fill_in_power_table(uint3 * table, int n, uint3 root){
    if (threadIdx.x == 0){
        table[0] = make_uint3(1, 0, 0);
    }
    __syncthreads();
    for (int l = 1; l < n; l += l){
        for (int i = l + threadIdx.x; i < l + l; i += blockDim.x){
            table[i] = mul_mod(table[i - l], root);
        }
        root = mul_mod(root, root);
        __syncthreads();
    }
}

__global__ void fill_in_power_table_linear(uint3 * table, int n, uint3 root){
    if (threadIdx.x == 0){
        uint3 value = make_uint3(1, 0, 0);
        for (int i = 0; i < n; ++i){
            table[i] = value;
            value = mul_mod(value, root);
        }
    }
}

__global__ void fill_in_power_table_bitrev16(uint3 * table, int n, uint3 root){
    if (threadIdx.x == 0){
        table[0] = make_uint3(1, 0, 0);
    }
    __syncthreads();
    for (int l = 1; l < n; l += l){
        for (int i = l + threadIdx.x; i < l + l; i += blockDim.x){
            table[bitrev16(i)] = mul_mod(table[bitrev16(i - l)], root);
        }
        root = mul_mod(root, root);
        __syncthreads();
    }
}

void init_ntt_precomputed_tables(NTTPrecomputedTables * tables){
    fill_in_power_table_bitrev16<<<1, 1024>>>(tables->roots_table_lv1, 65536, make_uint3(arr_root0, arr_root1, arr_root2));
    fill_in_power_table_bitrev16<<<1, 1024>>>(tables->roots_table_lv2, 65536, make_uint3(arr_root65536_0, arr_root65536_1, arr_root65536_2));
    fill_in_power_table_linear<<<1, 1>>>(tables->inv2n_table, 33, make_uint3(arr_inv2_0, arr_inv2_1, arr_inv2_2));
}

__global__ void fft_level_forward(uint3 * parts, int k, int i, const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2, size_t N){
    uint32_t step = 1 << (k - 1 - i);
    uint32_t seq_len = 1 << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 1)); j += blockDim.x * gridDim.x){
        size_t offset = (j >> (k - 1 - i)) << (k - i);
        uint32_t seq_id = (j >> (k - 1 - i)) & (seq_len - 1);
        uint32_t step_id = j & (step - 1);
        //uint32_t bitrev_seq_id = __brev((uint32_t)seq_id * 2);
        uint32_t bitrev_seq_id = (uint32_t)seq_id * 2;
        uint3 twiddle_factor = mul_mod(
            roots_table_lv1[bitrev_seq_id >> 16],
            roots_table_lv2[bitrev_seq_id & 0xffff]
        );
        uint3 u = parts[offset +        step_id];
        uint3 v = parts[offset + step + step_id];
        uint3 w = mul_mod(v, twiddle_factor);
        v = sub_mod(u, w);
        u = add_mod(u, w);
        parts[offset +        step_id] = u;
        parts[offset + step + step_id] = v;
    }
}

__global__ void fft_level_forward_radix16(
    uint3 *parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N
){
    const uint32_t step = 1u << (k - 4 - i);
    const uint32_t seq_len = 1u << i;

    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 4)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 4 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));

        uint32_t bitrev_seq_id0 = (uint32_t)seq_id * 2;
        uint32_t bitrev_seq_id10 = (uint32_t)(seq_id * 2 + 0) * 2;
        uint32_t bitrev_seq_id11 = (uint32_t)(seq_id * 2 + 1) * 2;
        uint32_t bitrev_seq_id20 = (uint32_t)(seq_id * 4 + 0) * 2;
        uint32_t bitrev_seq_id21 = (uint32_t)(seq_id * 4 + 1) * 2;
        uint32_t bitrev_seq_id22 = (uint32_t)(seq_id * 4 + 2) * 2;
        uint32_t bitrev_seq_id23 = (uint32_t)(seq_id * 4 + 3) * 2;

        uint3 w0 = mul_mod(roots_table_lv1[bitrev_seq_id0 >> 16], roots_table_lv2[bitrev_seq_id0 & 0xffff]);
        uint3 w10 = mul_mod(roots_table_lv1[bitrev_seq_id10 >> 16], roots_table_lv2[bitrev_seq_id10 & 0xffff]);
        uint3 w11 = mul_mod(roots_table_lv1[bitrev_seq_id11 >> 16], roots_table_lv2[bitrev_seq_id11 & 0xffff]);
        uint3 w20 = mul_mod(roots_table_lv1[bitrev_seq_id20 >> 16], roots_table_lv2[bitrev_seq_id20 & 0xffff]);
        uint3 w21 = mul_mod(roots_table_lv1[bitrev_seq_id21 >> 16], roots_table_lv2[bitrev_seq_id21 & 0xffff]);
        uint3 w22 = mul_mod(roots_table_lv1[bitrev_seq_id22 >> 16], roots_table_lv2[bitrev_seq_id22 & 0xffff]);
        uint3 w23 = mul_mod(roots_table_lv1[bitrev_seq_id23 >> 16], roots_table_lv2[bitrev_seq_id23 & 0xffff]);

        uint3 w3[8];
        #pragma unroll
        for (int g = 0; g < 8; g++){
            uint32_t bitrev_seq_id3 = (uint32_t)(seq_id * 8 + g) * 2;
            w3[g] = mul_mod(roots_table_lv1[bitrev_seq_id3 >> 16], roots_table_lv2[bitrev_seq_id3 & 0xffff]);
        }

        size_t base = offset + step_id;

        uint3 x[16];
        #pragma unroll
        for (int t = 0; t < 16; t++){
            x[t] = parts[base + (size_t)step * t];
        }

        // stage i
        uint3 a[16];
        #pragma unroll
        for (int t = 0; t < 8; t++){
            uint3 wx = mul_mod(x[t + 8], w0);
            a[t] = add_mod(x[t], wx);
            a[t + 8] = sub_mod(x[t], wx);
        }

        // stage i+1
        uint3 b[16];
        #pragma unroll
        for (int g = 0; g < 2; g++){
            uint3 w = (g == 0) ? w10 : w11;
            #pragma unroll
            for (int t = 0; t < 4; t++){
                int idx = g * 8 + t;
                uint3 wa = mul_mod(a[idx + 4], w);
                b[idx] = add_mod(a[idx], wa);
                b[idx + 4] = sub_mod(a[idx], wa);
            }
        }

        // stage i+2
        uint3 c[16];
        #pragma unroll
        for (int g = 0; g < 4; g++){
            uint3 w = (g == 0) ? w20 : (g == 1 ? w21 : (g == 2 ? w22 : w23));
            #pragma unroll
            for (int t = 0; t < 2; t++){
                int idx = g * 4 + t;
                uint3 wb = mul_mod(b[idx + 2], w);
                c[idx] = add_mod(b[idx], wb);
                c[idx + 2] = sub_mod(b[idx], wb);
            }
        }

        // stage i+3
        #pragma unroll
        for (int g = 0; g < 8; g++){
            int idx = g * 2;
            uint3 wc = mul_mod(c[idx + 1], w3[g]);
            parts[base + (size_t)step * idx] = add_mod(c[idx], wc);
            parts[base + (size_t)step * (idx + 1)] = sub_mod(c[idx], wc);
        }
    }
}

__global__ void fft_level_forward_radix16_initial(
    uint32_t * A, uint32_t * B, uint3 * parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N, uint32_t L_a, uint32_t L_b, uint32_t stride_A, uint32_t stride_B,
    bool square = false
){
    const uint32_t step = 1u << (k - 4 - i);
    size_t N_half = square ? N : N >> 1;

    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 4)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 4 - i);
        size_t offset = group << (k - i);
        uint32_t step_id = (uint32_t)(j & (step - 1));

        uint3 w0 = make_uint3(1, 0, 0);
        uint3 w10 = w0;
        uint3 w11 = roots_table_lv2[2];
        uint3 w20 = w0;
        uint3 w21 = w11;
        uint3 w22 = roots_table_lv2[4];
        uint3 w23 = roots_table_lv2[6];

        uint3 w3[8];
        w3[0] = w0;
        w3[1] = w11;
        w3[2] = w22;
        w3[3] = w23;
        w3[4] = roots_table_lv2[8];
        w3[5] = roots_table_lv2[10];
        w3[6] = roots_table_lv2[12];
        w3[7] = roots_table_lv2[14];

        size_t base = offset + step_id;

        uint3 x[16];
        #pragma unroll
        for (int t = 0; t < 16; t++){
            if (group < N_half){
                size_t offset_input = group * stride_A;
                uint32_t idx = step_id + (size_t)step * t;
                x[t] = make_uint3(idx < L_a ? A[offset_input + idx] : 0, 0, 0);
            }else{
                size_t offset_input = (group - N_half) * stride_B;
                uint32_t idx = step_id + (size_t)step * t;
                x[t] = make_uint3(idx < L_b ? B[offset_input + idx] : 0, 0, 0);
            }
        }

        // stage i
        uint3 a[16];
        #pragma unroll
        for (int t = 0; t < 8; t++){
            uint3 wx = mul_mod(x[t + 8], w0);
            a[t] = add_mod(x[t], wx);
            a[t + 8] = sub_mod(x[t], wx);
        }

        // stage i+1
        uint3 b[16];
        #pragma unroll
        for (int g = 0; g < 2; g++){
            uint3 w = (g == 0) ? w10 : w11;
            #pragma unroll
            for (int t = 0; t < 4; t++){
                int idx = g * 8 + t;
                uint3 wa = mul_mod(a[idx + 4], w);
                b[idx] = add_mod(a[idx], wa);
                b[idx + 4] = sub_mod(a[idx], wa);
            }
        }

        // stage i+2
        uint3 c[16];
        #pragma unroll
        for (int g = 0; g < 4; g++){
            uint3 w = (g == 0) ? w20 : (g == 1 ? w21 : (g == 2 ? w22 : w23));
            #pragma unroll
            for (int t = 0; t < 2; t++){
                int idx = g * 4 + t;
                uint3 wb = mul_mod(b[idx + 2], w);
                c[idx] = add_mod(b[idx], wb);
                c[idx + 2] = sub_mod(b[idx], wb);
            }
        }

        // stage i+3
        #pragma unroll
        for (int g = 0; g < 8; g++){
            int idx = g * 2;
            uint3 wc = mul_mod(c[idx + 1], w3[g]);
            parts[base + (size_t)step * idx] = add_mod(c[idx], wc);
            parts[base + (size_t)step * (idx + 1)] = sub_mod(c[idx], wc);
        }
    }
}

__global__ void fft_level_forward_radix32(
    uint3 *parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N
){
    __shared__ uint3 local_coefs[32][32];
    const uint32_t step = 1u << (k - 5 - i);
    const uint32_t seq_len = 1u << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 5)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 5 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));
        for (int t = threadIdx.y; t < 32; t += blockDim.y){
            local_coefs[t][threadIdx.x] = parts[offset + step_id + (size_t)step * t];
        }
        __syncthreads();
        for (int lv = 4; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.y; t < 16; t += blockDim.y){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (4 - lv)) | (idx >> (lv + 1))) * 2;
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                /*uint3 twiddle_factor = mul_mod(
                    roots_table_lv1[bitrev_seq_id >> 16],
                    roots_table_lv2[bitrev_seq_id & 0xffff]
                );*/
                uint3 u = local_coefs[idx][threadIdx.x];
                uint3 v = local_coefs[idx + stride][threadIdx.x];
                uint3 w = mul_mod(v, twiddle_factor);
                local_coefs[idx][threadIdx.x] = add_mod(u, w);
                local_coefs[idx + stride][threadIdx.x] = sub_mod(u, w);
            }
            __syncthreads();
        }
        for (int t = threadIdx.y; t < 32; t += blockDim.y){
             parts[offset + step_id + (size_t)step * t] = local_coefs[t][threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void fft_level_forward_radix32_initial(
    uint32_t * A, uint32_t * B, uint3 *parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N, uint32_t L_a, uint32_t L_b, uint32_t stride_A, uint32_t stride_B,
    bool square = false
){
    __shared__ uint3 local_coefs[32][32];
    const uint32_t step = 1u << (k - 5 - i);
    const uint32_t seq_len = 1u << i;
    size_t N_half = square ? N : N >> 1;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 5)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 5 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));
        for (int t = threadIdx.y; t < 32; t += blockDim.y){
            if (group < N_half){
                size_t offset_input = group * stride_A;
                uint32_t idx = step_id + (size_t)step * t;
                local_coefs[t][threadIdx.x] = make_uint3(idx < L_a ? A[offset_input + idx] : 0, 0, 0);
            }else{
                size_t offset_input = (group - N_half) * stride_B;
                uint32_t idx = step_id + (size_t)step * t;
                local_coefs[t][threadIdx.x] = make_uint3(idx < L_b ? B[offset_input + idx] : 0, 0, 0);
            }
        }
        __syncthreads();
        for (int lv = 4; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.y; t < 16; t += blockDim.y){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (4 - lv)) | (idx >> (lv + 1))) * 2;
                /*uint3 twiddle_factor = mul_mod(
                    roots_table_lv1[bitrev_seq_id >> 16],
                    roots_table_lv2[bitrev_seq_id & 0xffff]
                );*/
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                uint3 u = local_coefs[idx][threadIdx.x];
                uint3 v = local_coefs[idx + stride][threadIdx.x];
                uint3 w = mul_mod(v, twiddle_factor);
                local_coefs[idx][threadIdx.x] = add_mod(u, w);
                local_coefs[idx + stride][threadIdx.x] = sub_mod(u, w);
            }
            __syncthreads();
        }
        for (int t = threadIdx.y; t < 32; t += blockDim.y){
             parts[offset + step_id + (size_t)step * t] = local_coefs[t][threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void fft_level_forward_radix256(
    uint3 *parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N
){
    __shared__ uint3 local_coefs[256][16];
    const uint32_t step = 1u << (k - 8 - i);
    const uint32_t seq_len = 1u << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 8)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 8 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));
        for (int t = threadIdx.y; t < 256; t += blockDim.y){
            local_coefs[t][threadIdx.x] = parts[offset + step_id + (size_t)step * t];
        }
        __syncthreads();
        for (int lv = 7; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.y; t < 128; t += blockDim.y){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (7 - lv)) | (idx >> (lv + 1))) * 2;
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                /*uint3 twiddle_factor = mul_mod(
                    roots_table_lv1[bitrev_seq_id >> 16],
                    roots_table_lv2[bitrev_seq_id & 0xffff]
                );*/
                uint3 u = local_coefs[idx][threadIdx.x];
                uint3 v = local_coefs[idx + stride][threadIdx.x];
                uint3 w = mul_mod(v, twiddle_factor);
                local_coefs[idx][threadIdx.x] = add_mod(u, w);
                local_coefs[idx + stride][threadIdx.x] = sub_mod(u, w);
            }
            __syncthreads();
        }
        for (int t = threadIdx.y; t < 256; t += blockDim.y){
             parts[offset + step_id + (size_t)step * t] = local_coefs[t][threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void fft_level_forward_radix256_initial(
    uint32_t * A, uint32_t * B, uint3 *parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N, uint32_t L_a, uint32_t L_b, uint32_t stride_A, uint32_t stride_B,
    bool square = false
){
    __shared__ uint3 local_coefs[256][16];
    const uint32_t step = 1u << (k - 8 - i);
    const uint32_t seq_len = 1u << i;
    size_t N_half = square ? N : N >> 1;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 8)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 8 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));
        for (int t = threadIdx.y; t < 256; t += blockDim.y){
            if (group < N_half){
                size_t offset_input = group * stride_A;
                uint32_t idx = step_id + (size_t)step * t;
                local_coefs[t][threadIdx.x] = make_uint3(idx < L_a ? A[offset_input + idx] : 0, 0, 0);
            }else{
                size_t offset_input = (group - N_half) * stride_B;
                uint32_t idx = step_id + (size_t)step * t;
                local_coefs[t][threadIdx.x] = make_uint3(idx < L_b ? B[offset_input + idx] : 0, 0, 0);
            }
        }
        __syncthreads();
        for (int lv = 7; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.y; t < 128; t += blockDim.y){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (7 - lv)) | (idx >> (lv + 1))) * 2;
                /*uint3 twiddle_factor = mul_mod(
                    roots_table_lv1[bitrev_seq_id >> 16],
                    roots_table_lv2[bitrev_seq_id & 0xffff]
                );*/
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                uint3 u = local_coefs[idx][threadIdx.x];
                uint3 v = local_coefs[idx + stride][threadIdx.x];
                uint3 w = mul_mod(v, twiddle_factor);
                local_coefs[idx][threadIdx.x] = add_mod(u, w);
                local_coefs[idx + stride][threadIdx.x] = sub_mod(u, w);
            }
            __syncthreads();
        }
        for (int t = threadIdx.y; t < 256; t += blockDim.y){
             parts[offset + step_id + (size_t)step * t] = local_coefs[t][threadIdx.x];
        }
        __syncthreads();
    }
}

template<int max_local_size>
__global__ void fft_level_forward_final(
    uint3 * parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N
){
    uint32_t local_size = 1u << (k - i);
    uint32_t seq_len = 1u << i;
    __shared__ uint3 local_coefs[max_local_size];
    for (size_t j = ((size_t)blockIdx.x) << (k - i); j < (N << k); j += gridDim.x << (k - i)){
        uint32_t seq_id = (j >> (k - i)) & (seq_len - 1);
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs[t] = parts[j + t];
        }
        __syncthreads();
        for (int lv = k - i - 1; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size >> 1; t += blockDim.x){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (k - i - 1 - lv)) | (idx >> (lv + 1))) * 2;
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                /*uint3 twiddle_factor = mul_mod(
                    roots_table_lv1[bitrev_seq_id >> 16],
                    roots_table_lv2[bitrev_seq_id & 0xffff]
                );*/
                uint3 u = local_coefs[idx];
                uint3 v = local_coefs[idx + stride];
                uint3 w = mul_mod(v, twiddle_factor);
                local_coefs[idx] = add_mod(u, w);
                local_coefs[idx + stride] = sub_mod(u, w);
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            parts[j + t] = local_coefs[t];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ uint3 normalize_inverse(uint3 x, unsigned k, const uint3 *scalar) {
#ifdef GPUPI_OLD_NTT_SCALE
    return mul_mod(x, *scalar);
#else
    return scale_inverse_power2(x, k);
#endif
}

__global__ void pointwise_mul(uint3 * parts_a, uint3 * parts_b, size_t total, uint3 * scalar_ptr, int k){
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < total; j += blockDim.x * gridDim.x){
        parts_a[j] = normalize_inverse(mul_mod(parts_a[j], parts_b[j]), k, scalar_ptr);
    }
}

__global__ void pointwise_mul_broadcast(uint3 *a, const uint3 *b, size_t total,
    size_t K, const uint3 *scalar_ptr, int k) {
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < total;
         j += (size_t)blockDim.x * gridDim.x)
        a[j] = normalize_inverse(mul_mod(a[j], b[j & (K - 1)]), k, scalar_ptr);
}

// Keep both local forward transforms and their product in shared memory.
// When the inverse schedule uses an equal or smaller tile, do its local
// stages here as well, avoiding intermediate global-memory round trips.
template<int max_local_size, bool cached_b = false, bool broadcast_b = false>
__global__ void fft_local_product(
    uint3 * parts_a, const uint3 * parts_b, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N, const uint3 * scalar_ptr, int backward_levels
){
    const int depth = k - i;
    const uint32_t local_size = 1u << depth;
    __shared__ uint3 a[max_local_size];
    __shared__ uint3 b[cached_b ? 1 : max_local_size];
    for (size_t j = (size_t)blockIdx.x << depth; j < (N << k);
         j += (size_t)gridDim.x << depth){
        const uint32_t seq_id = (j >> depth) & ((1u << i) - 1u);
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            a[t] = parts_a[j + t];
            if constexpr (!cached_b) b[t] = parts_b[j + t];
        }
        __syncthreads();
        for (int lv = depth - 1; lv >= 0; --lv){
            const uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size / 2; t += blockDim.x){
                const int idx = ((t >> lv) << lv) + t;
                const uint32_t root_idx = ((seq_id << (depth - 1 - lv)) | (idx >> (lv + 1))) * 2;
                uint3 w = roots_table_lv2[root_idx & 0xffff];
                if (root_idx >= 65536) w = mul_mod(w, roots_table_lv1[root_idx >> 16]);
                const uint3 ua = a[idx];
                const uint3 va = mul_mod(a[idx + stride], w);
                a[idx] = add_mod(ua, va);
                a[idx + stride] = sub_mod(ua, va);
                if constexpr (!cached_b) {
                    const uint3 ub = b[idx], vb = mul_mod(b[idx + stride], w);
                    b[idx] = add_mod(ub, vb);
                    b[idx + stride] = sub_mod(ub, vb);
                }
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            const uint3 multiplier = cached_b
                ? parts_b[(broadcast_b ? (j & (((size_t)1 << k) - 1)) : j) + t] : b[t];
            a[t] = normalize_inverse(mul_mod(a[t], multiplier), k, scalar_ptr);
        }
        __syncthreads();
        for (int lv = 0; lv < backward_levels; ++lv){
            const uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size / 2; t += blockDim.x){
                const int idx = ((t >> lv) << lv) + t;
                uint32_t root_idx = ((seq_id << (depth - 1 - lv)) | (idx >> (lv + 1))) * 2;
                root_idx = __brev(-__brev(root_idx));
                uint3 w = roots_table_lv2[root_idx & 0xffff];
                if (root_idx >= 65536) w = mul_mod(w, roots_table_lv1[root_idx >> 16]);
                const uint3 u = a[idx], v = a[idx + stride];
                a[idx] = add_mod(u, v);
                a[idx + stride] = mul_mod(sub_mod(u, v), w);
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            parts_a[j + t] = a[t];
        }
        __syncthreads();
    }
}

__global__ void fft_level_backward(uint3 * parts, int k, int i, const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2, size_t N){
    uint32_t step = 1 << (k - 1 - i);
    uint32_t seq_len = 1 << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 1)); j += blockDim.x * gridDim.x){
        size_t offset = (j >> (k - 1 - i)) << (k - i);
        uint32_t seq_id = (j >> (k - 1 - i)) & (seq_len - 1);
        uint32_t step_id = j & (step - 1);
        uint32_t bitrev_seq_id = __brev(-__brev(seq_id * 2));
        uint3 twiddle_factor = mul_mod(
            roots_table_lv1[bitrev_seq_id >> 16],
            roots_table_lv2[bitrev_seq_id & 0xffff]
        );
        uint3 u = parts[offset +        step_id];
        uint3 v = parts[offset + step + step_id];
        uint3 w = sub_mod(u, v);
        u = add_mod(u, v);
        v = mul_mod(w, twiddle_factor);
        parts[offset +        step_id] = u;
        parts[offset + step + step_id] = v;
    }
}

__global__ void fft_level_backward_radix16(uint3 * parts, int k, int i, const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2, size_t N){
    i -= 3;
    uint32_t step = 1u << (k - 4 - i);
    uint32_t seq_len = 1u << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 4)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 4 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));

        uint32_t bitrev_seq_id0 = __brev(-__brev(seq_id * 2));
        uint32_t bitrev_seq_id10 = __brev(-__brev(seq_id * 4 + 0));
        uint32_t bitrev_seq_id11 = __brev(-__brev(seq_id * 4 + 2));
        uint32_t bitrev_seq_id20 = __brev(-__brev(seq_id * 8 + 0));
        uint32_t bitrev_seq_id21 = __brev(-__brev(seq_id * 8 + 2));
        uint32_t bitrev_seq_id22 = __brev(-__brev(seq_id * 8 + 4));
        uint32_t bitrev_seq_id23 = __brev(-__brev(seq_id * 8 + 6));

        uint3 w0 = mul_mod(roots_table_lv1[bitrev_seq_id0 >> 16], roots_table_lv2[bitrev_seq_id0 & 0xffff]);
        uint3 w10 = mul_mod(roots_table_lv1[bitrev_seq_id10 >> 16], roots_table_lv2[bitrev_seq_id10 & 0xffff]);
        uint3 w11 = mul_mod(roots_table_lv1[bitrev_seq_id11 >> 16], roots_table_lv2[bitrev_seq_id11 & 0xffff]);
        uint3 w20 = mul_mod(roots_table_lv1[bitrev_seq_id20 >> 16], roots_table_lv2[bitrev_seq_id20 & 0xffff]);
        uint3 w21 = mul_mod(roots_table_lv1[bitrev_seq_id21 >> 16], roots_table_lv2[bitrev_seq_id21 & 0xffff]);
        uint3 w22 = mul_mod(roots_table_lv1[bitrev_seq_id22 >> 16], roots_table_lv2[bitrev_seq_id22 & 0xffff]);
        uint3 w23 = mul_mod(roots_table_lv1[bitrev_seq_id23 >> 16], roots_table_lv2[bitrev_seq_id23 & 0xffff]);

        uint3 w3[8];
        #pragma unroll
        for (int g = 0; g < 8; g++){
            uint32_t bitrev_seq_id3 = __brev(-__brev(seq_id * 16 + g * 2));
            w3[g] = mul_mod(roots_table_lv1[bitrev_seq_id3 >> 16], roots_table_lv2[bitrev_seq_id3 & 0xffff]);
        }

        size_t base = offset + step_id;

        uint3 y[16];
        #pragma unroll
        for (int t = 0; t < 16; t++){
            y[t] = parts[base + (size_t)step * t];
        }

        // stage i+3
        uint3 a[16];
        #pragma unroll
        for (int g = 0; g < 8; g++){
            int idx = g * 2;
            a[idx] = add_mod(y[idx], y[idx + 1]);
            a[idx + 1] = mul_mod(sub_mod(y[idx], y[idx + 1]), w3[g]);
        }

        // stage i+2
        uint3 b[16];
        #pragma unroll
        for (int g = 0; g < 4; g++){
            uint3 w = (g == 0) ? w20 : (g == 1 ? w21 : (g == 2 ? w22 : w23));
            #pragma unroll
            for (int t = 0; t < 2; t++){
                int idx = g * 4 + t;
                b[idx] = add_mod(a[idx], a[idx + 2]);
                b[idx + 2] = mul_mod(sub_mod(a[idx], a[idx + 2]), w);
            }
        }

        // stage i+1
        uint3 c[16];
        #pragma unroll
        for (int g = 0; g < 2; g++){
            uint3 w = (g == 0) ? w10 : w11;
            #pragma unroll
            for (int t = 0; t < 4; t++){
                int idx = g * 8 + t;
                c[idx] = add_mod(b[idx], b[idx + 4]);
                c[idx + 4] = mul_mod(sub_mod(b[idx], b[idx + 4]), w);
            }
        }

        // stage i
        #pragma unroll
        for (int t = 0; t < 8; t++){
            parts[base + (size_t)step * t] = add_mod(c[t], c[t + 8]);
            parts[base + (size_t)step * (t + 8)] = mul_mod(sub_mod(c[t], c[t + 8]), w0);
        }
    }
}

__global__ void fft_level_backward_radix32(
    uint3 *parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N
){
    __shared__ uint3 local_coefs[32][32];
    i -= 4;
    const uint32_t step = 1u << (k - 5 - i);
    const uint32_t seq_len = 1u << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 5)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 5 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));

        for (int t = threadIdx.y; t < 32; t += blockDim.y){
            local_coefs[t][threadIdx.x] = parts[offset + step_id + (size_t)step * t];
        }
        __syncthreads();

        for (int lv = 0; lv <= 4; lv++){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.y; t < 16; t += blockDim.y){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (4 - lv)) | (idx >> (lv + 1))) * 2;
                bitrev_seq_id = __brev(-__brev(bitrev_seq_id));
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                uint3 u = local_coefs[idx][threadIdx.x];
                uint3 v = local_coefs[idx + stride][threadIdx.x];
                local_coefs[idx][threadIdx.x] = add_mod(u, v);
                local_coefs[idx + stride][threadIdx.x] = mul_mod(sub_mod(u, v), twiddle_factor);
            }
            __syncthreads();
        }

        for (int t = threadIdx.y; t < 32; t += blockDim.y){
            parts[offset + step_id + (size_t)step * t] = local_coefs[t][threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void fft_level_backward_radix256(
    uint3 *parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N
){
    __shared__ uint3 local_coefs[256][16];
    i -= 7;
    const uint32_t step = 1u << (k - 8 - i);
    const uint32_t seq_len = 1u << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 8)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 8 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));

        for (int t = threadIdx.y; t < 256; t += blockDim.y){
            local_coefs[t][threadIdx.x] = parts[offset + step_id + (size_t)step * t];
        }
        __syncthreads();

        for (int lv = 0; lv <= 7; lv++){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.y; t < 128; t += blockDim.y){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (7 - lv)) | (idx >> (lv + 1))) * 2;
                bitrev_seq_id = __brev(-__brev(bitrev_seq_id));
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                uint3 u = local_coefs[idx][threadIdx.x];
                uint3 v = local_coefs[idx + stride][threadIdx.x];
                local_coefs[idx][threadIdx.x] = add_mod(u, v);
                local_coefs[idx + stride][threadIdx.x] = mul_mod(sub_mod(u, v), twiddle_factor);
            }
            __syncthreads();
        }

        for (int t = threadIdx.y; t < 256; t += blockDim.y){
            parts[offset + step_id + (size_t)step * t] = local_coefs[t][threadIdx.x];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ ushort2 combine_carry(ushort2 a, ushort2 b){
    ushort compound = a.y + b.x;
    a.x += compound >> 2;
    if ((compound & 3) == 3){
        a.y = b.y;
    }else{
        a.y = 0;
    }
    return a;
}

__host__ __device__ static size_t carry_stride(int k, size_t L) {
    size_t K = (size_t)1 << k;
    return ((L > K ? L : K) + 31) / 32 * 2;
}

__global__ void fft_level_backward_radix32_initial(
    uint3 *parts, int k,
    uint32_t * ret, uint32_t * workspace,
    const uint3 * __restrict__ roots_table_lv2,
    size_t N, size_t L, uint32_t stride_ret,
    const uint3 *tail = nullptr, int tail_k = 0, uint32_t wrap = 0
){
    __shared__ uint3 local_coefs[32][32];
    uint32_t step = 1u << (k - 5);
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 5)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 5);
        size_t offset = group << k;
        uint32_t step_id = (uint32_t)(j & (step - 1));

        for (int t = threadIdx.y; t < 32; t += blockDim.y){
            local_coefs[t][threadIdx.x] = parts[offset + step_id + (size_t)step * t];
        }
        __syncthreads();

        for (int lv = 0; lv <= 4; lv++){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.y; t < 16; t += blockDim.y){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = (idx >> (lv + 1)) * 2;
                bitrev_seq_id = __brev(-__brev(bitrev_seq_id));
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                uint3 u = local_coefs[idx][threadIdx.x];
                uint3 v = local_coefs[idx + stride][threadIdx.x];
                local_coefs[idx][threadIdx.x] = add_mod(u, v);
                local_coefs[idx + stride][threadIdx.x] = mul_mod(sub_mod(u, v), twiddle_factor);
            }
            __syncthreads();
        }

        if (tail) {
            for (int t = threadIdx.y; t < 32; t += blockDim.y) {
                uint32_t idx = step_id + (size_t)step * t;
                if (idx + 1 < wrap)
                    local_coefs[t][threadIdx.x] = sub_mod(local_coefs[t][threadIdx.x],
                        tail[((size_t)group << tail_k) + wrap - 2 - idx]);
            }
            __syncthreads();
        }
        for (int t = threadIdx.y; t < 32; t += blockDim.y){
            uint32_t r0_value = local_coefs[t][threadIdx.x].x;
            uint32_t c0_value = (threadIdx.x != 0) ? local_coefs[t][threadIdx.x - 1].y : 0;
            uint32_t t0_value = (threadIdx.x > 1) ? local_coefs[t][threadIdx.x - 2].z : 0;

            ushort2 carry;
            r0_value = add_cc(r0_value, c0_value);
            carry.x = addc(0, 0);
            r0_value = add_cc(r0_value, t0_value);
            carry.x += addc(0, 0);
            add_cc(r0_value, 2);
            if (addc(0, 0)){
                carry.y = r0_value & 3;
            }else{
                carry.y = 0;
            }
            for (int delta = 1; delta < 32; delta *= 2){
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
                if (threadIdx.x >= delta){
                    carry = combine_carry(carry, carry_prev);
                }
            }
            ushort2 prev_carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if (threadIdx.x == 0){
                prev_carry = make_ushort2(0, 0);
            }
            r0_value += prev_carry.x;
            uint32_t idx = step_id + (size_t)step * t;
            if (idx < L){
                ret[(size_t)group * stride_ret + idx] = r0_value;
            }
            if (threadIdx.x == 31){
                r0_value = local_coefs[t][31].y;
                uint32_t r1_value = local_coefs[t][31].z;
                r0_value = add_cc(r0_value, local_coefs[t][30].z);
                r1_value = addc_cc(r1_value, 0);
                r0_value = add_cc(r0_value, carry.x);
                r1_value = addc_cc(r1_value, 0);
                workspace[((size_t)group * carry_stride(k, L)) + (idx >> 4) - 1] = r0_value;
                workspace[((size_t)group * carry_stride(k, L)) + (idx >> 4)] = r1_value;
            }
        }
        __syncthreads();
    }
}

__global__ void fft_level_backward_radix16_initial(
    uint3 *parts, int k,
    uint32_t * ret, uint32_t * workspace,
    const uint3 * __restrict__ roots_table_lv2,
    size_t N, size_t L, uint32_t stride_ret,
    const uint3 *tail = nullptr, int tail_k = 0, uint32_t wrap = 0
){
    uint32_t step = 1u << (k - 4);
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 4)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 4);
        size_t offset = group << k;
        uint32_t step_id = (uint32_t)(j & (step - 1));

        uint32_t bitrev_seq_id0 = __brev(-__brev(0));
        uint32_t bitrev_seq_id10 = __brev(-__brev(0));
        uint32_t bitrev_seq_id11 = __brev(-__brev(2));
        uint32_t bitrev_seq_id20 = __brev(-__brev(0));
        uint32_t bitrev_seq_id21 = __brev(-__brev(2));
        uint32_t bitrev_seq_id22 = __brev(-__brev(4));
        uint32_t bitrev_seq_id23 = __brev(-__brev(6));

        uint3 w0 = roots_table_lv2[bitrev_seq_id0 & 0xffff];
        uint3 w10 = roots_table_lv2[bitrev_seq_id10 & 0xffff];
        uint3 w11 = roots_table_lv2[bitrev_seq_id11 & 0xffff];
        uint3 w20 = roots_table_lv2[bitrev_seq_id20 & 0xffff];
        uint3 w21 = roots_table_lv2[bitrev_seq_id21 & 0xffff];
        uint3 w22 = roots_table_lv2[bitrev_seq_id22 & 0xffff];
        uint3 w23 = roots_table_lv2[bitrev_seq_id23 & 0xffff];

        uint3 w3[8];
        #pragma unroll
        for (int g = 0; g < 8; g++){
            uint32_t bitrev_seq_id3 = __brev(-__brev(g * 2));
            w3[g] = roots_table_lv2[bitrev_seq_id3 & 0xffff];
        }

        size_t base = offset + step_id;

        uint3 y[16];
        #pragma unroll
        for (int t = 0; t < 16; t++){
            y[t] = parts[base + (size_t)step * t];
        }

        // stage i+3
        uint3 a[16];
        #pragma unroll
        for (int g = 0; g < 8; g++){
            int idx = g * 2;
            a[idx] = add_mod(y[idx], y[idx + 1]);
            a[idx + 1] = mul_mod(sub_mod(y[idx], y[idx + 1]), w3[g]);
        }

        // stage i+2
        uint3 b[16];
        #pragma unroll
        for (int g = 0; g < 4; g++){
            uint3 w = (g == 0) ? w20 : (g == 1 ? w21 : (g == 2 ? w22 : w23));
            #pragma unroll
            for (int t = 0; t < 2; t++){
                int idx = g * 4 + t;
                b[idx] = add_mod(a[idx], a[idx + 2]);
                b[idx + 2] = mul_mod(sub_mod(a[idx], a[idx + 2]), w);
            }
        }

        // stage i+1
        uint3 c[16];
        #pragma unroll
        for (int g = 0; g < 2; g++){
            uint3 w = (g == 0) ? w10 : w11;
            #pragma unroll
            for (int t = 0; t < 4; t++){
                int idx = g * 8 + t;
                c[idx] = add_mod(b[idx], b[idx + 4]);
                c[idx + 4] = mul_mod(sub_mod(b[idx], b[idx + 4]), w);
            }
        }

        uint3 d[16];

        // stage i
        #pragma unroll
        for (int t = 0; t < 8; t++){
            d[t] = add_mod(c[t], c[t + 8]);
            d[t+8] = mul_mod(sub_mod(c[t], c[t + 8]), w0);
        }

        for (int t = 0; t < 16; t ++){
            uint3 dt = d[t];
            uint32_t coefficient_idx = step_id + (size_t)step * t;
            if (tail && coefficient_idx + 1 < wrap)
                dt = sub_mod(dt, tail[((size_t)group << tail_k) + wrap - 2 - coefficient_idx]);
            uint32_t r0_value = dt.x;
            uint32_t d_prev_y = cuda::device::warp_shuffle_up<32>(dt.y, 1);
            uint32_t d_prev_z = cuda::device::warp_shuffle_up<32>(dt.z, 1);
            uint32_t d_prev2_z = cuda::device::warp_shuffle_up<32>(dt.z, 2);
            uint32_t c0_value = ((threadIdx.x & 31) != 0) ? d_prev_y : 0;
            uint32_t t0_value = ((threadIdx.x & 31) > 1) ? d_prev2_z : 0;

            ushort2 carry;
            r0_value = add_cc(r0_value, c0_value);
            carry.x = addc(0, 0);
            r0_value = add_cc(r0_value, t0_value);
            carry.x += addc(0, 0);
            add_cc(r0_value, 2);
            if (addc(0, 0)){
                carry.y = r0_value & 3;
            }else{
                carry.y = 0;
            }
            for (int delta = 1; delta < 32; delta *= 2){
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
                if ((threadIdx.x & 31) >= delta){
                    carry = combine_carry(carry, carry_prev);
                }
            }
            ushort2 prev_carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if ((threadIdx.x & 31) == 0){
                prev_carry = make_ushort2(0, 0);
            }
            r0_value += prev_carry.x;
            uint32_t idx = step_id + (size_t)step * t;
            if (idx < L){
                ret[(size_t)group * stride_ret + idx] = r0_value;
            }
            if ((threadIdx.x & 31) == 31){
                r0_value = dt.y;
                uint32_t r1_value = dt.z;
                r0_value = add_cc(r0_value, d_prev_z);
                r1_value = addc_cc(r1_value, 0);
                r0_value = add_cc(r0_value, carry.x);
                r1_value = addc_cc(r1_value, 0);
                workspace[((size_t)group * carry_stride(k, L)) + (idx >> 4) - 1] = r0_value;
                workspace[((size_t)group * carry_stride(k, L)) + (idx >> 4)] = r1_value;
            }
        }
        __syncthreads();
    }
}

template<int max_local_size>
__global__ void fft_level_backward_final(
    uint3 * parts, int k, int i,
    const uint3 * __restrict__ roots_table_lv1, const uint3 * __restrict__ roots_table_lv2,
    size_t N
){
    uint32_t local_size = 1u << (k - i);
    uint32_t seq_len = 1u << i;
    __shared__ uint3 local_coefs[max_local_size];
    for (size_t j = ((size_t)blockIdx.x) << (k - i); j < (N << k); j += gridDim.x << (k - i)){
        uint32_t seq_id = (j >> (k - i)) & (seq_len - 1);
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs[t] = parts[j + t];
        }
        __syncthreads();
        for (int lv = 0; lv < k - i; lv ++){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size >> 1; t += blockDim.x){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = ((seq_id << (k - i - 1 - lv)) | (idx >> (lv + 1))) * 2;
                bitrev_seq_id = __brev(-__brev(bitrev_seq_id));
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                if (bitrev_seq_id >= 65536){
                    twiddle_factor = mul_mod(twiddle_factor, roots_table_lv1[bitrev_seq_id >> 16]);
                }
                /*uint3 twiddle_factor = mul_mod(
                    roots_table_lv1[bitrev_seq_id >> 16],
                    roots_table_lv2[bitrev_seq_id & 0xffff]
                );*/
                uint3 u = local_coefs[idx];
                uint3 v = local_coefs[idx + stride];
                local_coefs[idx] = add_mod(u, v);
                local_coefs[idx + stride] = mul_mod(sub_mod(u, v), twiddle_factor);
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            parts[j + t] = local_coefs[t];
        }
        __syncthreads();
    }
}

__global__ void add3_single_block(uint3 * parts, uint32_t * ret, size_t N, int k, size_t L, uint32_t stride_ret){
    __shared__ ushort2 carryInfo[32];
    parts += ((size_t)blockIdx.x) << k;
    ret += ((size_t)blockIdx.x) * stride_ret;
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        ushort block_carry = 0;
        for (size_t i0 = 0; i0 < L; i0 += blockDim.x * blockDim.y * 2){
            uint32_t i = i0 + (threadIdx.y * 32 + threadIdx.x) * 2;
            uint32_t r0_value, r1_value;
            uint32_t c0_value, c1_value;
            uint32_t t0_value, t1_value;
            r0_value = parts[i].x;
            r1_value = parts[i + 1].x;
            c0_value = (i == 0) ? 0 : parts[i - 1].y;
            c1_value = parts[i].y;
            t0_value = (i <= 1) ? 0 : parts[i - 2].z;
            t1_value = (i == 0) ? 0 : parts[i - 1].z;
            ushort2 carry;

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
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
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
            if (threadIdx.x == 32 - 1){
                carryInfo[threadIdx.y] = carry;
            }
            __syncthreads();

            if (threadIdx.y == 0){
                ushort2 bc = carryInfo[threadIdx.x];
                for (int delta = 1; delta < blockDim.y; delta *= 2){
                    ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(bc, delta);
                    if (threadIdx.x >= delta){
                        ushort compound = bc.y + carry_prev.x;
                        bc.x += compound >> 2;
                        if ((compound & 3) == 3){
                            bc.y = carry_prev.y;
                        }else{
                            bc.y = 0;
                        }
                    }
                }
                ushort compound = bc.y + block_carry;
                bc.x += compound >> 2;
                bc.y = 0;
                carryInfo[threadIdx.x] = bc;
            }
            __syncthreads();

            ushort compound = carry.y + ((threadIdx.y > 0) ? carryInfo[threadIdx.y - 1].x : block_carry);
            carry.x += compound >> 2;
            carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if (threadIdx.x == 0){
                if (threadIdx.y == 0){
                    carry.x = block_carry;
                }else{
                    carry.x = carryInfo[threadIdx.y - 1].x;
                }
            }
            r0_value = add_cc(r0_value, (uint32_t)carry.x);
            r1_value = addc_cc(r1_value, 0);

            if (i < L){
                ret[i] = r0_value;
            }
            if (i + 1 < L){
                ret[i + 1] = r1_value;
            }
            block_carry = carryInfo[31].x;
            __syncthreads();
        }
        parts += ((size_t)gridDim.x) << k;
        ret += ((size_t)stride_ret) * gridDim.x;
    }
}

__global__ void add2_combine_blocks(ushort2 * ret_carry, uint32_t N, uint32_t L){
    __shared__ ushort2 carryInfo[32];
    const int block_size = 2048;
    uint32_t blocks_per_num = (L + block_size - 1) / block_size;
    ret_carry += ((size_t)blockIdx.x) * blocks_per_num;
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        ushort block_carry = 0;
        for (int i0 = 0; i0 < blocks_per_num; i0 += 1024){
            int i = i0 + threadIdx.x + threadIdx.y * 32;
            ushort2 carry = (i < blocks_per_num) ? ret_carry[i] : make_ushort2(0, 0);
            for (int delta = 1; delta < 32; delta *= 2){
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
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
            if (threadIdx.x == 32 - 1){
                carryInfo[threadIdx.y] = carry;
            }
            __syncthreads();
            if (threadIdx.y == 0){
                ushort2 bc = carryInfo[threadIdx.x];
                for (int delta = 1; delta < blockDim.y; delta *= 2){
                    ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(bc, delta);
                    if (threadIdx.x >= delta){
                        ushort compound = bc.y + carry_prev.x;
                        bc.x += compound >> 2;
                        if ((compound & 3) == 3){
                            bc.y = carry_prev.y;
                        }else{
                            bc.y = 0;
                        }
                    }
                }
                ushort compound = bc.y + block_carry;
                bc.x += compound >> 2;
                bc.y = 0;
                carryInfo[threadIdx.x] = bc;
            }
            __syncthreads();
            ushort compound = carry.y + ((threadIdx.y > 0) ? carryInfo[threadIdx.y - 1].x : block_carry);
            carry.x += compound >> 2;
            carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if (threadIdx.x == 0){
                if (threadIdx.y == 0){
                    carry.x = block_carry;
                }else{
                    carry.x = carryInfo[threadIdx.y - 1].x;
                }
            }
            if (i < blocks_per_num){
                ret_carry[i] = carry;
            }
            block_carry = carryInfo[31].x;
            __syncthreads();
        }
        ret_carry += blocks_per_num * gridDim.x;
    }
}

__global__ void add2_apply_blocks(uint32_t * ret, ushort2 * ret_carry, size_t N, int k, size_t L, uint32_t stride_ret){
    __shared__ uint32_t carryInfo[32];
    const int block_size = 2048;
    uint32_t blocks_per_num = (L + block_size - 1) / block_size;
    ret_carry += ((size_t)blockIdx.y) * blocks_per_num;
    ret += ((size_t)blockIdx.y) * stride_ret;
    for (int idx = blockIdx.y; idx < N; idx += gridDim.y){
        for (size_t i0 = blockIdx.x * block_size; i0 < L; i0 += block_size * gridDim.x){       
            uint32_t block_carry = ret_carry[i0 / block_size].x;     
            if (block_carry){
                uint32_t i = i0 + (threadIdx.y * 32 + threadIdx.x) * 2;
                uint32_t r0_value = (i < L) ? ret[i] : 0;
                uint32_t r1_value = (i + 1 < L) ? ret[i + 1] : 0;
                uint32_t c0_value = (i == i0) ? block_carry : 0;
                uint32_t c1_value = 0;
                const uint32_t r0_value_old = r0_value;
                const uint32_t r1_value_old = r1_value;
                batch_mul_add_64_all_warp<32>(r0_value, r1_value, c0_value, c1_value, carryInfo);

                if (r0_value != r0_value_old || r1_value != r1_value_old || c0_value != 0u || c1_value != 0u){
                    if (i < L){
                        ret[i] = r0_value;
                    }
                    if (i + 1 < L){
                        ret[i + 1] = r1_value;
                    }
                }
                __syncthreads();
            }
        }
        ret_carry += blocks_per_num * gridDim.y;
        ret += ((size_t)stride_ret) * gridDim.y;
    }
}

__global__ void add2_reduce_blocks(uint32_t * ret, uint32_t * workspace, ushort2 * ret_carry, size_t N, int k, size_t L, uint32_t stride_ret){
    __shared__ ushort2 carryInfo[32];
    ret += ((size_t)blockIdx.y) * stride_ret;
    const int block_size = 2048;
    uint32_t blocks_per_num = (L + block_size - 1) / block_size;
    ret_carry += ((size_t)blockIdx.y) * blocks_per_num;
    workspace += (size_t)blockIdx.y * carry_stride(k, L);
    for (int idx = blockIdx.y; idx < N; idx += gridDim.y){
        for (size_t i0 = blockIdx.x * block_size; i0 < L; i0 += block_size * gridDim.x){
            uint32_t i = i0 + (threadIdx.y * 32 + threadIdx.x) * 2;
            uint32_t r0_value, r1_value;
            uint32_t c0_value, c1_value;
            r0_value = (i < L) ? ret[i] : 0;
            r1_value = (i < L - 1) ? ret[i + 1] : 0;

            c0_value = (i >= 32 && i < L && (i & 31) < 2) ? workspace[(((i >> 5) - 1) << 1) | (i & 1)] : 0;
            c1_value = (i >= 32 && i < L - 1 && ((i + 1) & 31) < 2) ? workspace[(((i >> 5) - 1) << 1) | ((i + 1) & 1)] : 0;
            ushort2 carry;

            r0_value = add_cc(r0_value, c0_value);
            r1_value = addc_cc(r1_value, c1_value);
            carry.x = addc(0, 0);
            add_cc(r0_value, 2);
            addc_cc(r1_value, 0);
            if (addc(0, 0)){
                carry.y = r0_value & 3;
            }else{
                carry.y = 0;
            }
            for (int delta = 1; delta < 32; delta *= 2){
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
                if (threadIdx.x >= delta){
                    carry = combine_carry(carry, carry_prev);
                }
            }
            if (threadIdx.x == 32 - 1){
                carryInfo[threadIdx.y] = carry;
            }
            __syncthreads();

            if (threadIdx.y == 0){
                ushort2 bc = carryInfo[threadIdx.x];
                for (int delta = 1; delta < blockDim.y; delta *= 2){
                    ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(bc, delta);
                    if (threadIdx.x >= delta){
                        bc = combine_carry(bc, carry_prev);
                    }
                }
                if (threadIdx.x == 31){
                    ret_carry[i0 / block_size] = bc;
                }
                bc.y = 0;
                carryInfo[threadIdx.x] = bc;
            }
            __syncthreads();

            ushort compound = carry.y + ((threadIdx.y > 0) ? carryInfo[threadIdx.y - 1].x : 0);
            carry.x += compound >> 2;
            carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if (threadIdx.x == 0){
                if (threadIdx.y == 0){
                    carry.x = 0;
                }else{
                    carry.x = carryInfo[threadIdx.y - 1].x;
                }
            }
            r0_value = add_cc(r0_value, (uint32_t)carry.x);
            r1_value = addc_cc(r1_value, 0);

            if (i < L){
                ret[i] = r0_value;
            }
            if (i + 1 < L){
                ret[i + 1] = r1_value;
            }
            __syncthreads();
        }
        ret += ((size_t)stride_ret) * gridDim.y;
        ret_carry += blocks_per_num * gridDim.y;
        workspace += (size_t)gridDim.y * carry_stride(k, L);
    }
}

__global__ void add2_single_block(uint32_t * ret, uint32_t * workspace, size_t N, int k, size_t L, uint32_t stride_ret){
    __shared__ ushort2 carryInfo[32];
    ret += ((size_t)blockIdx.x) * stride_ret;
    workspace += (size_t)blockIdx.x * carry_stride(k, L);
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x){
        ushort block_carry = 0;
        for (size_t i0 = 0; i0 < L; i0 += blockDim.x * blockDim.y * 2){
            uint32_t i = i0 + (threadIdx.y * 32 + threadIdx.x) * 2;
            uint32_t r0_value, r1_value;
            uint32_t c0_value, c1_value;
            r0_value = (i < L) ? ret[i] : 0;
            r1_value = (i < L - 1) ? ret[i + 1] : 0;

            c0_value = (i >= 32 && i < L && (i & 31) < 2) ? workspace[(((i >> 5) - 1) << 1) | (i & 1)] : 0;
            c1_value = (i >= 32 && i < L - 1 && ((i + 1) & 31) < 2) ? workspace[(((i >> 5) - 1) << 1) | ((i + 1) & 1)] : 0;
            ushort2 carry;

            r0_value = add_cc(r0_value, c0_value);
            r1_value = addc_cc(r1_value, c1_value);
            carry.x = addc(0, 0);
            add_cc(r0_value, 2);
            addc_cc(r1_value, 0);
            if (addc(0, 0)){
                carry.y = r0_value & 3;
            }else{
                carry.y = 0;
            }
            for (int delta = 1; delta < 32; delta *= 2){
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
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
            if (threadIdx.x == 32 - 1){
                carryInfo[threadIdx.y] = carry;
            }
            __syncthreads();

            if (threadIdx.y == 0){
                ushort2 bc = carryInfo[threadIdx.x];
                for (int delta = 1; delta < blockDim.y; delta *= 2){
                    ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(bc, delta);
                    if (threadIdx.x >= delta){
                        ushort compound = bc.y + carry_prev.x;
                        bc.x += compound >> 2;
                        if ((compound & 3) == 3){
                            bc.y = carry_prev.y;
                        }else{
                            bc.y = 0;
                        }
                    }
                }
                ushort compound = bc.y + block_carry;
                bc.x += compound >> 2;
                bc.y = 0;
                carryInfo[threadIdx.x] = bc;
            }
            __syncthreads();

            ushort compound = carry.y + ((threadIdx.y > 0) ? carryInfo[threadIdx.y - 1].x : block_carry);
            carry.x += compound >> 2;
            carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if (threadIdx.x == 0){
                if (threadIdx.y == 0){
                    carry.x = block_carry;
                }else{
                    carry.x = carryInfo[threadIdx.y - 1].x;
                }
            }
            r0_value = add_cc(r0_value, (uint32_t)carry.x);
            r1_value = addc_cc(r1_value, 0);

            if (i < L){
                ret[i] = r0_value;
            }
            if (i + 1 < L){
                ret[i + 1] = r1_value;
            }
            block_carry = carryInfo[31].x;
            __syncthreads();
        }
        ret += ((size_t)stride_ret) * gridDim.x;
        workspace += (size_t)gridDim.x * carry_stride(k, L);
    }
}

template<int max_local_size, bool broadcast_b = false>
__global__ void mul_fft_local(uint32_t * A, uint32_t * B, uint32_t * ret, const uint3 * __restrict__ roots_table_lv2, uint3 * inv2n, uint32_t N, uint32_t L_a, uint32_t L_b, uint32_t stride_A, uint32_t stride_B, uint32_t stride_ret, int k, const uint3 *cached_b = nullptr){
    __shared__ uint3 local_coefs_a[max_local_size], local_coefs_b[max_local_size];
    //__shared__ ushort2 carry_prop[32];
    ushort2 * carry_prop = (ushort2 *)local_coefs_b;
    uint32_t local_size = 1u << k;
    size_t L = L_a + L_b;
    A += ((size_t)blockIdx.x) * stride_A;
    B += ((size_t)blockIdx.x) * stride_B;
    ret += ((size_t)blockIdx.x) * stride_ret;
    for (uint32_t j = blockIdx.x; j < N; j += gridDim.x){
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs_a[t] = make_uint3((t < L_a) ? A[t] : 0, 0, 0);
            local_coefs_b[t] = broadcast_b ? cached_b[t] : make_uint3((t < L_b) ? B[t] : 0, 0, 0);
        }
        __syncthreads();
        for (int lv = k - 1; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size >> 1; t += blockDim.x){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = (idx >> (lv + 1)) * 2;
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                uint3 u_a = local_coefs_a[idx];
                uint3 v_a = local_coefs_a[idx + stride];
                uint3 w_a = mul_mod(v_a, twiddle_factor);
                local_coefs_a[idx] = add_mod(u_a, w_a);
                local_coefs_a[idx + stride] = sub_mod(u_a, w_a);

                if constexpr (!broadcast_b) {
                    uint3 u_b = local_coefs_b[idx];
                    uint3 v_b = local_coefs_b[idx + stride];
                    uint3 w_b = mul_mod(v_b, twiddle_factor);
                    local_coefs_b[idx] = add_mod(u_b, w_b);
                    local_coefs_b[idx + stride] = sub_mod(u_b, w_b);
                }
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs_a[t] = normalize_inverse(mul_mod(local_coefs_a[t], local_coefs_b[t]), k, inv2n);
        }
        __syncthreads();
        for (int lv = 0; lv < k; lv++){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size >> 1; t += blockDim.x){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = (idx >> (lv + 1)) * 2;
                bitrev_seq_id = __brev(-__brev(bitrev_seq_id));
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                uint3 u = local_coefs_a[idx];
                uint3 v = local_coefs_a[idx + stride];
                local_coefs_a[idx] = add_mod(u, v);
                local_coefs_a[idx + stride] = mul_mod(sub_mod(u, v), twiddle_factor);
            }
            __syncthreads();
        }
        uint32_t r0_value, r1_value;
        uint32_t c0_value, c1_value;
        uint32_t t0_value, t1_value;
        uint32_t block_carry = 0;
        for (int t = threadIdx.x * 2; t < local_size; t += blockDim.x * 2){
            r0_value = local_coefs_a[t].x;
            r1_value = local_coefs_a[t + 1].x;
            c0_value = (t == 0) ? 0 : local_coefs_a[t - 1].y;
            c1_value = local_coefs_a[t].y;
            t0_value = (t <= 1) ? 0 : local_coefs_a[t - 2].z;
            t1_value = (t == 0) ? 0 : local_coefs_a[t - 1].z;
            ushort2 carry;

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
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
                if ((threadIdx.x & 31) >= delta){
                    ushort compound = carry.y + carry_prev.x;
                    carry.x += compound >> 2;
                    if ((compound & 3) == 3){
                        carry.y = carry_prev.y;
                    }else{
                        carry.y = 0;
                    }
                }
            }
            if ((threadIdx.x & 31) == 32 - 1){
                carry_prop[threadIdx.x >> 5] = carry;
            }
            __syncthreads();

            if (threadIdx.x < 32){
                const uint32_t warp_count = blockDim.x >> 5;
                ushort2 bc = (threadIdx.x < warp_count) ? carry_prop[threadIdx.x] : make_ushort2(0, 0);
                for (int delta = 1; delta < warp_count; delta *= 2){
                    ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(bc, delta);
                    if (threadIdx.x >= delta){
                        ushort compound = bc.y + carry_prev.x;
                        bc.x += compound >> 2;
                        if ((compound & 3) == 3){
                            bc.y = carry_prev.y;
                        }else{
                            bc.y = 0;
                        }
                    }
                }
                ushort compound = bc.y + block_carry;
                bc.x += compound >> 2;
                bc.y = 0;
                if (threadIdx.x < warp_count){
                    carry_prop[threadIdx.x] = bc;
                }
            }
            __syncthreads();

            ushort compound = carry.y + ((threadIdx.x >= 32) ? carry_prop[(threadIdx.x >> 5) - 1].x : block_carry);
            carry.x += compound >> 2;
            carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if ((threadIdx.x & 31) == 0){
                if (threadIdx.x == 0){
                    carry.x = block_carry;
                }else{
                    carry.x = carry_prop[(threadIdx.x >> 5) - 1].x;
                }
            }
            r0_value = add_cc(r0_value, (uint32_t)carry.x);
            r1_value = addc_cc(r1_value, 0);

            if (t < L){
                ret[t] = r0_value;
            }
            if (t + 1 < L){
                ret[t + 1] = r1_value;
            }
            block_carry = carry_prop[(blockDim.x >> 5) - 1].x;
            __syncthreads();
        }

        A += ((size_t)gridDim.x) * stride_A;
        B += ((size_t)gridDim.x) * stride_B;
        ret += ((size_t)gridDim.x) * stride_ret;
    }
}

template<int max_local_size>
__global__ void mul_fft_local_spill(uint32_t * A, uint32_t * B, uint32_t * ret, uint3 * workspace, const uint3 * __restrict__ roots_table_lv2, uint3 * inv2n, uint32_t N, uint32_t L_a, uint32_t L_b, uint32_t stride_A, uint32_t stride_B, uint32_t stride_ret, int k){
    __shared__ uint3 local_coefs_a[max_local_size];
    ushort2 * carry_prop = (ushort2 *)local_coefs_a;
    uint32_t local_size = 1u << k;
    size_t L = L_a + L_b;
    A += ((size_t)blockIdx.x) * stride_A;
    B += ((size_t)blockIdx.x) * stride_B;
    ret += ((size_t)blockIdx.x) * stride_ret;
    workspace += ((size_t)blockIdx.x) * local_size;
    for (uint32_t j = blockIdx.x; j < N; j += gridDim.x){
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs_a[t] = make_uint3((t < L_a) ? A[t] : 0, 0, 0);
        }
        __syncthreads();
        for (int lv = k - 1; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size >> 1; t += blockDim.x){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = (idx >> (lv + 1)) * 2;
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                uint3 u_a = local_coefs_a[idx];
                uint3 v_a = local_coefs_a[idx + stride];
                uint3 w_a = mul_mod(v_a, twiddle_factor);
                local_coefs_a[idx] = add_mod(u_a, w_a);
                local_coefs_a[idx + stride] = sub_mod(u_a, w_a);
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            workspace[t] = local_coefs_a[t];
        }
        __syncthreads();
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs_a[t] = make_uint3((t < L_b) ? B[t] : 0, 0, 0);
        }
        __syncthreads();
        for (int lv = k - 1; lv >= 0; lv--){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size >> 1; t += blockDim.x){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = (idx >> (lv + 1)) * 2;
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                uint3 u_a = local_coefs_a[idx];
                uint3 v_a = local_coefs_a[idx + stride];
                uint3 w_a = mul_mod(v_a, twiddle_factor);
                local_coefs_a[idx] = add_mod(u_a, w_a);
                local_coefs_a[idx + stride] = sub_mod(u_a, w_a);
            }
            __syncthreads();
        }
        __syncthreads();
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs_a[t] = normalize_inverse(mul_mod(local_coefs_a[t], workspace[t]), k, inv2n);
        }
        __syncthreads();
        for (int lv = 0; lv < k; lv++){
            uint32_t stride = 1u << lv;
            for (int t = threadIdx.x; t < local_size >> 1; t += blockDim.x){
                int idx = ((t >> lv) << lv) + t;
                uint32_t bitrev_seq_id = (idx >> (lv + 1)) * 2;
                bitrev_seq_id = __brev(-__brev(bitrev_seq_id));
                uint3 twiddle_factor = roots_table_lv2[bitrev_seq_id & 0xffff];
                uint3 u = local_coefs_a[idx];
                uint3 v = local_coefs_a[idx + stride];
                local_coefs_a[idx] = add_mod(u, v);
                local_coefs_a[idx + stride] = mul_mod(sub_mod(u, v), twiddle_factor);
            }
            __syncthreads();
        }
        uint32_t r0_value, r1_value;
        uint32_t c0_value, c1_value;
        uint32_t t0_value, t1_value;
        uint32_t block_carry = 0;
        for (int t = threadIdx.x * 2; t < local_size; t += blockDim.x * 2){
            r0_value = local_coefs_a[t].x;
            r1_value = local_coefs_a[t + 1].x;
            c0_value = (t == 0) ? 0 : local_coefs_a[t - 1].y;
            c1_value = local_coefs_a[t].y;
            t0_value = (t <= 1) ? 0 : local_coefs_a[t - 2].z;
            t1_value = (t == 0) ? 0 : local_coefs_a[t - 1].z;
            __syncthreads();
            ushort2 carry;

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
                ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
                if ((threadIdx.x & 31) >= delta){
                    ushort compound = carry.y + carry_prev.x;
                    carry.x += compound >> 2;
                    if ((compound & 3) == 3){
                        carry.y = carry_prev.y;
                    }else{
                        carry.y = 0;
                    }
                }
            }
            if ((threadIdx.x & 31) == 32 - 1){
                carry_prop[threadIdx.x >> 5] = carry;
            }
            __syncthreads();

            if (threadIdx.x < 32){
                ushort2 bc = carry_prop[threadIdx.x];
                for (int delta = 1; delta < blockDim.x >> 5; delta *= 2){
                    ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(bc, delta);
                    if (threadIdx.x >= delta){
                        ushort compound = bc.y + carry_prev.x;
                        bc.x += compound >> 2;
                        if ((compound & 3) == 3){
                            bc.y = carry_prev.y;
                        }else{
                            bc.y = 0;
                        }
                    }
                }
                ushort compound = bc.y + block_carry;
                bc.x += compound >> 2;
                bc.y = 0;
                carry_prop[threadIdx.x] = bc;
            }
            __syncthreads();

            ushort compound = carry.y + ((threadIdx.x >= 32) ? carry_prop[(threadIdx.x >> 5) - 1].x : block_carry);
            carry.x += compound >> 2;
            carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if ((threadIdx.x & 31) == 0){
                if (threadIdx.x == 0){
                    carry.x = block_carry;
                }else{
                    carry.x = carry_prop[(threadIdx.x >> 5) - 1].x;
                }
            }
            r0_value = add_cc(r0_value, (uint32_t)carry.x);
            r1_value = addc_cc(r1_value, 0);

            if (t < L){
                ret[t] = r0_value;
            }
            if (t + 1 < L){
                ret[t + 1] = r1_value;
            }
            block_carry = carry_prop[(blockDim.x >> 5) - 1].x;
            __syncthreads();
        }

        A += ((size_t)gridDim.x) * stride_A;
        B += ((size_t)gridDim.x) * stride_B;
        ret += ((size_t)gridDim.x) * stride_ret;
    }
}

static uint32_t cyclic_wrap(size_t L, uint32_t La, uint32_t Lb);

static uint32_t get_N_max(uint32_t N, uint32_t L_a, uint32_t L_b){
    size_t L = (size_t)L_a + (size_t)L_b;
    size_t K = 1;
    while (K < L){
        K <<= 1;
    }
    if (K <= 2048){
        return N;
    }
    size_t cache_coefs = 6000000;
    if (const char *value = std::getenv("GPUPI_CACHE_COEFS")) {
        cache_coefs = std::strtoull(value, nullptr, 10);
        cache_coefs = max((size_t)65536, min((size_t)64000000, cache_coefs));
    }
    size_t transform_coefs = K;
    if (!std::getenv("GPUPI_OLD_BATCHING")) {
        uint32_t wrap = cyclic_wrap(L, L_a, L_b);
        if (wrap) {
            size_t tail = 2;
            while (tail < (size_t)wrap * 2) tail *= 2;
            transform_coefs = K / 2 + tail;
        }
    }
    uint32_t cache_limit = max(cache_coefs / transform_coefs, (size_t)1);
    if (K == 4096){
        cache_limit = 8000000 / K;
    }
    
    if (N >= cache_limit){
        N = cache_limit;
    }
    return max(1, N);
}

enum FFTScheduleOp {
    FFT_SCHEDULE_LOCAL = 0,
    FFT_SCHEDULE_RADIX16 = 1,
    FFT_SCHEDULE_RADIX32 = 2,
    FFT_SCHEDULE_RADIX256 = 3,
    FFT_SCHEDULE_RADIX16_INITIAL = 4,
    FFT_SCHEDULE_RADIX32_INITIAL = 5,
};

static int forward_schedule_op(int k, int i, bool cached = false){
    // Match the inverse tiles and keep their first stages fused with products.
    if (cached && (k == 16 || k == 17) && !std::getenv("GPUPI_OLD_CACHED_ALIGN"))
        return i == 0 ? FFT_SCHEDULE_RADIX32 : FFT_SCHEDULE_LOCAL;
    if (cached && k == 25 && !std::getenv("GPUPI_OLD_CACHED_TILE")) {
        if (i == 0) return FFT_SCHEDULE_RADIX32;
        if (i == 5) return FFT_SCHEDULE_RADIX256;
        return FFT_SCHEDULE_LOCAL;
    }
    // Align the final forward tile with the inverse tile so multiplication
    // and the first inverse stages stay in shared memory.
    if ((k == 24 || k == 28) && !std::getenv("GPUPI_OLD_MEDIUM_SCHEDULE")) {
        if (i == 0) return FFT_SCHEDULE_RADIX256;
        if (i == 8 || (k == 28 && i == 13)) return FFT_SCHEDULE_RADIX32;
        return FFT_SCHEDULE_LOCAL;
    }
    if (k >= 29 && !std::getenv("GPUPI_OLD_LARGE_SCHEDULE")) {
        if (i == 0 || i == 8) return FFT_SCHEDULE_RADIX256;
        if (i == 16) return FFT_SCHEDULE_RADIX32;
        return FFT_SCHEDULE_LOCAL;
    }
    if (k - i <= 12){
        return FFT_SCHEDULE_LOCAL;
    }
    if (k < 14 || k > 28){
        if (k >= 23){
            return FFT_SCHEDULE_RADIX256;
        }
        if (k >= 22 || (i != 0 && k - i == 16)){
            return FFT_SCHEDULE_RADIX32;
        }
        return FFT_SCHEDULE_RADIX16;
    }
    if (k == 14){
        return (i == 0) ? FFT_SCHEDULE_RADIX16 : FFT_SCHEDULE_LOCAL;
    }
    if (k == 15){
        return (i == 0) ? FFT_SCHEDULE_RADIX32 : FFT_SCHEDULE_LOCAL;
    }
    if (k <= 20){
        return (i == 0) ? FFT_SCHEDULE_RADIX256 : FFT_SCHEDULE_LOCAL;
    }
    if (k <= 23){
        if (i == 0){
            return FFT_SCHEDULE_RADIX32;
        }
        if (i == 5){
            return FFT_SCHEDULE_RADIX256;
        }
        return FFT_SCHEDULE_LOCAL;
    }
    if (i == 0 || i == 8){
        return FFT_SCHEDULE_RADIX256;
    }
    return FFT_SCHEDULE_LOCAL;
}

static int backward_schedule_local_start(int k){
    if (k >= 29 && !std::getenv("GPUPI_OLD_LARGE_SCHEDULE")) return 21;
    if (k < 14 || k > 28){
        bool use_path_5 = k >= 24;
        int i0 = max(0, k - 12);
        if (use_path_5){
            i0 += (100 - i0) % 5;
        }else{
            i0 += (4 - i0) & 3;
        }
        return i0;
    }
    if (k == 14){
        return 4;
    }
    if (k <= 17){
        return 5;
    }
    if (k == 18){
        return 9;
    }
    if (k == 19){
        return 10;
    }
    if (k == 20){
        return 12;
    }
    if (k <= 25){
        return 13;
    }
    return 18;
}

static int backward_schedule_op(int k, int i){
    if (k >= 29 && !std::getenv("GPUPI_OLD_LARGE_SCHEDULE"))
        return i == 4 ? FFT_SCHEDULE_RADIX32_INITIAL : FFT_SCHEDULE_RADIX256;
    if (k < 14 || k > 28){
        bool use_path_5 = k >= 24;
        if (use_path_5){
            return (i - 4 != 0) ? FFT_SCHEDULE_RADIX32 : FFT_SCHEDULE_RADIX32_INITIAL;
        }
        return (i - 3 != 0) ? FFT_SCHEDULE_RADIX16 : FFT_SCHEDULE_RADIX16_INITIAL;
    }
    switch (k){
        case 14:
            return FFT_SCHEDULE_RADIX16_INITIAL;
        case 15:
        case 16:
        case 17:
            return FFT_SCHEDULE_RADIX32_INITIAL;
        case 18:
            return (i == 8) ? FFT_SCHEDULE_RADIX16 : FFT_SCHEDULE_RADIX32_INITIAL;
        case 19:
            return (i == 9) ? FFT_SCHEDULE_RADIX32 : FFT_SCHEDULE_RADIX32_INITIAL;
        case 20:
            return (i == 11) ? FFT_SCHEDULE_RADIX256 : FFT_SCHEDULE_RADIX16_INITIAL;
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
            return (i == 12) ? FFT_SCHEDULE_RADIX256 : FFT_SCHEDULE_RADIX32_INITIAL;
        default:
            if (i == 17){
                return FFT_SCHEDULE_RADIX32;
            }
            if (i == 12){
                return FFT_SCHEDULE_RADIX256;
            }
            return FFT_SCHEDULE_RADIX32_INITIAL;
    }
}

// A product just beyond a power of two can use a cyclic transform of
// half the padded size. Its wrapped tail is the low part of a much smaller
// product of reversed high limbs. Recover coefficients exactly modulo p,
// then use the same full-width carry scan as for a linear convolution.
static uint32_t cyclic_wrap(size_t L, uint32_t La, uint32_t Lb) {
    if (std::getenv("GPUPI_LINEAR_NTT")) return 0;
    size_t K = 1;
    while (K * 2 < L) K *= 2;
    return K >= 16384 && La <= K && Lb <= K && L > K && L - K <= K / 8 ? (uint32_t)(L - K) : 0;
}

__global__ void fill_reverse_tail(const uint32_t *A, const uint32_t *B,
    uint3 *parts, uint32_t N, uint32_t La, uint32_t Lb,
    uint32_t sa, uint32_t sb, uint32_t wrap, int k) {
    size_t T = (size_t)1 << k;
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         j < (size_t)N * 2 * T; j += (size_t)gridDim.x * blockDim.x) {
        size_t row = j >> k;
        uint32_t i = j & (T - 1);
        uint32_t value = 0;
        if (i + 1 < wrap) {
            if (row < N) value = A[row * sa + La - 1 - i];
            else value = B[(row - N) * sb + Lb - 1 - i];
        }
        parts[j] = make_uint3(value, 0, 0);
    }
}

static void compute_reverse_tail(const uint32_t *A, const uint32_t *B,
    uint3 *parts, uint32_t N, uint32_t La, uint32_t Lb,
    uint32_t sa, uint32_t sb, uint32_t wrap, int k, NTTPrecomputedTables t) {
    size_t T = (size_t)1 << k;
    fill_reverse_tail<<<min((size_t)65535, (N * 2 * T + 255) / 256), 256>>>(
        A, B, parts, N, La, Lb, sa, sb, wrap, k);
    int i = 0;
    while (k - i > 12) {
        fft_level_forward_radix16<<<min((size_t)65535, ((size_t)N << (k-4)) / 64 + 1), 64>>>(
            parts, k, i, t.roots_table_lv1, t.roots_table_lv2, (size_t)N * 2);
        i += 4;
    }
    int local = 1 << (k-i);
    int blocks = min((size_t)65535, (size_t)N * 2 << i);
    if (local <= 1024)
        fft_level_forward_final<1024><<<blocks, min(256,local/2)>>>(parts,k,i,t.roots_table_lv1,t.roots_table_lv2,(size_t)N*2);
    else if (local <= 2048)
        fft_level_forward_final<2048><<<blocks,256>>>(parts,k,i,t.roots_table_lv1,t.roots_table_lv2,(size_t)N*2);
    else
        fft_level_forward_final<4096><<<blocks,512>>>(parts,k,i,t.roots_table_lv1,t.roots_table_lv2,(size_t)N*2);
    pointwise_mul<<<min((size_t)65535,(N*T+255)/256),256>>>(parts,parts+N*T,N*T,t.inv2n_table+k,k);
    blocks = min((size_t)65535, (size_t)N << i);
    if(local<=1024)
        fft_level_backward_final<1024><<<blocks,min(256,local/2)>>>(parts,k,i,t.roots_table_lv1,t.roots_table_lv2,N);
    else if(local<=2048)
        fft_level_backward_final<2048><<<blocks,256>>>(parts,k,i,t.roots_table_lv1,t.roots_table_lv2,N);
    else
        fft_level_backward_final<4096><<<blocks,512>>>(parts,k,i,t.roots_table_lv1,t.roots_table_lv2,N);
    for(i-=1;i>=0;i-=4)
        fft_level_backward_radix16<<<min((size_t)65535, ((size_t)N << (k-4))/64+1),64>>>(
            parts,k,i,t.roots_table_lv1,t.roots_table_lv2,N);
}

__global__ void emit_unwrapped_tail(const uint3 *tail, int tail_k, uint32_t wrap,
    uint32_t *ret, uint32_t *workspace, uint32_t N, int k, size_t L, uint32_t sr) {
    size_t padded = ((size_t)wrap + 31) & ~(size_t)31;
    for (size_t j=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
         j<(size_t)N*padded; j+=(size_t)gridDim.x*blockDim.x) {
        size_t row=j/padded;
        uint32_t i=j%padded;
        uint3 v=i+1<wrap?tail[(row<<tail_k)+wrap-2-i]:make_uint3(0,0,0);
        uint32_t py=__shfl_up_sync(0xffffffffu,v.y,1);
        uint32_t pz=__shfl_up_sync(0xffffffffu,v.z,1);
        uint32_t ppz=__shfl_up_sync(0xffffffffu,v.z,2);
        uint32_t lane=threadIdx.x&31;
        uint32_t r=add_cc(v.x,lane?py:0);
        ushort2 carry;carry.x=addc(0,0);
        r=add_cc(r,lane>1?ppz:0);carry.x+=addc(0,0);
        add_cc(r,2);carry.y=addc(0,0)?r&3:0;
        for(int d=1;d<32;d*=2){
            ushort2 prev=cuda::device::warp_shuffle_up<32,ushort2>(carry,d);
            if(lane>=d)carry=combine_carry(carry,prev);
        }
        ushort2 prev=cuda::device::warp_shuffle_up<32,ushort2>(carry,1);
        if(lane)r+=prev.x;
        size_t idx=((size_t)1<<k)+i;
        if(idx<L)ret[row*sr+idx]=r;
        if(lane==31){
            uint32_t lo=add_cc(v.y,pz);uint32_t hi=addc(v.z,0);
            lo=add_cc(lo,carry.x);hi=addc(hi,0);
            workspace[row*carry_stride(k,L)+(idx>>4)-1]=lo;
            workspace[row*carry_stride(k,L)+(idx>>4)]=hi;
        }
    }
}

__global__ void fill_broadcast(const uint32_t *B, uint32_t L_b, uint3 *parts, uint32_t K) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i += blockDim.x * gridDim.x)
        parts[i] = make_uint3(i < L_b ? B[i] : 0, 0, 0);
}

// A broadcast multiplier is transformed once and kept outside the per-batch scratch.
static void forward_broadcast(uint32_t *B, uint32_t L_b, uint3 *parts,
    int k, NTTPrecomputedTables tables, uint32_t N = 1, uint32_t stride_B = 0) {
    if (k <= 12) fill_broadcast<<<((1u << k) + 255) / 256, 256>>>(B, L_b, parts, 1u << k);
    for (int i = 0; i < k;) {
        const int op = forward_schedule_op(k, i);
        if (op == FFT_SCHEDULE_LOCAL) {
            const int local = 1 << (k - i);
            const int blocks = min((size_t)N << i, (size_t)65536);
            if (local <= 1024)
                fft_level_forward_final<1024><<<blocks, min(local / 2, 256)>>>(
                    parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2, N);
            else if (local <= 2048)
                fft_level_forward_final<2048><<<blocks, 256>>>(
                    parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2, N);
            else
                fft_level_forward_final<4096><<<blocks, 512>>>(
                    parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2, N);
            break;
        }
        if (op == FFT_SCHEDULE_RADIX256) {
            const int blocks = min((size_t)N << (k - 11), (size_t)65536);
            if (i == 0)
                fft_level_forward_radix256_initial<<<blocks, dim3(16, 32)>>>(
                    B, B, parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N, L_b, L_b, stride_B, stride_B, true);
            else
                fft_level_forward_radix256<<<blocks, dim3(16, 32)>>>(
                    parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2, N);
            i += 8;
        } else if (op == FFT_SCHEDULE_RADIX32) {
            const int blocks = min((size_t)N << (k - 9), (size_t)65536);
            if (i == 0)
                fft_level_forward_radix32_initial<<<blocks, dim3(32, 8)>>>(
                    B, B, parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N, L_b, L_b, stride_B, stride_B, true);
            else
                fft_level_forward_radix32<<<blocks, dim3(32, 8)>>>(
                    parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2, N);
            i += 5;
        } else {
            const int blocks = min((size_t)N << (k - 9), (size_t)65536);
            if (i == 0)
                fft_level_forward_radix16_initial<<<blocks, 64>>>(
                    B, B, parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N, L_b, L_b, stride_B, stride_B, true);
            else
                fft_level_forward_radix16<<<blocks, 64>>>(
                    parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2, N);
            i += 4;
        }
    }
}

// With a pretransformed B, its former N*K scratch region only needs the
// compact carry summaries. Keep reversed-tail scratch after those summaries.
static size_t cached_carry_bytes(uint32_t N, int k, size_t L) {
    size_t bytes = (size_t)N * carry_stride(k, L) * sizeof(uint32_t);
    return (bytes + sizeof(uint3) - 1) / sizeof(uint3) * sizeof(uint3);
}

static size_t cached_scratch_size(uint32_t N, uint32_t La, uint32_t Lb) {
    size_t L = (size_t)La + Lb, K = 1, T = 0;
    int k = 0; while (K < L) { K *= 2; ++k; }
    uint32_t wrap = cyclic_wrap(L, La, Lb);
    if (wrap) {
        K /= 2; --k; T = 2;
        while (T < (size_t)wrap * 2) T *= 2;
    }
    return (size_t)N * (K + 2*T) * sizeof(uint3) + cached_carry_bytes(N, k, L);
}

static void batch_mul_ntt_impl(
    uint32_t * A,
    uint32_t * B,
    uint32_t * ret,
    uint32_t * workspace,
    NTTPrecomputedTables tables,
    uint32_t N_total,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_ret,
    uint3 *cached_b = nullptr
){
    size_t L = ((size_t)L_a) + L_b;
    size_t K = 1;
    int k = 0;
    while (K < L){
        K = K * 2;
        k++;
    }

    const uint32_t wrap = cyclic_wrap(L, L_a, L_b);
    int tail_k = 1;
    if (wrap) {
        K >>= 1; --k;
        while (((size_t)1 << tail_k) < (size_t)wrap * 2) ++tail_k;
    }

    if (k >= 6 && k <= 11 && stride_B == 0 && N_total > 1 &&
        !std::getenv("GPUPI_OLD_SMALL_BROADCAST")) {
        uint3 *cached = reinterpret_cast<uint3 *>(workspace);
        forward_broadcast(B, L_b, cached, k, tables);
        if (k <= 9)
            mul_fft_local<512, true><<<min(N_total, 65536), (1 << k) / 2>>>(
                A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k,
                N_total, L_a, L_b, stride_A, stride_B, stride_ret, k, cached);
        else if (k == 10)
            mul_fft_local<1024, true><<<min(N_total, 65536), 256>>>(
                A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k,
                N_total, L_a, L_b, stride_A, stride_B, stride_ret, k, cached);
        else
            mul_fft_local<2048, true><<<min(N_total, 65536), 512>>>(
                A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k,
                N_total, L_a, L_b, stride_A, stride_B, stride_ret, k, cached);
        return;
    }

    if (k <= 9){
        mul_fft_local<512><<<min(N_total, 65536), max(1, (1 << k) >> 1)>>>(
            A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, stride_A, stride_B, stride_ret, k
        );
        return;
    }
    if (k == 10){
        mul_fft_local<1024><<<min(N_total, 65536), 256>>>(
            A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, stride_A, stride_B, stride_ret, k
        );
        return;
    }
    if (k == 11){
        mul_fft_local<2048><<<min(N_total, 65536), 512>>>(
            A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, stride_A, stride_B, stride_ret, k
        );
        return;
    }

    uint32_t N_batch = get_N_max(N_total, L_a, L_b);

    if (k == 12){
        mul_fft_local_spill<4096><<<N_batch, 512>>>(
            A, B, ret, reinterpret_cast<uint3 *>(workspace), tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, stride_A, stride_B, stride_ret, k
        );
        return;
    }

    // Squaring needs only one forward transform. Keep parts_b as scratch for
    // the inverse transform's carry reconstruction.
    const bool square = A == B && L_a == L_b && stride_A == stride_B;
    const bool broadcast = !cached_b && !square && stride_B == 0 && N_total > 1 &&
        !std::getenv("GPUPI_OLD_BROADCAST");
    const bool cached = broadcast || cached_b != nullptr;
    uint3 *broadcast_parts = nullptr;
    if (broadcast) {
        broadcast_parts = reinterpret_cast<uint3 *>(workspace) +
            batch_mul_ntt_workspace_size(N_total, L_a, L_b) / sizeof(uint3) - K;
        forward_broadcast(B, L_b, broadcast_parts, k, tables);
    }
    for (uint32_t ni = 0; ni < N_total; ni += N_batch){
        uint32_t N = min(N_total - ni, N_batch);
        const uint32_t forward_N = (square || cached) ? N : N * 2;
        uint3 * parts_a = reinterpret_cast<uint3 *>(workspace);
        uint3 * parts_b = reinterpret_cast<uint3 *>(workspace + (((size_t)N) << k) * 3);
        uint3 *tail = wrap ? parts_b + (cached_b
            ? cached_carry_bytes(N, k, L) / sizeof(uint3) : (size_t)N * K) : nullptr;
        if (wrap) compute_reverse_tail(A, B, tail, N, L_a, L_b, stride_A, stride_B, wrap, tail_k, tables);
        const int i0 = backward_schedule_local_start(k);
        bool fused_product = false;
        bool fused_backward = false;

        for (int i = 0; i < k; ){
            int schedule_op = forward_schedule_op(k, i, cached);
            if (schedule_op == FFT_SCHEDULE_LOCAL){
                int local_size = 1 << (k - i);
                if (!square && (local_size <= 2048 || (cached && local_size <= 4096))){
                    fused_product = true;
                    fused_backward = i0 >= i;
                    const int blocks = min((size_t)N << i, (size_t)65536);
                    if (local_size <= 1024){
                        if (broadcast) {
                            fft_local_product<1024, true, true><<<blocks, min(local_size >> 1, 256)>>>(
                                parts_a, broadcast_parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0
                            );
                        } else if (cached_b) {
                            fft_local_product<1024, true, false><<<blocks, min(local_size >> 1, 256)>>>(
                                parts_a, cached_b, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0
                            );
                        } else {
                            fft_local_product<1024><<<blocks, min(local_size >> 1, 256)>>>(
                                parts_a, parts_b, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0
                            );
                        }
                    }else if (local_size <= 2048){
                        if (broadcast) {
                            fft_local_product<2048, true, true><<<blocks, 512>>>(
                                parts_a, broadcast_parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0
                            );
                        } else if (cached_b) {
                            fft_local_product<2048, true, false><<<blocks, 512>>>(
                                parts_a, cached_b, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0
                            );
                        } else {
                            fft_local_product<2048><<<blocks, 512>>>(
                                parts_a, parts_b, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0
                            );
                        }
                    }else{
                        if (broadcast)
                            fft_local_product<4096, true, true><<<blocks, 512>>>(
                                parts_a, broadcast_parts, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0);
                        else
                            fft_local_product<4096, true, false><<<blocks, 512>>>(
                                parts_a, cached_b, k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                                N, tables.inv2n_table + k, fused_backward ? k - i0 : 0);
                    }
                    break;
                }
                int num_blocks = min((((size_t)forward_N) << i), (size_t)65536);
                if (k - i <= 10){
                    fft_level_forward_final<1024><<<num_blocks, min(local_size >> 1, 256)>>>(
                        parts_a,
                        k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                        forward_N
                    );
                }else if (k - i == 11){
                    fft_level_forward_final<2048><<<num_blocks, min(local_size >> 1, 256)>>>(
                        parts_a,
                        k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                        forward_N
                    );
                }else{
                    fft_level_forward_final<4096><<<num_blocks, min(local_size >> 1, 512)>>>(
                        parts_a,
                        k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                        forward_N
                    );
                }
                break;
            }else{
                if (schedule_op == FFT_SCHEDULE_RADIX256){
                    int num_blocks = min(((size_t)N) << (k - 11), (size_t)65536);
                    if (i != 0){
                        fft_level_forward_radix256<<<num_blocks, dim3(16, 32, 1)>>>(
                            parts_a, // parts_b must be adjacent to it
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            forward_N
                        );
                    }else{
                        fft_level_forward_radix256_initial<<<num_blocks, dim3(16, 32, 1)>>>(
                            A, B,
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            forward_N, L_a, L_b, stride_A, stride_B, square || cached
                        );
                    }
                    i += 8;
                }else if (schedule_op == FFT_SCHEDULE_RADIX32){
                    int num_blocks = min(((size_t)N) << (k - 9), (size_t)65536);
                    if (i != 0){
                        fft_level_forward_radix32<<<num_blocks, dim3(32, 8, 1)>>>(
                            parts_a, // parts_b must be adjacent to it
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            forward_N
                        );
                        
                    }else{
                        fft_level_forward_radix32_initial<<<num_blocks, dim3(32, 8, 1)>>>(
                            A, B,
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            forward_N, L_a, L_b, stride_A, stride_B, square || cached
                        );
                    }
                    i += 5;
                }else{
                    int num_blocks = min(((size_t)N) << (k - 9), (size_t)65536);
                    if (i != 0){
                        fft_level_forward_radix16<<<num_blocks, 64>>>(
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            forward_N
                        );
                    }else{
                        fft_level_forward_radix16_initial<<<num_blocks, 64>>>(
                            A, B,
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            forward_N, L_a, L_b, stride_A, stride_B, square || cached
                        );
                    }
                    i += 4;
                }
            }
        }

        // fuse divide by K into pointwise mul
        if (!fused_product){
            const int threads_per_block = 512;
            int num_blocks = min(((((size_t)N) << k) + threads_per_block - 1) / threads_per_block, (size_t)65536);
            if (cached_b)
                pointwise_mul<<<num_blocks, threads_per_block>>>(parts_a, cached_b, (size_t)N * K, tables.inv2n_table + k,k);
            else if (broadcast)
                pointwise_mul_broadcast<<<num_blocks, threads_per_block>>>(parts_a, broadcast_parts, (size_t)N * K, K, tables.inv2n_table + k,k);
            else
                pointwise_mul<<<num_blocks, threads_per_block>>>(parts_a, square ? parts_a : parts_b, ((size_t)N) << k, tables.inv2n_table + k,k);
        }
        
        if (!fused_backward){
            int num_blocks = min((((size_t)N) << (i0 + 1)), (size_t)65536);
            int local_size = 1 << (k - i0);
            if (local_size <= 1024){
                fft_level_backward_final<1024><<<num_blocks, min(local_size >> 1, 256)>>>(
                    parts_a,
                    k, i0, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
            }else if (local_size == 2048){
                fft_level_backward_final<2048><<<num_blocks, min(local_size >> 1, 256)>>>(
                    parts_a,
                    k, i0, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
            }else{
                fft_level_backward_final<4096><<<num_blocks, min(local_size >> 1, 512)>>>(
                    parts_a,
                    k, i0, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
            }
        }

        for (int i = i0 - 1; i >= 0; ){
            int schedule_op = backward_schedule_op(k, i);
            if (schedule_op == FFT_SCHEDULE_RADIX256){
                const int threads_per_block = 16;
                int num_blocks = min(((((size_t)N) << (k - 8)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_backward_radix256<<<num_blocks, dim3(16, 32, 1)>>>(
                    parts_a,
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
                i -= 8;
            }else if (schedule_op == FFT_SCHEDULE_RADIX32){
                const int threads_per_block = 32;
                int num_blocks = min(((((size_t)N) << (k - 5)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_backward_radix32<<<num_blocks, dim3(32, 8, 1)>>>(
                    parts_a,
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
                i -= 5;
            }else if (schedule_op == FFT_SCHEDULE_RADIX32_INITIAL){
                const int threads_per_block = 32;
                int num_blocks = min(((((size_t)N) << (k - 5)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_backward_radix32_initial<<<num_blocks, dim3(32, 8, 1)>>>(
                    parts_a, k, ret, reinterpret_cast<uint32_t*>(parts_b),
                    tables.roots_table_lv2,
                    N, L, stride_ret, tail, tail_k, wrap
                );
                break;
            }else{
                const int threads_per_block = 64;
                int num_blocks = min(((((size_t)N) << (k - 4)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                if (schedule_op == FFT_SCHEDULE_RADIX16 ||
                    (schedule_op == FFT_SCHEDULE_RADIX16_INITIAL && K == 8192)){
                    fft_level_backward_radix16<<<num_blocks, threads_per_block>>>(
                        parts_a,
                        k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                        N
                    );
                    i -= 4;
                }else{
                    fft_level_backward_radix16_initial<<<num_blocks, threads_per_block>>>(
                        parts_a, k, ret, reinterpret_cast<uint32_t*>(parts_b),
                        tables.roots_table_lv2,
                        N, L, stride_ret, tail, tail_k, wrap
                    );
                    break;
                }
            }
        }

        if (wrap) {
            size_t count=(size_t)N*((wrap+31u)&~31u);
            emit_unwrapped_tail<<<min((size_t)65535,(count+255)/256),256>>>(
                tail,tail_k,wrap,ret,reinterpret_cast<uint32_t*>(parts_b),N,k,L,stride_ret);
        }

        if (K == 8192){
            int threads_per_block = min(L>>1, (size_t)1024);
            int num_blocks = min(N, 65536);
            add3_single_block<<<num_blocks, dim3(32, (threads_per_block + 31) / 32, 1)>>>(parts_a, ret, N, k, L, stride_ret);
        }else{
            if (N >= 85){
                int threads_per_block = min(L>>1, (size_t)1024);
                int num_blocks = min(N, 65536);
                add2_single_block<<<num_blocks, dim3(32, (threads_per_block + 31) / 32, 1)>>>(
                    ret, reinterpret_cast<uint32_t *>(parts_b),
                    N, k, L, stride_ret
                );
            }else{
                const int block_size = 2048;
                int num_blocks_x = min((size_t)256, max((size_t)1, (L + block_size - 1) / block_size));
                int num_blocks_y = min(N, 65536 / num_blocks_x);
                add2_reduce_blocks<<<dim3(num_blocks_x, num_blocks_y, 1), dim3(32, 32, 1)>>>(
                    ret, reinterpret_cast<uint32_t *>(parts_b), reinterpret_cast<ushort2 *>(parts_a), N, k, L, stride_ret
                );
                add2_combine_blocks<<<min(N, 65536), dim3(32, 32, 1)>>>(
                    reinterpret_cast<ushort2 *>(parts_a), N, L
                );
                add2_apply_blocks<<<dim3(num_blocks_x, num_blocks_y, 1), dim3(32, 32, 1)>>>(
                    ret, reinterpret_cast<ushort2 *>(parts_a), N, k, L, stride_ret
                );
            }
        }

        if (cached_b) cached_b += (size_t)N * K;
        A += ((size_t)N) * stride_A;
        B += ((size_t)N) * stride_B;
        ret += ((size_t)N) * stride_ret;
    }
}

void batch_mul_ntt(uint32_t *A, uint32_t *B, uint32_t *ret, uint32_t *workspace,
    NTTPrecomputedTables tables, uint32_t N, uint32_t La, uint32_t Lb,
    uint32_t sa, uint32_t sb, uint32_t sr) {
    batch_mul_ntt_impl(A, B, ret, workspace, tables, N, La, Lb, sa, sb, sr);
}

size_t batch_mul_ntt_workspace_size(uint32_t N, uint32_t L_a, uint32_t L_b){
    size_t L = (size_t)L_a + (size_t)L_b;
    size_t K = 1;
    while (K < L){
        K <<= 1;
    }
    if (K <= 2048){
        return N > 1 && K >= 64 ? K * sizeof(uint3) : 0;
    }
    // Per-batch transforms and tail scratch, plus one persistent broadcast
    // transform for multiple rows (the size API does not receive strides).
    uint32_t N_batch = get_N_max(N, L_a, L_b);
    if (K == 4096){
        return N_batch * K * sizeof(uint3);
    }
    uint32_t wrap = cyclic_wrap(L, L_a, L_b);
    if (wrap) {
        size_t T=2;
        while(T<(size_t)wrap*2)T*=2;
        return (N_batch * (K + T*2) + (N > 1 ? K/2 : 0)) * sizeof(uint3);
    }
    return (N_batch * K * 2 + (N > 1 ? K : 0)) * sizeof(uint3);
}

// Reuse the transform of each B row for two products with the same effective
// NTT length. Compact carry scratch keeps even the 2^31 case within 96 GiB.
static size_t shared_transform_size(uint32_t La, uint32_t Lc, uint32_t Lb) {
    auto effective = [](uint32_t a, uint32_t b) {
        size_t L = (size_t)a + b, K = 1;
        while (K < L) K *= 2;
        return cyclic_wrap(L, a, b) ? K / 2 : K;
    };
    size_t K = effective(La, Lb);
    if (K < 8192 || K > ((size_t)1 << 31) || K != effective(Lc, Lb) ||
        std::getenv("GPUPI_OLD_SHARED_PAIR")) return 0;
    return K;
}

static uint32_t shared_batch_size(uint32_t N, uint32_t La, uint32_t Lc, uint32_t Lb, size_t K) {
    size_t budget = 6000000;
    if (const char *value = std::getenv("GPUPI_PAIR_CACHE_COEFS"))
        budget = max((size_t)65536, min((size_t)64000000, (size_t)std::strtoull(value, nullptr, 10)));
    uint32_t nb = min(get_N_max(N, La, Lb), get_N_max(N, Lc, Lb));
    return min(nb, (uint32_t)max((size_t)1, budget / K));
}

size_t batch_mul_ntt_shared_b_workspace_size(uint32_t N, uint32_t La, uint32_t Lc, uint32_t Lb) {
    size_t K = shared_transform_size(La, Lc, Lb);
    if (!K) return 0;
    uint32_t nb = shared_batch_size(N, La, Lc, Lb, K);
    return max(cached_scratch_size(nb, La, Lb),
               cached_scratch_size(nb, Lc, Lb)) + (size_t)nb * K * sizeof(uint3);
}

void batch_mul_ntt_shared_b(NTTProduct a, NTTProduct c, uint32_t *B, uint32_t Lb,
    uint32_t sb, uint32_t N, uint32_t *workspace, NTTPrecomputedTables tables) {
    size_t K = shared_transform_size(a.length, c.length, Lb);
    uint32_t nb = shared_batch_size(N, a.length, c.length, Lb, K);
    size_t scratch_bytes = max(cached_scratch_size(nb, a.length, Lb),
                              cached_scratch_size(nb, c.length, Lb));
    uint3 *cache = reinterpret_cast<uint3 *>(reinterpret_cast<char *>(workspace) + scratch_bytes);
    int k = 0; while (((size_t)1 << k) < K) ++k;
    for (uint32_t offset = 0; offset < N; offset += nb) {
        uint32_t rows = min(nb, N - offset);
        uint32_t *b = B + (size_t)offset * sb;
        forward_broadcast(b, Lb, cache, k, tables, rows, sb);
        for (NTTProduct x : {a, c})
            batch_mul_ntt_impl(x.input + (size_t)offset * x.stride_input, b,
                x.output + (size_t)offset * x.stride_output, workspace, tables,
                rows, x.length, Lb, x.stride_input, sb, x.stride_output, cache);
    }
}
