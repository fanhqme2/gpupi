#include "batch_mul_ntt.h"
#include "batch_mul_addsub_asm.h"
#include "batch_mul_addsub_warp.h"
#include <cuda/warp>

const uint32_t arr_root0 = 0x46038aeb;
const uint32_t arr_root1 = 0x55c21a87;
const uint32_t arr_root2 = 0xa257c693;

const uint32_t arr_root65536_0 = 0xea78dff7;
const uint32_t arr_root65536_1 = 0xbcc6f614;
const uint32_t arr_root65536_2 = 0x2c17ad1;

const uint32_t arr_inv2_0 = 0x80000001;
const uint32_t arr_inv2_1 = 0xffffffff;
const uint32_t arr_inv2_2 = 0x7fffffff;

__device__ __forceinline__ uint3 add_mod(uint3 a, uint3 b){
    uint32_t c0, c1, c2;
    c0 = add_cc(a.x, b.x);
    c1 = addc_cc(a.y, b.y);
    c2 = addc_cc(a.z, b.z);
    uint32_t carry = addc(0u, 0u);

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
    return make_uint3(out0, out1, out2);
}

__device__ __forceinline__ uint3 sub_mod(uint3 a, uint3 b){
    uint32_t c0 = sub_cc(a.x, b.x);
    uint32_t c1 = subc_cc(a.y, b.y);
    uint32_t c2 = subc_cc(a.z, b.z);
    uint32_t borrow = subc(0u, 0u) & 1u;

    uint32_t bs0 = sub_cc(0u, borrow);
    uint32_t bs1 = subc(borrow, 0u);

    uint32_t out0 = sub_cc(c0, bs0);
    uint32_t out1 = subc_cc(c1, bs1);
    uint32_t out2 = subc(c2, 0u);
    return make_uint3(out0, out1, out2);
}

__device__ __forceinline__ uint3 mul_mod(uint3 a, uint3 b){
    uint32_t a0 = a.x, a1 = a.y, a2 = a.z;
    uint32_t b0 = b.x, b1 = b.y, b2 = b.z;

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
    return make_uint3(out0, out1, out2);
}

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
    fill_in_power_table<<<1, 32>>>(tables->inv2n_table, 33, make_uint3(arr_inv2_0, arr_inv2_1, arr_inv2_2));
}

__global__ void fft_level_forward(uint3 * parts, int k, int i, uint3 * roots_table_lv1, uint3 * roots_table_lv2, size_t N){
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
    uint3 *roots_table_lv1, uint3 *roots_table_lv2,
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
    uint3 *roots_table_lv1, uint3 *roots_table_lv2,
    size_t N, uint32_t L_a, uint32_t L_b
){
    const uint32_t step = 1u << (k - 4 - i);
    size_t N_half = N >> 1;

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
                size_t offset_input = group * L_a;
                uint32_t idx = step_id + (size_t)step * t;
                x[t] = make_uint3(idx < L_a ? A[offset_input + idx] : 0, 0, 0);
            }else{
                size_t offset_input = (group - N_half) * L_b;
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
    uint3 *roots_table_lv1, uint3 *roots_table_lv2,
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
    uint3 *roots_table_lv1, uint3 *roots_table_lv2,
    size_t N, uint32_t L_a, uint32_t L_b
){
    __shared__ uint3 local_coefs[32][32];
    const uint32_t step = 1u << (k - 5 - i);
    const uint32_t seq_len = 1u << i;
    size_t N_half = N >> 1;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 5)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 5 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));
        for (int t = threadIdx.y; t < 32; t += blockDim.y){
            if (group < N_half){
                size_t offset_input = group * L_a;
                uint32_t idx = step_id + (size_t)step * t;
                local_coefs[t][threadIdx.x] = make_uint3(idx < L_a ? A[offset_input + idx] : 0, 0, 0);
            }else{
                size_t offset_input = (group - N_half) * L_b;
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

template<int max_local_size>
__global__ void fft_level_forward_final(
    uint3 * parts, int k, int i,
    uint3 * roots_table_lv1, uint3 * roots_table_lv2,
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

__global__ void pointwise_mul(uint3 * parts_a, uint3 * parts_b, size_t total, uint3 * scalar_ptr){
    uint3 scalar = scalar_ptr[0];
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < total; j += blockDim.x * gridDim.x){
        parts_a[j] = mul_mod(mul_mod(parts_a[j], parts_b[j]), scalar);
    }
}

__global__ void fft_level_backward(uint3 * parts, int k, int i, uint3 * roots_table_lv1, uint3 * roots_table_lv2, size_t N){
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

__global__ void fft_level_backward_radix16(uint3 * parts, int k, int i, uint3 * roots_table_lv1, uint3 * roots_table_lv2, size_t N){
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
    uint3 *roots_table_lv1, uint3 *roots_table_lv2,
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

__global__ void fft_level_backward_radix32_initial(
    uint3 *parts, int k,
    uint32_t * ret, uint32_t * workspace,
    uint3 *roots_table_lv2,
    size_t N, size_t L
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
                ret[group * L + idx] = r0_value;
            }
            if (threadIdx.x == 31){
                r0_value = local_coefs[t][31].y;
                uint32_t r1_value = local_coefs[t][31].z;
                r0_value = add_cc(r0_value, local_coefs[t][30].z);
                r1_value = addc_cc(r1_value, 0);
                r0_value = add_cc(r0_value, carry.x);
                r1_value = addc_cc(r1_value, 0);
                workspace[(((size_t)group) << (k - 4)) + (idx >> 4) - 1] = r0_value;
                workspace[(((size_t)group) << (k - 4)) + (idx >> 4)] = r1_value;
            }
        }
        __syncthreads();
    }
}

__global__ void fft_level_backward_radix16_initial(
    uint3 *parts, int k,
    uint32_t * ret, uint32_t * workspace,
    uint3 *roots_table_lv2,
    size_t N, size_t L
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
                ret[group * L + idx] = r0_value;
            }
            if ((threadIdx.x & 31) == 31){
                r0_value = dt.y;
                uint32_t r1_value = dt.z;
                r0_value = add_cc(r0_value, d_prev_z);
                r1_value = addc_cc(r1_value, 0);
                r0_value = add_cc(r0_value, carry.x);
                r1_value = addc_cc(r1_value, 0);
                workspace[(((size_t)group) << (k - 4)) + (idx >> 4) - 1] = r0_value;
                workspace[(((size_t)group) << (k - 4)) + (idx >> 4)] = r1_value;
            }
        }
        __syncthreads();
    }
}

template<int max_local_size>
__global__ void fft_level_backward_final(
    uint3 * parts, int k, int i,
    uint3 * roots_table_lv1, uint3 * roots_table_lv2,
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

__global__ void add3_single_block(uint3 * parts, uint32_t * ret, size_t N, int k, size_t L){
    __shared__ ushort2 carryInfo[32];
    parts += ((size_t)blockIdx.x) << k;
    ret += ((size_t)blockIdx.x) * L;
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
        ret += ((size_t)L) * gridDim.x;
    }
}

__global__ void add3_reduce_blocks(uint3 * parts, uint32_t * ret, ushort2 * ret_carry, size_t N, int k, size_t L){
    __shared__ ushort2 carryInfo[32];
    parts += ((size_t)blockIdx.y) << k;
    ret += ((size_t)blockIdx.y) * L;
    const int block_size = 2048;
    uint32_t blocks_per_num = (L + block_size - 1) / block_size;
    ret_carry += ((size_t)blockIdx.y) * blocks_per_num;
    for (int idx = blockIdx.y; idx < N; idx += gridDim.y){
        for (size_t i0 = blockIdx.x * block_size; i0 < L; i0 += block_size * gridDim.x){
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
        parts += ((size_t)gridDim.y) << k;
        ret += ((size_t)L) * gridDim.y;
        ret_carry += blocks_per_num * gridDim.y;
    }
}
__global__ void add3_combine_blocks(ushort2 * ret_carry, uint32_t N, uint32_t L){
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

__global__ void add3_apply_blocks(uint32_t * ret, ushort2 * ret_carry, size_t N, int k, size_t L){
    __shared__ uint32_t carryInfo[32];
    const int block_size = 2048;
    uint32_t blocks_per_num = (L + block_size - 1) / block_size;
    ret_carry += ((size_t)blockIdx.y) * blocks_per_num;
    ret += ((size_t)blockIdx.y) * L;
    for (int idx = blockIdx.y; idx < N; idx += gridDim.y){
        for (size_t i0 = blockIdx.x * block_size; i0 < L; i0 += block_size * gridDim.x){       
            uint32_t block_carry = ret_carry[i0 / block_size].x;     
            if (block_carry){
                uint32_t i = i0 + (threadIdx.y * 32 + threadIdx.x) * 2;
                uint32_t r0_value = (i < L) ? ret[i] : 0;
                uint32_t r1_value = (i + 1 < L) ? ret[i + 1] : 0;
                uint32_t c0_value = (i == i0) ? block_carry : 0;
                uint32_t c1_value = 0;
                uint32_t r0_value_old = r0_value;
                batch_mul_add_64_all_warp<32>(r0_value, r1_value, c0_value, c1_value, carryInfo);

                if (r0_value != r0_value_old){
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
        ret += ((size_t)L) * gridDim.y;
    }
}

__global__ void add2_reduce_blocks(uint32_t * ret, uint32_t * workspace, ushort2 * ret_carry, size_t N, int k, size_t L){
    __shared__ ushort2 carryInfo[32];
    ret += ((size_t)blockIdx.y) * L;
    const int block_size = 2048;
    uint32_t blocks_per_num = (L + block_size - 1) / block_size;
    ret_carry += ((size_t)blockIdx.y) * blocks_per_num;
    workspace += ((size_t)blockIdx.y) << (k - 4);
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
                    carry_prev = combine_carry(carry, carry_prev);
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
        ret += ((size_t)L) * gridDim.y;
        ret_carry += blocks_per_num * gridDim.y;
        workspace += gridDim.y << (k - 4);
    }
}

__global__ void add2_single_block(uint32_t * ret, uint32_t * workspace, size_t N, int k, size_t L){
    __shared__ ushort2 carryInfo[32];
    ret += ((size_t)blockIdx.x) * L;
    workspace += ((size_t)blockIdx.x) << (k - 4);
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
        ret += ((size_t)L) * gridDim.x;
        workspace += gridDim.x << (k - 4);
    }
}

template<int max_local_size>
__global__ void mul_fft_local(uint32_t * A, uint32_t * B, uint32_t * ret, uint3 * roots_table_lv2, uint3 * inv2n, uint32_t N, uint32_t L_a, uint32_t L_b, int k){
    __shared__ uint3 local_coefs_a[max_local_size], local_coefs_b[max_local_size];
    //__shared__ ushort2 carry_prop[32];
    ushort2 * carry_prop = (ushort2 *)local_coefs_b;
    uint32_t local_size = 1u << k;
    size_t L = L_a + L_b;
    uint3 inv2n_val = inv2n[0];
    A += ((size_t)blockIdx.x) * L_a;
    B += ((size_t)blockIdx.x) * L_b;
    ret += ((size_t)blockIdx.x) * L;
    for (uint32_t j = blockIdx.x; j < N; j += gridDim.x){
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs_a[t] = make_uint3((t < L_a) ? A[t] : 0, 0, 0);
            local_coefs_b[t] = make_uint3((t < L_b) ? B[t] : 0, 0, 0);
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

                uint3 u_b = local_coefs_b[idx];
                uint3 v_b = local_coefs_b[idx + stride];
                uint3 w_b = mul_mod(v_b, twiddle_factor);
                local_coefs_b[idx] = add_mod(u_b, w_b);
                local_coefs_b[idx + stride] = sub_mod(u_b, w_b);
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < local_size; t += blockDim.x){
            local_coefs_a[t] = mul_mod(mul_mod(local_coefs_a[t], local_coefs_b[t]), inv2n_val);
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

        A += ((size_t)gridDim.x) * L_a;
        B += ((size_t)gridDim.x) * L_b;
        ret += ((size_t)gridDim.x) * L;
    }
}

template<int max_local_size>
__global__ void mul_fft_local_spill(uint32_t * A, uint32_t * B, uint32_t * ret, uint3 * workspace, uint3 * roots_table_lv2, uint3 * inv2n, uint32_t N, uint32_t L_a, uint32_t L_b, int k){
    __shared__ uint3 local_coefs_a[max_local_size];
    ushort2 * carry_prop = (ushort2 *)local_coefs_a;
    uint32_t local_size = 1u << k;
    size_t L = L_a + L_b;
    uint3 inv2n_val = inv2n[0];
    A += ((size_t)blockIdx.x) * L_a;
    B += ((size_t)blockIdx.x) * L_b;
    ret += ((size_t)blockIdx.x) * L;
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
            local_coefs_a[t] = mul_mod(mul_mod(local_coefs_a[t], workspace[t]), inv2n_val);
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

        A += ((size_t)gridDim.x) * L_a;
        B += ((size_t)gridDim.x) * L_b;
        ret += ((size_t)gridDim.x) * L;
    }
}

static uint32_t get_N_max(uint32_t N, uint32_t L_a, uint32_t L_b){
    size_t L = (size_t)L_a + (size_t)L_b;
    size_t K = 1;
    while (K < L){
        K <<= 1;
    }
    if (K <= 2048){
        return N;
    }
    uint32_t cache_limit = max(4000000 / K, (size_t) 1);
    if (K == 4096){
        cache_limit = 8000000 / K;
    }
    
    if (N >= cache_limit){
        N = cache_limit;
    }
    return max(1, N);
}

void batch_mul_ntt(
    uint32_t * A,
    uint32_t * B,
    uint32_t * ret,
    uint32_t * workspace,
    NTTPrecomputedTables tables,
    uint32_t N_total,
    uint32_t L_a,
    uint32_t L_b
){
    size_t L = ((size_t)L_a) + L_b;
    size_t K = 1;
    int k = 0;
    while (K < L){
        K = K * 2;
        k++;
    }

    if (k <= 9){
        mul_fft_local<512><<<min(N_total, 65536), max(1, (1 << k) >> 1)>>>(
            A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, k
        );
        return;
    }
    if (k == 10){
        mul_fft_local<1024><<<min(N_total, 65536), 256>>>(
            A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, k
        );
        return;
    }
    if (k == 11){
        mul_fft_local<2048><<<min(N_total, 65536), 512>>>(
            A, B, ret, tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, k
        );
        return;
    }

    uint32_t N_batch = get_N_max(N_total, L_a, L_b);

    if (k == 12){
        mul_fft_local_spill<4096><<<N_batch, 512>>>(
            A, B, ret, reinterpret_cast<uint3 *>(workspace), tables.roots_table_lv2, tables.inv2n_table + k, N_total, L_a, L_b, k
        );
        return;
    }

    for (uint32_t ni = 0; ni < N_total; ni += N_batch){
        uint32_t N = min(N_total - ni, N_batch);
        uint3 * parts_a = reinterpret_cast<uint3 *>(workspace);
        uint3 * parts_b = reinterpret_cast<uint3 *>(workspace + (((size_t)N) << k) * 3);

        for (int i = 0; i < k; i ++){
            if (k - i <= 12){
                int local_size = 1 << (k - i);
                int num_blocks = min((((size_t)N) << (i + 1)), (size_t)65536);
                if (k - i <= 10){
                    fft_level_forward_final<1024><<<num_blocks, min(local_size >> 1, 256)>>>(
                        parts_a,
                        k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                        N * 2 // we use 2 * N to do parts_a and parts_b
                    );
                }else if (k - i == 11){
                    fft_level_forward_final<2048><<<num_blocks, min(local_size >> 1, 256)>>>(
                        parts_a,
                        k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                        N * 2 // we use 2 * N to do parts_a and parts_b
                    );
                }else{
                    fft_level_forward_final<4096><<<num_blocks, min(local_size >> 1, 512)>>>(
                        parts_a,
                        k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                        N * 2 // we use 2 * N to do parts_a and parts_b
                    );
                }
                i = k - 1;
            }else{
                int num_blocks = min(((size_t)N) << (k - 9), (size_t)65536);
                if (k >= 22 ||  (i != 0 && k - i == 16)){
                    if (i != 0){
                        fft_level_forward_radix32<<<num_blocks, dim3(32, 8, 1)>>>(
                            parts_a, // parts_b must be adjacent to it
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            N * 2 // we use 2 * N to do parts_a and parts_b simultaneously
                        );
                    }else{
                        fft_level_forward_radix32_initial<<<num_blocks, dim3(32, 8, 1)>>>(
                            A, B,
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            N * 2, L_a, L_b
                        );
                    }
                    i += 4;
                }else{
                    if (i != 0){
                        fft_level_forward_radix16<<<num_blocks, 64>>>(
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            N * 2
                        );
                    }else{
                        fft_level_forward_radix16_initial<<<num_blocks, 64>>>(
                            A, B,
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            N * 2, L_a, L_b
                        );
                    }
                    i += 3;
                }
            }
        }

        // fuse divide by K into pointwise mul
        if (true){
            const int threads_per_block = 512;
            int num_blocks = min(((((size_t)N) << k) + threads_per_block - 1) / threads_per_block, (size_t)65536);
            pointwise_mul<<<num_blocks, threads_per_block>>>(parts_a, parts_b, ((size_t)N) << k, tables.inv2n_table + k);
        }
        
        bool use_path_5 = k >= 24;

        for (int i = k - 1; i >= 0; i --){
            if (i == k - 1){
                int i0 = max(0, i - 11);
                if (use_path_5){
                    i0 += (100 - i0) % 5;
                }else{
                    i0 += (4 - i0) & 3;
                }
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
                i = i0;
            }else{
                if (use_path_5){
                    const int threads_per_block = 32;
                    int num_blocks = min(((((size_t)N) << (k - 5)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                    if (i - 4 != 0){
                        fft_level_backward_radix32<<<num_blocks, dim3(32, 8, 1)>>>(
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            N
                        );
                    }else{
                        fft_level_backward_radix32_initial<<<num_blocks, dim3(32, 8, 1)>>>(
                            parts_a, k, ret, reinterpret_cast<uint32_t*>(parts_b),
                            tables.roots_table_lv2,
                            N, L
                        );
                    }
                    i -= 4;
                }else{
                    const int threads_per_block = 64;
                    int num_blocks = min(((((size_t)N) << (k - 4)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                    if (i - 3 != 0 || L == 8192){
                        fft_level_backward_radix16<<<num_blocks, threads_per_block>>>(
                            parts_a,
                            k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                            N
                        );
                    }else{
                        fft_level_backward_radix16_initial<<<num_blocks, threads_per_block>>>(
                            parts_a, k, ret, reinterpret_cast<uint32_t*>(parts_b),
                            tables.roots_table_lv2,
                            N, L
                        );
                    }
                    i -= 3;
                }
            }
        }

        if (!use_path_5 && L == 8192){
            if (L <= 8192 || N >= 85){
                int threads_per_block = min(L>>1, (size_t)1024);
                int num_blocks = min(N, 65536);
                add3_single_block<<<num_blocks, dim3(32, (threads_per_block + 31) / 32, 1)>>>(parts_a, ret, N, k, L);
            }else{
                const int block_size = 2048;
                int num_blocks_x = min((size_t)256, max((size_t)1, (L + block_size - 1) / block_size));
                int num_blocks_y = min(N, 65536 / num_blocks_x);
                add3_reduce_blocks<<<dim3(num_blocks_x, num_blocks_y, 1), dim3(32, 32, 1)>>>(
                    parts_a, ret, reinterpret_cast<ushort2 *>(parts_b), N, k, L
                );
                add3_combine_blocks<<<min(N, 65536), dim3(32, 32, 1)>>>(
                    reinterpret_cast<ushort2 *>(parts_b), N, L
                );
                add3_apply_blocks<<<dim3(num_blocks_x, num_blocks_y, 1), dim3(32, 32, 1)>>>(
                    ret, reinterpret_cast<ushort2 *>(parts_b), N, k, L
                );
            }
        }else{
            if (L <= 8192 || N >= 85){
                int threads_per_block = min(L>>1, (size_t)1024);
                int num_blocks = min(N, 65536);
                add2_single_block<<<num_blocks, dim3(32, (threads_per_block + 31) / 32, 1)>>>(
                    ret, reinterpret_cast<uint32_t *>(parts_b),
                    N, k, L
                );
            }else{
                const int block_size = 2048;
                int num_blocks_x = min((size_t)256, max((size_t)1, (L + block_size - 1) / block_size));
                int num_blocks_y = min(N, 65536 / num_blocks_x);
                add2_reduce_blocks<<<dim3(num_blocks_x, num_blocks_y, 1), dim3(32, 32, 1)>>>(
                    ret, reinterpret_cast<uint32_t *>(parts_b), reinterpret_cast<ushort2 *>(parts_a), N, k, L
                );
                add3_combine_blocks<<<min(N, 65536), dim3(32, 32, 1)>>>(
                    reinterpret_cast<ushort2 *>(parts_a), N, L
                );
                add3_apply_blocks<<<dim3(num_blocks_x, num_blocks_y, 1), dim3(32, 32, 1)>>>(
                    ret, reinterpret_cast<ushort2 *>(parts_a), N, k, L
                );
            }
        }

        A += ((size_t)N) * L_a;
        B += ((size_t)N) * L_b;
        ret += N * L;
    }
}

size_t batch_mul_ntt_workspace_size(uint32_t N, uint32_t L_a, uint32_t L_b){
    size_t L = (size_t)L_a + (size_t)L_b;
    size_t K = 1;
    while (K < L){
        K <<= 1;
    }
    if (K <= 2048){
        return 0;
    }
    // parts_a + parts_b, each is N*K uint3 values
    uint32_t N_batch = get_N_max(N, L_a, L_b);
    if (K == 4096){
        return N_batch * K * sizeof(uint3);
    }
    return N_batch * K * 2 * sizeof(uint3);
}
