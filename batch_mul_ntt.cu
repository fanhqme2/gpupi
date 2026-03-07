#include "batch_mul_ntt.h"
#include "batch_mul_addsub_asm.h"
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

__global__ void copy_to_parts(uint3 * parts, uint32_t * src, size_t N, int k, uint32_t L){
    size_t K = ((size_t)1) << k;
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < (N << k); i += blockDim.x * gridDim.x){
        uint32_t val = 0;
        size_t bid = i >> k;
        size_t idx = i & (K - 1);
        if (idx < L){
            val = src[bid * L + idx];
        }
        parts[i] = make_uint3(val, 0, 0);
    }
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

__global__ void fft_level_forward_radix4(
    uint3 *parts, int k, int i,
    uint3 *roots_table_lv1, uint3 *roots_table_lv2,
    size_t N
){
    const uint32_t step = 1u << (k - 2 - i);
    const uint32_t seq_len = 1u << i;

    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 2)); j += blockDim.x * gridDim.x){

        size_t group = j >> (k - 2 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));

        // seq | X | step+1    i
        // seq+1 | X | step    i + 1
        // seq | X X | step

        uint32_t bitrev_seq_id = (uint32_t)seq_id * 2;
        uint32_t bitrev_seq_id1 = (uint32_t)(seq_id * 2 + 0) * 2;
        uint32_t bitrev_seq_id2 = (uint32_t)(seq_id * 2 + 1) * 2;

        uint3 w0 = mul_mod(roots_table_lv1[bitrev_seq_id >> 16], roots_table_lv2[bitrev_seq_id & 0xffff]);
        uint3 w1 = mul_mod(roots_table_lv1[bitrev_seq_id1 >> 16], roots_table_lv2[bitrev_seq_id1 & 0xffff]);
        uint3 w2 = mul_mod(roots_table_lv1[bitrev_seq_id2 >> 16], roots_table_lv2[bitrev_seq_id2 & 0xffff]);

        size_t base = offset + step_id;

        uint3 x0 = parts[base];
        uint3 x1 = parts[base + step];
        uint3 x2 = parts[base + step * 2];
        uint3 x3 = parts[base + step * 3];

        // stage i
        uint3 wx2 = mul_mod(x2, w0);
        uint3 wx3 = mul_mod(x3, w0);

        uint3 a0 = add_mod(x0, wx2);
        uint3 a2 = sub_mod(x0, wx2);
        uint3 a1 = add_mod(x1, wx3);
        uint3 a3 = sub_mod(x1, wx3);

        // stage i+1
        uint3 wa1 = mul_mod(a1, w1);
        uint3 wa3 = mul_mod(a3, w2);

        parts[base]              = add_mod(a0, wa1);
        parts[base + step]       = sub_mod(a0, wa1);
        parts[base + step * 2]   = add_mod(a2, wa3);
        parts[base + step * 3]   = sub_mod(a2, wa3);
    }
}

__global__ void fft_level_forward_radix8(
    uint3 *parts, int k, int i,
    uint3 *roots_table_lv1, uint3 *roots_table_lv2,
    size_t N
){
    const uint32_t step = 1u << (k - 3 - i);
    const uint32_t seq_len = 1u << i;

    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 3)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 3 - i);
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

        size_t base = offset + step_id;

        uint3 x0 = parts[base];
        uint3 x1 = parts[base + step];
        uint3 x2 = parts[base + step * 2];
        uint3 x3 = parts[base + step * 3];
        uint3 x4 = parts[base + step * 4];
        uint3 x5 = parts[base + step * 5];
        uint3 x6 = parts[base + step * 6];
        uint3 x7 = parts[base + step * 7];

        // stage i
        uint3 wx4 = mul_mod(x4, w0);
        uint3 wx5 = mul_mod(x5, w0);
        uint3 wx6 = mul_mod(x6, w0);
        uint3 wx7 = mul_mod(x7, w0);

        uint3 b0 = add_mod(x0, wx4);
        uint3 b1 = add_mod(x1, wx5);
        uint3 b2 = add_mod(x2, wx6);
        uint3 b3 = add_mod(x3, wx7);
        uint3 b4 = sub_mod(x0, wx4);
        uint3 b5 = sub_mod(x1, wx5);
        uint3 b6 = sub_mod(x2, wx6);
        uint3 b7 = sub_mod(x3, wx7);

        // stage i+1
        uint3 wb2 = mul_mod(b2, w10);
        uint3 wb3 = mul_mod(b3, w10);
        uint3 wb6 = mul_mod(b6, w11);
        uint3 wb7 = mul_mod(b7, w11);

        uint3 c0 = add_mod(b0, wb2);
        uint3 c1 = add_mod(b1, wb3);
        uint3 c2 = sub_mod(b0, wb2);
        uint3 c3 = sub_mod(b1, wb3);
        uint3 c4 = add_mod(b4, wb6);
        uint3 c5 = add_mod(b5, wb7);
        uint3 c6 = sub_mod(b4, wb6);
        uint3 c7 = sub_mod(b5, wb7);

        // stage i+2
        uint3 wc1 = mul_mod(c1, w20);
        uint3 wc3 = mul_mod(c3, w21);
        uint3 wc5 = mul_mod(c5, w22);
        uint3 wc7 = mul_mod(c7, w23);

        parts[base]            = add_mod(c0, wc1);
        parts[base + step]     = sub_mod(c0, wc1);
        parts[base + step * 2] = add_mod(c2, wc3);
        parts[base + step * 3] = sub_mod(c2, wc3);
        parts[base + step * 4] = add_mod(c4, wc5);
        parts[base + step * 5] = sub_mod(c4, wc5);
        parts[base + step * 6] = add_mod(c6, wc7);
        parts[base + step * 7] = sub_mod(c6, wc7);
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

__global__ void fft_level_backward_radix4(uint3 * parts, int k, int i, uint3 * roots_table_lv1, uint3 * roots_table_lv2, size_t N){
    i--;
    uint32_t step = 1 << (k - 2 - i);
    uint32_t seq_len = 1 << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 2)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 2 - i);
        size_t offset = group << (k - i);
        uint32_t seq_id = (uint32_t)(group & (seq_len - 1));
        uint32_t step_id = (uint32_t)(j & (step - 1));

        // seq | X | step+1    i
        // seq+1 | X | step    i + 1
        // seq | X X | step

        uint32_t bitrev_seq_id = __brev(-__brev(seq_id * 2));
        uint32_t bitrev_seq_id1 = __brev(-__brev(seq_id * 4));
        uint32_t bitrev_seq_id2 = __brev(-__brev(seq_id * 4 + 2));

        uint3 w0 = mul_mod(roots_table_lv1[bitrev_seq_id >> 16], roots_table_lv2[bitrev_seq_id & 0xffff]);
        uint3 w1 = mul_mod(roots_table_lv1[bitrev_seq_id1 >> 16], roots_table_lv2[bitrev_seq_id1 & 0xffff]);
        uint3 w2 = mul_mod(roots_table_lv1[bitrev_seq_id2 >> 16], roots_table_lv2[bitrev_seq_id2 & 0xffff]);


        size_t base = offset + step_id;

        uint3 y0 = parts[base];
        uint3 y1 = parts[base + step];
        uint3 y2 = parts[base + step * 2];
        uint3 y3 = parts[base + step * 3];

        // stage i + 1

        uint3 a0 = add_mod(y0, y1);
        uint3 a1 = mul_mod(sub_mod(y0, y1), w1);
        uint3 a2 = add_mod(y2, y3);
        uint3 a3 = mul_mod(sub_mod(y2, y3), w2);

        // stage i
        uint3 x0 = add_mod(a0, a2);
        uint3 x2 = mul_mod(sub_mod(a0, a2), w0);
        uint3 x1 = add_mod(a1, a3);
        uint3 x3 = mul_mod(sub_mod(a1, a3), w0);

        parts[base]              = x0;
        parts[base + step]       = x1;
        parts[base + step * 2]   = x2;
        parts[base + step * 3]   = x3;
    }
}

__global__ void fft_level_backward_radix8(uint3 * parts, int k, int i, uint3 * roots_table_lv1, uint3 * roots_table_lv2, size_t N){
    i -= 2;
    uint32_t step = 1u << (k - 3 - i);
    uint32_t seq_len = 1u << i;
    for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < (N << (k - 3)); j += blockDim.x * gridDim.x){
        size_t group = j >> (k - 3 - i);
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

        size_t base = offset + step_id;

        uint3 y0 = parts[base];
        uint3 y1 = parts[base + step];
        uint3 y2 = parts[base + step * 2];
        uint3 y3 = parts[base + step * 3];
        uint3 y4 = parts[base + step * 4];
        uint3 y5 = parts[base + step * 5];
        uint3 y6 = parts[base + step * 6];
        uint3 y7 = parts[base + step * 7];

        // stage i+2
        uint3 a0 = add_mod(y0, y1);
        uint3 a1 = mul_mod(sub_mod(y0, y1), w20);
        uint3 a2 = add_mod(y2, y3);
        uint3 a3 = mul_mod(sub_mod(y2, y3), w21);
        uint3 a4 = add_mod(y4, y5);
        uint3 a5 = mul_mod(sub_mod(y4, y5), w22);
        uint3 a6 = add_mod(y6, y7);
        uint3 a7 = mul_mod(sub_mod(y6, y7), w23);

        // stage i+1
        uint3 b0 = add_mod(a0, a2);
        uint3 b2 = mul_mod(sub_mod(a0, a2), w10);
        uint3 b1 = add_mod(a1, a3);
        uint3 b3 = mul_mod(sub_mod(a1, a3), w10);
        uint3 b4 = add_mod(a4, a6);
        uint3 b6 = mul_mod(sub_mod(a4, a6), w11);
        uint3 b5 = add_mod(a5, a7);
        uint3 b7 = mul_mod(sub_mod(a5, a7), w11);

        // stage i
        parts[base]            = add_mod(b0, b4);
        parts[base + step]     = add_mod(b1, b5);
        parts[base + step * 2] = add_mod(b2, b6);
        parts[base + step * 3] = add_mod(b3, b7);
        parts[base + step * 4] = mul_mod(sub_mod(b0, b4), w0);
        parts[base + step * 5] = mul_mod(sub_mod(b1, b5), w0);
        parts[base + step * 6] = mul_mod(sub_mod(b2, b6), w0);
        parts[base + step * 7] = mul_mod(sub_mod(b3, b7), w0);
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

static uint32_t get_N_max(uint32_t N, uint32_t L_a, uint32_t L_b){
    size_t L = (size_t)L_a + (size_t)L_b;
    uint32_t cache_limit = max(4000000 / L, (size_t) 16);
    if (N >= cache_limit){
        N = cache_limit;
    }
    return max(1, min(N, 65536));
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

    uint32_t N_batch = get_N_max(N_total, L_a, L_b);

    for (uint32_t ni = 0; ni < N_total; ni += N_batch){
        uint32_t N = min(N_total - ni, N_batch);
        uint3 * parts_a = reinterpret_cast<uint3 *>(workspace);
        uint3 * parts_b = reinterpret_cast<uint3 *>(workspace + (((size_t)N) << k) * 3);

        // copy + pointwise mul time : 5.808 ms

        if (true){
            const int threads_per_block = 512;
            int num_blocks = min(((((size_t)N) << k) + threads_per_block - 1) / threads_per_block, (size_t)65536);
            copy_to_parts<<<num_blocks, threads_per_block>>>(parts_a, A, N, k, L_a);
            copy_to_parts<<<num_blocks, threads_per_block>>>(parts_b, B, N, k, L_b);
        }

        //if (L_a != 33554432){ // 56.765 ms

        for (int i = 0; i < k; i ++){
            if (i + 3 < k){
                const int threads_per_block = 64;
                int num_blocks = min(((((size_t)N) << (k - 3)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_forward_radix16<<<num_blocks, threads_per_block>>>(
                    parts_a, // parts_b must be adjacent to it
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N * 2 // we use 2 * N to do parts_a and parts_b simultaneously
                );
                i += 3;
            }else if (i + 2 < k){
                const int threads_per_block = 128;
                int num_blocks = min(((((size_t)N) << (k - 2)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_forward_radix8<<<num_blocks, threads_per_block>>>(
                    parts_a, // parts_b must be adjacent to it
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N * 2 // we use 2 * N to do parts_a and parts_b simultaneously
                );
                i += 2;
            }else if (i + 1 < k){
                const int threads_per_block = 256;
                int num_blocks = min(((((size_t)N) << (k - 1)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_forward_radix4<<<num_blocks, threads_per_block>>>(
                    parts_a, // parts_b must be adjacent to it
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N * 2 // we use 2 * N to do parts_a and parts_b simultaneously
                );
                i++;
            }else{
                const int threads_per_block = 512;
                int num_blocks = min(((((size_t)N) << (k)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_forward<<<num_blocks, threads_per_block>>>(
                    parts_a, // parts_b must be adjacent to it
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N * 2 // we use 2 * N to do parts_a and parts_b simultaneously
                );
            }
        }

        //}

        // fuse divide by K into pointwise mul
        if (true){
            const int threads_per_block = 512;
            int num_blocks = min(((((size_t)N) << k) + threads_per_block - 1) / threads_per_block, (size_t)65536);
            pointwise_mul<<<num_blocks, threads_per_block>>>(parts_a, parts_b, ((size_t)N) << k, tables.inv2n_table + k);
        }

        //if (L_a != 33554432){ // 30.580ms
        
        for (int i = k - 1; i >= 0; i --){
            if (i - 3 > 0){
                const int threads_per_block = 64;
                int num_blocks = min(((((size_t)N) << (k - 4)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_backward_radix16<<<num_blocks, threads_per_block>>>(
                    parts_a,
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
                i -= 3;
            }else if (i - 2 > 0){
                const int threads_per_block = 128;
                int num_blocks = min(((((size_t)N) << (k - 3)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_backward_radix8<<<num_blocks, threads_per_block>>>(
                    parts_a,
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
                i -= 2;
            }else if (i - 1 > 0){
                const int threads_per_block = 256;
                int num_blocks = min(((((size_t)N) << (k - 2)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_backward_radix4<<<num_blocks, threads_per_block>>>(
                    parts_a,
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
                i--;
            }else{
                const int threads_per_block = 512;
                int num_blocks = min(((((size_t)N) << (k - 1)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
                fft_level_backward<<<num_blocks, threads_per_block>>>(
                    parts_a,
                    k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                    N
                );
            }
        }

        //}

        //if (L_a != 33554432){ // 39.777ms

        int threads_per_block = min(L, (size_t)1024);
        int num_blocks = min(N, 65536);
        add3_single_block<<<num_blocks, dim3(32, (threads_per_block + 31) / 32, 1)>>>(parts_a, ret, N, k, L);

        //}

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
    // parts_a + parts_b, each is N*K uint3 values
    uint32_t N_batch = get_N_max(N, L_a, L_b);
    return N_batch * K * 2 * sizeof(uint3);
}
