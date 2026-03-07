#include "batch_mul_ntt.h"
#include "batch_mul_addsub_asm.h"

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

void init_ntt_precomputed_tables(NTTPrecomputedTables * tables){
    fill_in_power_table<<<1, 1024>>>(tables->roots_table_lv1, 65536, make_uint3(arr_root0, arr_root1, arr_root2));
    fill_in_power_table<<<1, 1024>>>(tables->roots_table_lv2, 65536, make_uint3(arr_root65536_0, arr_root65536_1, arr_root65536_2));
    fill_in_power_table<<<1, 32>>>(tables->inv2n_table, 32, make_uint3(arr_inv2_0, arr_inv2_1, arr_inv2_2));
}
