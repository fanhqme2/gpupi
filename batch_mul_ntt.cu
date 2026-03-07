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
        uint32_t bitrev_seq_id = __brev((uint32_t)seq_id) >> (32 - i);
        uint3 twiddle_factor = mul_mod(
            roots_table_lv2[(bitrev_seq_id << (31 - i)) >> 16],
            roots_table_lv1[(bitrev_seq_id << (31 - i)) & 0xffff]
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
        uint32_t bitrev_seq_id = __brev((uint32_t)seq_id) >> (32 - i);
        uint3 twiddle_factor = mul_mod(
            roots_table_lv2[(uint32_t)(-(bitrev_seq_id << (31 - i))) >> 16],
            roots_table_lv1[(-(bitrev_seq_id << (31 - i))) & 0xffff]
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

        copy_to_parts<<<170, 128>>>(parts_a, A, N, k, L_a);
        copy_to_parts<<<170, 128>>>(parts_b, B, N, k, L_b);

        for (int i = 0; i < k; i ++){
            const int threads_per_block = 128;
            int num_blocks = min(((((size_t)N) << (k + 1)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
            fft_level_forward<<<num_blocks, threads_per_block>>>(
                parts_a, // parts_b must be adjacent to it
                k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                N * 2 // we use 2 * N to do parts_a and parts_b simultaneously
            );
        }

        // fuse divide by K into pointwise mul
        pointwise_mul<<<170, 128>>>(parts_a, parts_b, ((size_t)N) << k, tables.inv2n_table + k);
        
        for (int i = k - 1; i >= 0; i --){
            const int threads_per_block = 128;
            int num_blocks = min(((((size_t)N) << (k + 1)) + threads_per_block - 1) / threads_per_block, (size_t)65536);
            fft_level_backward<<<num_blocks, threads_per_block>>>(
                parts_a,
                k, i, tables.roots_table_lv1, tables.roots_table_lv2,
                N
            );
        }

        int threads_per_block = min(L, (size_t)1024);
        int num_blocks = min(N, 65536);
        add3_single_block<<<num_blocks, dim3(32, (threads_per_block + 31) / 32, 1)>>>(parts_a, ret, N, k, L);
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
