#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>
#include <gmp.h>

#include "batch_mul_ntt.h"

static const uint32_t ROOT0 = 0x46038aebu;
static const uint32_t ROOT1 = 0x55c21a87u;
static const uint32_t ROOT2 = 0xa257c693u;

static const uint32_t ROOT65536_0 = 0xea78dff7u;
static const uint32_t ROOT65536_1 = 0xbcc6f614u;
static const uint32_t ROOT65536_2 = 0x02c17ad1u;

static const uint32_t INV2_0 = 0x80000001u;
static const uint32_t INV2_1 = 0xffffffffu;
static const uint32_t INV2_2 = 0x7fffffffu;

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

static void mpz_from_u96(mpz_t out, uint32_t w0, uint32_t w1, uint32_t w2) {
    mpz_set_ui(out, (unsigned long)w2);
    mpz_mul_2exp(out, out, 32);
    mpz_add_ui(out, out, (unsigned long)w1);
    mpz_mul_2exp(out, out, 32);
    mpz_add_ui(out, out, (unsigned long)w0);
}

static void u96_from_mpz(uint32_t out[3], const mpz_t in) {
    mpz_t t;
    mpz_init_set(t, in);
    out[0] = (uint32_t)mpz_get_ui(t);
    mpz_fdiv_q_2exp(t, t, 32);
    out[1] = (uint32_t)mpz_get_ui(t);
    mpz_fdiv_q_2exp(t, t, 32);
    out[2] = (uint32_t)mpz_get_ui(t);
    mpz_clear(t);
}

static void build_expected_power_table(
    std::vector<uint32_t>& out,
    int n,
    uint32_t base0,
    uint32_t base1,
    uint32_t base2,
    const mpz_t mod
) {
    out.resize((size_t)n * 3);

    mpz_t base, cur;
    mpz_init(base);
    mpz_init(cur);
    mpz_from_u96(base, base0, base1, base2);
    mpz_set_ui(cur, 1);

    for (int i = 0; i < n; ++i) {
        uint32_t limbs[3];
        u96_from_mpz(limbs, cur);
        out[(size_t)i * 3 + 0] = limbs[0];
        out[(size_t)i * 3 + 1] = limbs[1];
        out[(size_t)i * 3 + 2] = limbs[2];

        mpz_mul(cur, cur, base);
        mpz_mod(cur, cur, mod);
    }

    mpz_clear(base);
    mpz_clear(cur);
}

static bool compare_table(
    const char* name,
    const std::vector<uint32_t>& got,
    const std::vector<uint32_t>& expected,
    int n
) {
    for (int i = 0; i < n; ++i) {
        size_t off = (size_t)i * 3;
        if (got[off + 0] != expected[off + 0] ||
            got[off + 1] != expected[off + 1] ||
            got[off + 2] != expected[off + 2]) {
            printf("%s mismatch at i=%d\n", name, i);
            printf("  got      = %08x %08x %08x\n", got[off + 2], got[off + 1], got[off + 0]);
            printf("  expected = %08x %08x %08x\n", expected[off + 2], expected[off + 1], expected[off + 0]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N_ROOT = 65536;
    const int N_INV2N = 32; // matches init_ntt_precomputed_tables implementation

    const size_t bytes_lv1 = (size_t)N_ROOT * sizeof(uint3);
    const size_t bytes_lv2 = (size_t)N_ROOT * sizeof(uint3);
    const size_t bytes_inv = (size_t)N_INV2N * sizeof(uint3);

    uint3* d_lv1 = nullptr;
    uint3* d_lv2 = nullptr;
    uint3* d_inv = nullptr;

    CUDA_CHECK(cudaMalloc(&d_lv1, bytes_lv1));
    CUDA_CHECK(cudaMalloc(&d_lv2, bytes_lv2));
    CUDA_CHECK(cudaMalloc(&d_inv, bytes_inv));

    NTTPrecomputedTables tables;
    tables.roots_table_lv1 = d_lv1;
    tables.roots_table_lv2 = d_lv2;
    tables.inv2n_table = d_inv;

    init_ntt_precomputed_tables(&tables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint3> h_lv1_u3(N_ROOT);
    std::vector<uint3> h_lv2_u3(N_ROOT);
    std::vector<uint3> h_inv_u3(N_INV2N);
    CUDA_CHECK(cudaMemcpy(h_lv1_u3.data(), d_lv1, bytes_lv1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lv2_u3.data(), d_lv2, bytes_lv2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_inv_u3.data(), d_inv, bytes_inv, cudaMemcpyDeviceToHost));

    std::vector<uint32_t> h_lv1((size_t)N_ROOT * 3);
    std::vector<uint32_t> h_lv2((size_t)N_ROOT * 3);
    std::vector<uint32_t> h_inv((size_t)N_INV2N * 3);
    for (int i = 0; i < N_ROOT; ++i) {
        h_lv1[(size_t)i * 3 + 0] = h_lv1_u3[i].x;
        h_lv1[(size_t)i * 3 + 1] = h_lv1_u3[i].y;
        h_lv1[(size_t)i * 3 + 2] = h_lv1_u3[i].z;
        h_lv2[(size_t)i * 3 + 0] = h_lv2_u3[i].x;
        h_lv2[(size_t)i * 3 + 1] = h_lv2_u3[i].y;
        h_lv2[(size_t)i * 3 + 2] = h_lv2_u3[i].z;
    }
    for (int i = 0; i < N_INV2N; ++i) {
        h_inv[(size_t)i * 3 + 0] = h_inv_u3[i].x;
        h_inv[(size_t)i * 3 + 1] = h_inv_u3[i].y;
        h_inv[(size_t)i * 3 + 2] = h_inv_u3[i].z;
    }

    mpz_t p;
    mpz_init(p);
    mpz_ui_pow_ui(p, 2u, 96u);
    mpz_sub_ui(p, p, 4294967296ul);
    mpz_add_ui(p, p, 1u);

    std::vector<uint32_t> exp_lv1;
    std::vector<uint32_t> exp_lv2;
    std::vector<uint32_t> exp_inv;

    build_expected_power_table(exp_lv1, N_ROOT, ROOT0, ROOT1, ROOT2, p);
    build_expected_power_table(exp_lv2, N_ROOT, ROOT65536_0, ROOT65536_1, ROOT65536_2, p);
    build_expected_power_table(exp_inv, N_INV2N, INV2_0, INV2_1, INV2_2, p);

    bool ok = true;
    ok = compare_table("roots_table_lv1", h_lv1, exp_lv1, N_ROOT) && ok;
    ok = compare_table("roots_table_lv2", h_lv2, exp_lv2, N_ROOT) && ok;
    ok = compare_table("inv2n_table", h_inv, exp_inv, N_INV2N) && ok;

    if (ok) {
        printf("init_ntt_precomputed_tables: PASSED\n");
    } else {
        printf("init_ntt_precomputed_tables: FAILED\n");
    }

    mpz_clear(p);
    CUDA_CHECK(cudaFree(d_lv1));
    CUDA_CHECK(cudaFree(d_lv2));
    CUDA_CHECK(cudaFree(d_inv));

    return ok ? 0 : 1;
}
