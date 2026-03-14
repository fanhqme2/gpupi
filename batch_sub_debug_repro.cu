#include <cuda_runtime.h>
#include <cuda/warp>

#include <cstdio>
#include <cstdint>
#include <vector>

#include "batch_mul_addsub_asm.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        return 1; \
    } \
} while (0)

namespace {

__device__ __forceinline__ ushort2 combine_borrow_summary_dbg(ushort2 a, ushort2 b) {
    ushort compound = a.y + b.x;
    a.x += compound >> 2;
    if ((compound & 3) == 3) {
        a.y = b.y;
    } else {
        a.y = 0;
    }
    return a;
}

__global__ void debug_reduce_chunk_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    ushort2 * raw_borrow,
    ushort2 * warp_prefix_borrow,
    ushort2 * final_borrow,
    uint32_t * raw_r0,
    uint32_t * raw_r1,
    uint32_t * final_r0,
    uint32_t * final_r1,
    uint32_t L_a,
    uint32_t L_b
) {
    __shared__ ushort2 borrow_info[32];

    const uint32_t linear_tid = threadIdx.y * 32u + threadIdx.x;
    const uint32_t i = linear_tid * 2u;
    const uint32_t g0 = 2048u + i;
    const uint32_t g1 = g0 + 1u;

    uint32_t r0_value = (g0 < L_a) ? A[g0] : 0u;
    uint32_t r1_value = (g1 < L_a) ? A[g1] : 0u;
    const uint32_t c0_value = (g0 < L_b) ? B[g0] : 0u;
    const uint32_t c1_value = (g1 < L_b) ? B[g1] : 0u;
    ushort2 borrow;

    r0_value = sub_cc(r0_value, c0_value);
    r1_value = subc_cc(r1_value, c1_value);
    borrow.x = -subc(0, 0);
    sub_cc(r0_value, 2);
    subc_cc(r1_value, 0);
    if (subc(0, 0)) {
        borrow.y = 3u - (r0_value & 3u);
    } else {
        borrow.y = 0u;
    }

    raw_borrow[linear_tid] = borrow;
    raw_r0[linear_tid] = (g0 < 4096u) ? ((g0 < L_a) ? A[g0] : 0u) - ((g0 < L_b) ? B[g0] : 0u) : 0u;
    raw_r1[linear_tid] = (g1 < 4096u) ? ((g1 < L_a) ? A[g1] : 0u) - ((g1 < L_b) ? B[g1] : 0u) : 0u;

    for (int delta = 1; delta < 32; delta *= 2) {
        const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(borrow, delta);
        if (threadIdx.x >= delta) {
            borrow = combine_borrow_summary_dbg(borrow, borrow_prev);
        }
    }
    warp_prefix_borrow[linear_tid] = borrow;

    if (threadIdx.x == 31) {
        borrow_info[threadIdx.y] = borrow;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        ushort2 warp_borrow = borrow_info[threadIdx.x];
        for (int delta = 1; delta < 32; delta *= 2) {
            const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(warp_borrow, delta);
            if (threadIdx.x >= delta) {
                warp_borrow = combine_borrow_summary_dbg(warp_borrow, borrow_prev);
            }
        }
        borrow_info[threadIdx.x] = warp_borrow;
    }
    __syncthreads();

    ushort2 applied_borrow = borrow;
    applied_borrow = cuda::device::warp_shuffle_up<32, ushort2>(applied_borrow, 1);
    if (threadIdx.x == 0) {
        applied_borrow.x = (threadIdx.y == 0) ? 0u : borrow_info[threadIdx.y - 1].x;
    }
    final_borrow[linear_tid] = applied_borrow;

    r0_value = (g0 < L_a) ? A[g0] : 0u;
    r1_value = (g1 < L_a) ? A[g1] : 0u;
    r0_value = sub_cc(r0_value, c0_value);
    r1_value = subc_cc(r1_value, c1_value);
    r0_value = sub_cc(r0_value, (uint32_t)applied_borrow.x);
    r1_value = subc_cc(r1_value, 0);

    final_r0[linear_tid] = r0_value;
    final_r1[linear_tid] = r1_value;
    if (g0 < 4096u) C[g0] = r0_value;
    if (g1 < 4096u) C[g1] = r1_value;

    if (linear_tid < 32u) {
        warp_prefix_borrow[1024u + linear_tid] = borrow_info[linear_tid];
    }
}

void print_thread_window(
    const std::vector<ushort2> & raw_borrow,
    const std::vector<ushort2> & warp_prefix_borrow,
    const std::vector<ushort2> & final_borrow,
    const std::vector<uint32_t> & final_r0,
    const std::vector<uint32_t> & final_r1,
    uint32_t tid_lo,
    uint32_t tid_hi
) {
    for (uint32_t tid = tid_lo; tid <= tid_hi; ++tid) {
        const uint32_t warp = tid / 32u;
        const uint32_t lane = tid % 32u;
        printf(
            "tid=%4u warp=%2u lane=%2u raw=(%u,%u) warp=(%u,%u) applied=(%u,%u) out=(%08x,%08x)\n",
            tid,
            warp,
            lane,
            raw_borrow[tid].x,
            raw_borrow[tid].y,
            warp_prefix_borrow[tid].x,
            warp_prefix_borrow[tid].y,
            final_borrow[tid].x,
            final_borrow[tid].y,
            final_r0[tid],
            final_r1[tid]
        );
    }
}

}  // namespace

int main() {
    constexpr uint32_t L_a = 2049;
    constexpr uint32_t L_b = 2049;
    constexpr uint32_t L_c = 4096;

    std::vector<uint32_t> h_A(L_c, 0u);
    std::vector<uint32_t> h_B(L_c, 0u);
    std::vector<uint32_t> h_C(L_c, 0u);
    h_B[2048] = 1u;

    uint32_t * d_A = nullptr;
    uint32_t * d_B = nullptr;
    uint32_t * d_C = nullptr;
    ushort2 * d_raw_borrow = nullptr;
    ushort2 * d_warp_prefix_borrow = nullptr;
    ushort2 * d_final_borrow = nullptr;
    uint32_t * d_raw_r0 = nullptr;
    uint32_t * d_raw_r1 = nullptr;
    uint32_t * d_final_r0 = nullptr;
    uint32_t * d_final_r1 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_raw_borrow, 1024u * sizeof(ushort2)));
    CUDA_CHECK(cudaMalloc(&d_warp_prefix_borrow, (1024u + 32u) * sizeof(ushort2)));
    CUDA_CHECK(cudaMalloc(&d_final_borrow, 1024u * sizeof(ushort2)));
    CUDA_CHECK(cudaMalloc(&d_raw_r0, 1024u * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_raw_r1, 1024u * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_final_r0, 1024u * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_final_r1, 1024u * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    debug_reduce_chunk_kernel<<<1, dim3(32u, 32u, 1u)>>>(
        d_A, d_B, d_C,
        d_raw_borrow, d_warp_prefix_borrow, d_final_borrow,
        d_raw_r0, d_raw_r1, d_final_r0, d_final_r1,
        L_a, L_b
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<ushort2> h_raw_borrow(1024u);
    std::vector<ushort2> h_warp_prefix_borrow(1024u + 32u);
    std::vector<ushort2> h_final_borrow(1024u);
    std::vector<uint32_t> h_final_r0(1024u);
    std::vector<uint32_t> h_final_r1(1024u);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_raw_borrow.data(), d_raw_borrow, h_raw_borrow.size() * sizeof(ushort2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_warp_prefix_borrow.data(), d_warp_prefix_borrow, h_warp_prefix_borrow.size() * sizeof(ushort2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_final_borrow.data(), d_final_borrow, h_final_borrow.size() * sizeof(ushort2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_final_r0.data(), d_final_r0, h_final_r0.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_final_r1.data(), d_final_r1, h_final_r1.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("Warp summaries after block-level scan:\n");
    for (uint32_t warp = 0; warp < 4u; ++warp) {
        const ushort2 s = h_warp_prefix_borrow[1024u + warp];
        printf("  warp %u summary=(%u,%u)\n", warp, s.x, s.y);
    }

    printf("\nThreads around the failure boundary:\n");
    print_thread_window(h_raw_borrow, h_warp_prefix_borrow, h_final_borrow, h_final_r0, h_final_r1, 30u, 35u);

    printf("\nOutput limbs around the failure boundary:\n");
    for (uint32_t i = 2108u; i <= 2118u; ++i) {
        printf("  limb %u: %08x\n", i, h_C[i]);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_raw_borrow));
    CUDA_CHECK(cudaFree(d_warp_prefix_borrow));
    CUDA_CHECK(cudaFree(d_final_borrow));
    CUDA_CHECK(cudaFree(d_raw_r0));
    CUDA_CHECK(cudaFree(d_raw_r1));
    CUDA_CHECK(cudaFree(d_final_r0));
    CUDA_CHECK(cudaFree(d_final_r1));
    return 0;
}
