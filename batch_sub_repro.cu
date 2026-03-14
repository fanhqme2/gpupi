#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <vector>

#include "batch_sub.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        return 1; \
    } \
} while (0)

int main() {
    constexpr uint32_t N = 1;
    constexpr uint32_t L_a = 2049;
    constexpr uint32_t L_b = 2049;
    constexpr uint32_t L_c = 4096;
    constexpr uint32_t stride_A = L_a;
    constexpr uint32_t stride_B = L_b;
    constexpr uint32_t stride_C = L_c;

    std::vector<uint32_t> h_A((size_t)N * stride_A, 0u);
    std::vector<uint32_t> h_B((size_t)N * stride_B, 0u);
    std::vector<uint32_t> h_C((size_t)N * stride_C, 0u);

    // Force a borrow at the start of chunk 1 and require it to propagate through the full chunk.
    h_B[2048] = 1u;

    uint32_t * d_A = nullptr;
    uint32_t * d_B = nullptr;
    uint32_t * d_C = nullptr;
    uint32_t * d_workspace = nullptr;

    const size_t size_A = h_A.size() * sizeof(uint32_t);
    const size_t size_B = h_B.size() * sizeof(uint32_t);
    const size_t size_C = h_C.size() * sizeof(uint32_t);
    const size_t workspace_size = batch_sub_simple_workspace_size(N, L_a, L_b, L_c);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice));

    batch_sub_simple(d_A, d_B, d_C, d_workspace, N, L_a, L_b, L_c, stride_A, stride_B, stride_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = true;
    uint32_t mismatch_idx = 0;
    for (uint32_t i = 0; i < L_c; ++i) {
        const uint32_t expected = (i < 2048u) ? 0u : 0xffffffffu;
        if (h_C[i] != expected) {
            pass = false;
            mismatch_idx = i;
            break;
        }
    }

    if (!pass) {
        const uint32_t lo = (mismatch_idx > 4u) ? mismatch_idx - 4u : 0u;
        const uint32_t hi = (mismatch_idx + 5u < L_c) ? mismatch_idx + 5u : L_c;
        printf("Mismatch at limb %u\n", mismatch_idx);
        for (uint32_t i = lo; i < hi; ++i) {
            const uint32_t expected = (i < 2048u) ? 0u : 0xffffffffu;
            printf("  limb %u: expected=%08x actual=%08x\n", i, expected, h_C[i]);
        }
    } else {
        printf("Repro case PASSED\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_workspace));
    return pass ? 0 : 1;
}
