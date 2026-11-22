#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// The pi executable submits its arithmetic and transfers on the default
// stream of a single device. Library callers retain ordinary CUDA allocation
// unless they explicitly establish this scope on their calling thread.
namespace pi_memory {
inline thread_local cudaMemPool_t pool = nullptr;

class PoolScope {
public:
    PoolScope() {
        int device, supported;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device) != cudaSuccess ||
            !supported) return;

        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        if (cudaMemPoolCreate(&owned_, &props) != cudaSuccess) return;
        uint64_t threshold = UINT64_MAX;
        if (cudaMemPoolSetAttribute(owned_, cudaMemPoolAttrReleaseThreshold, &threshold) == cudaSuccess)
            pool = owned_;
    }

    ~PoolScope() {
        pool = previous_;
        if (owned_) cudaMemPoolDestroy(owned_);
    }
    PoolScope(const PoolScope &) = delete;
    PoolScope &operator=(const PoolScope &) = delete;

private:
    cudaMemPool_t previous_ = pool;
    cudaMemPool_t owned_ = nullptr;
};

template<typename T>
cudaError_t allocate(T **ptr, size_t bytes) {
    return pool ? cudaMallocFromPoolAsync(reinterpret_cast<void **>(ptr), bytes, pool, 0)
                : cudaMalloc(ptr, bytes);
}

inline cudaError_t release(void *ptr) {
    return pool ? cudaFreeAsync(ptr, 0) : cudaFree(ptr);
}
} // namespace pi_memory
