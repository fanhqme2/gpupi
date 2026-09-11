#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>

// Exact host-to-host multiplication using a six-step NTT split across 1/2/4/8
// GPUs. Buffers belong to the engine and persist between calls. Calls on one
// engine are serialized; the caller must exclude other work on these devices.
class DistributedNTT {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    explicit DistributedNTT(int devices);
    ~DistributedNTT();
    void multiply(const uint32_t*a,size_t na,const uint32_t*b,size_t nb,uint32_t*out,size_t first=0);
    // Return words [first, na+nb). A short wrapped tail permits products up
    // to 1.25 * 2^32 words while retaining the field's 2^32 transform limit.
    void multiply_window(const uint32_t*a,size_t na,const uint32_t*b,size_t nb,uint32_t*out,size_t first);
};
