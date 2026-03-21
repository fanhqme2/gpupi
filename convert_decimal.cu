#include "convert_decimal.h"

#include <cuda_runtime.h>
#include <gmp.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

#include "batch_decimal_naive.h"

namespace {

constexpr double kLog2_10 = 3.32192809488736234787;
constexpr uint32_t kPrecisionMarginBits = 34u;

struct ArrayOwner {
    BatchMPArray array{};

    ~ArrayOwner() {
        batch_mp_array_release(array);
    }

    BatchMPArray release() {
        BatchMPArray out = array;
        array = {};
        return out;
    }
};

uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t bits_for_digits(uint32_t digits) {
    const double bits = std::ceil((double)digits * kLog2_10) + (double)kPrecisionMarginBits;
    return (uint32_t)bits;
}

uint32_t levels_for_shape(uint32_t total_digits, uint32_t leaf_digits) {
    if (leaf_digits == 0u || total_digits == 0u || (total_digits % leaf_digits) != 0u) {
        return UINT32_MAX;
    }
    uint32_t ratio = total_digits / leaf_digits;
    uint32_t levels = 0u;
    while (ratio > 1u && (ratio & 1u) == 0u) {
        ratio >>= 1u;
        ++levels;
    }
    return (ratio == 1u) ? levels : UINT32_MAX;
}

cudaError_t store_mpz_to_device(const mpz_t value, BatchMPArray array) {
    std::vector<uint32_t> host(array.length == 0u ? 1u : array.length, 0u);
    size_t written = 0u;
    mpz_export(host.data(), &written, -1, sizeof(uint32_t), 0, 0, value);
    if (written > array.length) {
        return cudaErrorInvalidValue;
    }
    if (array.length == 0u) {
        return cudaSuccess;
    }
    return cudaMemcpy(array.data, host.data(), array.length * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

__global__ void extract_shifted_rows_kernel(
    const uint32_t * src,
    uint32_t src_stride,
    uint32_t src_length,
    uint32_t * dst,
    uint32_t dst_stride,
    uint32_t dst_length,
    uint32_t rows,
    uint32_t shift_bits,
    uint32_t dst_row_offset
) {
    const uint64_t total = (uint64_t)rows * (uint64_t)dst_length;
    const uint32_t word_shift = shift_bits >> 5;
    const uint32_t bit_shift = shift_bits & 31u;
    for (uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += (uint64_t)blockDim.x * gridDim.x) {
        const uint32_t row = (uint32_t)(tid / dst_length);
        const uint32_t limb = (uint32_t)(tid % dst_length);
        const uint32_t src_idx = limb + word_shift;
        uint64_t value = 0u;
        if (src_idx < src_length) {
            value = (uint64_t)src[(size_t)row * src_stride + src_idx] >> bit_shift;
            if (bit_shift != 0u && src_idx + 1u < src_length) {
                value |= (uint64_t)src[(size_t)row * src_stride + src_idx + 1u] << (32u - bit_shift);
            }
        }
        dst[(size_t)(row * 2u + dst_row_offset) * dst_stride + limb] = (uint32_t)value;
    }
}

__global__ void extract_low_bits_rows_kernel(
    const uint32_t * src,
    uint32_t src_stride,
    uint32_t * dst,
    uint32_t dst_stride,
    uint32_t dst_length,
    uint32_t rows,
    uint32_t top_mask
) {
    const uint64_t total = (uint64_t)rows * (uint64_t)dst_length;
    for (uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += (uint64_t)blockDim.x * gridDim.x) {
        const uint32_t row = (uint32_t)(tid / dst_length);
        const uint32_t limb = (uint32_t)(tid % dst_length);
        uint32_t value = src[(size_t)row * src_stride + limb];
        if (top_mask != 0xffffffffu && limb + 1u == dst_length) {
            value &= top_mask;
        }
        dst[(size_t)row * dst_stride + limb] = value;
    }
}

__global__ void assemble_decimal_output_kernel(
    const char * chunks_reversed,
    char * output_digits,
    uint32_t chunk_digits,
    uint32_t chunk_count
) {
    const uint64_t total = (uint64_t)chunk_digits * (uint64_t)chunk_count;
    for (uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += (uint64_t)blockDim.x * gridDim.x) {
        const uint32_t chunk = (uint32_t)(tid / chunk_digits);
        const uint32_t digit = (uint32_t)(tid % chunk_digits);
        output_digits[(size_t)chunk * chunk_digits + digit] =
            chunks_reversed[(size_t)chunk * chunk_digits + (chunk_digits - 1u - digit)];
    }
}

cudaError_t launch_extract_shifted_rows(
    const BatchMPArray & src,
    BatchMPArray & dst,
    uint32_t rows,
    uint32_t shift_bits,
    uint32_t dst_row_offset
) {
    if (dst.length == 0u || rows == 0u) {
        return cudaSuccess;
    }
    const uint64_t total = (uint64_t)rows * (uint64_t)dst.length;
    const uint32_t blocks = (uint32_t)std::min<uint64_t>((total + 255u) / 256u, 65535u);
    extract_shifted_rows_kernel<<<blocks, 256>>>(
        src.data, src.stride, src.length,
        dst.data, dst.stride, dst.length, rows, shift_bits, dst_row_offset
    );
    return cudaGetLastError();
}

cudaError_t launch_extract_low_bits_rows(
    const BatchMPArray & src,
    BatchMPArray & dst,
    uint32_t rows,
    uint32_t valid_bits
) {
    if (dst.length == 0u || rows == 0u) {
        return cudaSuccess;
    }
    uint32_t top_mask = 0xffffffffu;
    const uint32_t top_bits = valid_bits & 31u;
    if (top_bits != 0u) {
        top_mask = (1u << top_bits) - 1u;
    }
    const uint64_t total = (uint64_t)rows * (uint64_t)dst.length;
    const uint32_t blocks = (uint32_t)std::min<uint64_t>((total + 255u) / 256u, 65535u);
    extract_low_bits_rows_kernel<<<blocks, 256>>>(
        src.data, src.stride, dst.data, dst.stride, dst.length, rows, top_mask
    );
    return cudaGetLastError();
}

cudaError_t launch_assemble_decimal_output(
    const char * chunks_reversed,
    char * output_digits,
    uint32_t chunk_digits,
    uint32_t chunk_count
) {
    if (chunk_digits == 0u || chunk_count == 0u) {
        return cudaSuccess;
    }
    const uint64_t total = (uint64_t)chunk_digits * (uint64_t)chunk_count;
    const uint32_t blocks = (uint32_t)std::min<uint64_t>((total + 255u) / 256u, 65535u);
    assemble_decimal_output_kernel<<<blocks, 256>>>(
        chunks_reversed, output_digits, chunk_digits, chunk_count
    );
    return cudaGetLastError();
}

}  // namespace

cudaError_t convert_decimal_precompute_powers(
    BatchMPContext * ctx,
    ConvertDecimalPowerTable * table,
    uint32_t leaf_digits,
    uint32_t levels
) {
    if (ctx == nullptr || table == nullptr || leaf_digits == 0u) {
        return cudaErrorInvalidValue;
    }
    table->powers = nullptr;
    table->levels = 0u;
    table->leaf_digits = leaf_digits;

    if (levels == 0u) {
        return cudaSuccess;
    }

    BatchMPArray * powers = new (std::nothrow) BatchMPArray[levels];
    if (powers == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    for (uint32_t i = 0; i < levels; ++i) {
        powers[i] = {};
    }
    auto release_partial = [&](uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            batch_mp_array_release(powers[i]);
        }
        delete [] powers;
    };

    mpz_t base_value;
    mpz_init(base_value);
    mpz_ui_pow_ui(base_value, 5u, (unsigned long)leaf_digits);
    const uint32_t base_limbs = std::max<uint32_t>(
        1u, ceil_div_u32((uint32_t)mpz_sizeinbase(base_value, 2), 32u)
    );
    powers[0] = batch_mp_array_create(1u, base_limbs);
    if (powers[0].data == nullptr) {
        mpz_clear(base_value);
        release_partial(0u);
        return cudaErrorMemoryAllocation;
    }
    cudaError_t err = store_mpz_to_device(base_value, powers[0]);
    mpz_clear(base_value);
    if (err != cudaSuccess) {
        release_partial(1u);
        return err;
    }
    for (uint32_t level = 1u; level < levels; ++level) {
        ArrayOwner next{batch_mp_array_create(1u, powers[level - 1u].length * 2u)};
        if (next.array.data == nullptr) {
            release_partial(level);
            return cudaErrorMemoryAllocation;
        }
        err = batch_mp_mul(ctx, powers[level - 1u], powers[level - 1u], next.array);
        if (err != cudaSuccess) {
            release_partial(level);
            return err;
        }
        err = next.array.compact(ctx);
        if (err != cudaSuccess) {
            release_partial(level);
            return err;
        }
        powers[level] = next.release();
    }

    table->powers = powers;
    table->levels = levels;
    return cudaSuccess;
}

void convert_decimal_release_powers(ConvertDecimalPowerTable * table) {
    if (table == nullptr) {
        return;
    }
    if (table->powers != nullptr) {
        for (uint32_t i = 0; i < table->levels; ++i) {
            batch_mp_array_release(table->powers[i]);
        }
        delete [] table->powers;
    }
    table->powers = nullptr;
    table->levels = 0u;
}

cudaError_t convert_decimal_equal_split(
    BatchMPContext * ctx,
    BatchMPArray input,
    uint32_t input_bits,
    uint32_t total_digits,
    uint32_t leaf_digits,
    const ConvertDecimalPowerTable * table,
    char * output_digits
) {
    if (ctx == nullptr || output_digits == nullptr || input.batch_size != 1u || input.data == nullptr) {
        return cudaErrorInvalidValue;
    }
    const uint32_t levels = levels_for_shape(total_digits, leaf_digits);
    if (levels == UINT32_MAX || table == nullptr || table->leaf_digits != leaf_digits || table->levels != levels) {
        return cudaErrorInvalidValue;
    }

    std::vector<uint32_t> digits_per_level(levels + 1u);
    std::vector<uint32_t> bits_per_level(levels + 1u);
    std::vector<uint32_t> limbs_per_level(levels + 1u);
    digits_per_level[0] = total_digits;
    bits_per_level[0] = input_bits;
    limbs_per_level[0] = input.length;
    for (uint32_t level = 1u; level <= levels; ++level) {
        digits_per_level[level] = digits_per_level[level - 1u] >> 1u;
        bits_per_level[level] = bits_for_digits(digits_per_level[level]);
        limbs_per_level[level] = ceil_div_u32(bits_per_level[level], 32u);
    }

    BatchMPArray current = input;
    bool own_current = false;

    auto cleanup_current = [&]() {
        if (own_current) {
            batch_mp_array_release(current);
            current = {};
            own_current = false;
        }
    };

    if (levels == 0u) {
        const uint32_t leaf_bits_needed = bits_for_digits(total_digits);
        const uint32_t leaf_limbs_needed = ceil_div_u32(leaf_bits_needed, 32u);
        if (input_bits < leaf_bits_needed) {
            return cudaErrorInvalidValue;
        }
        ArrayOwner truncated{batch_mp_array_create(1u, leaf_limbs_needed)};
        if (truncated.array.data == nullptr) {
            return cudaErrorMemoryAllocation;
        }
        cudaError_t err = cudaMemset(truncated.array.data, 0, (size_t)truncated.array.batch_size * truncated.array.stride * sizeof(uint32_t));
        if (err != cudaSuccess) {
            return err;
        }
        err = launch_extract_shifted_rows(
            input, truncated.array, 1u, input_bits - leaf_bits_needed, 0u
        );
        if (err != cudaSuccess) {
            return err;
        }
        current = truncated.release();
        own_current = true;
    } else {
        for (uint32_t level = 0u; level < levels; ++level) {
            const uint32_t child_digits = digits_per_level[level + 1u];
            const uint32_t child_bits = bits_per_level[level + 1u];
            const uint32_t half_digits = child_digits;
            if (current.length < limbs_per_level[level]) {
                cleanup_current();
                return cudaErrorInvalidValue;
            }
            if (bits_per_level[level] <= half_digits || bits_per_level[level] < child_bits) {
                cleanup_current();
                return cudaErrorInvalidValue;
            }

            ArrayOwner next{batch_mp_array_create(current.batch_size * 2u, limbs_per_level[level + 1u])};
            if (next.array.data == nullptr) {
                cleanup_current();
                return cudaErrorMemoryAllocation;
            }
            cudaError_t err = cudaMemset(
                next.array.data, 0, (size_t)next.array.batch_size * next.array.stride * sizeof(uint32_t)
            );
            if (err != cudaSuccess) {
                cleanup_current();
                return err;
            }

            err = launch_extract_shifted_rows(
                current, next.array, current.batch_size, bits_per_level[level] - child_bits, 0u
            );
            if (err != cudaSuccess) {
                cleanup_current();
                return err;
            }

            const uint32_t mantissa_bits = bits_per_level[level] - half_digits;
            ArrayOwner mantissa{batch_mp_array_create(current.batch_size, ceil_div_u32(mantissa_bits, 32u))};
            if (mantissa.array.data == nullptr) {
                cleanup_current();
                return cudaErrorMemoryAllocation;
            }
            err = cudaMemset(
                mantissa.array.data, 0, (size_t)mantissa.array.batch_size * mantissa.array.stride * sizeof(uint32_t)
            );
            if (err != cudaSuccess) {
                cleanup_current();
                return err;
            }
            err = launch_extract_low_bits_rows(current, mantissa.array, current.batch_size, mantissa_bits);
            if (err != cudaSuccess) {
                cleanup_current();
                return err;
            }

            const BatchMPArray power = {
                .data = table->powers[levels - 1u - level].data,
                .length = table->powers[levels - 1u - level].length,
                .batch_size = current.batch_size,
                .stride = 0u
            };
            ArrayOwner product{batch_mp_array_create(
                current.batch_size, mantissa.array.length + power.length
            )};
            if (product.array.data == nullptr) {
                cleanup_current();
                return cudaErrorMemoryAllocation;
            }
            err = batch_mp_mul(ctx, mantissa.array, power, product.array);
            if (err != cudaSuccess) {
                cleanup_current();
                return err;
            }

            err = launch_extract_shifted_rows(
                product.array, next.array, current.batch_size, mantissa_bits - child_bits, 1u
            );
            if (err != cudaSuccess) {
                cleanup_current();
                return err;
            }

            cleanup_current();
            current = next.release();
            own_current = true;
        }
    }

    ArrayOwner aligned{batch_mp_array_create(current.batch_size, current.length)};
    if (aligned.array.data == nullptr) {
        cleanup_current();
        return cudaErrorMemoryAllocation;
    }
    const uint32_t leaf_bits = (levels == 0u) ? bits_for_digits(total_digits) : bits_per_level[levels];
    const uint32_t align_shift = current.length * 32u - leaf_bits;
    cudaError_t err = batch_mp_shift_bits(ctx, current, aligned.array, (int32_t)align_shift);
    if (err != cudaSuccess) {
        cleanup_current();
        return err;
    }

    const size_t chunk_bytes = (size_t)current.batch_size * leaf_digits;
    char * chunks_reversed = nullptr;
    err = cudaMalloc(&chunks_reversed, chunk_bytes);
    if (err != cudaSuccess) {
        cleanup_current();
        return err;
    }

    batch_decimal_naive(
        aligned.array.data,
        chunks_reversed,
        current.batch_size,
        (int)aligned.array.length,
        aligned.array.stride,
        (int)leaf_digits,
        leaf_digits
    );
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = launch_assemble_decimal_output(
            chunks_reversed, output_digits, leaf_digits, current.batch_size
        );
    }

    cudaFree(chunks_reversed);
    cleanup_current();
    return err;
}
