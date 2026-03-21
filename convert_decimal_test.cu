#include <cuda_runtime.h>
#include <gmp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "batch_arith.h"
#include "convert_decimal.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

namespace {

constexpr double kLog2_10 = 3.32192809488736234787;
constexpr uint32_t kLeafDigits = 477u;
constexpr uint32_t kFullLevels = 21u;
constexpr uint32_t kFullDigits = kLeafDigits << kFullLevels;
constexpr uint32_t kFullLimbs = 103812502u;

struct ArrayOwner {
    BatchMPArray array{};
    ~ArrayOwner() { batch_mp_array_release(array); }
};

uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t bits_for_digits(uint32_t digits) {
    return (uint32_t)std::ceil((double)digits * kLog2_10) + 34u;
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

uint32_t next_word(uint64_t & state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (uint32_t)((state * 2685821657736338717ull) >> 32);
}

void fill_random_fraction(std::vector<uint32_t> & words, uint32_t bits, uint64_t seed) {
    uint64_t state = seed ? seed : 1u;
    for (size_t i = 0; i < words.size(); ++i) {
        words[i] = next_word(state);
    }
    const uint32_t extra = bits & 31u;
    if (extra != 0u) {
        words.back() &= (1u << extra) - 1u;
    }
    if (!words.empty() && words.back() == 0u) {
        words.back() = 1u;
        if (extra != 0u) {
            words.back() &= (1u << extra) - 1u;
        }
    }
}

std::string exact_decimal_digits(const std::vector<uint32_t> & words, uint32_t bits, uint32_t digits) {
    mpz_t numer;
    mpz_t denom;
    mpz_t q;
    mpz_t r;
    mpz_inits(numer, denom, q, r, NULL);
    mpz_import(numer, words.size(), -1, sizeof(uint32_t), 0, 0, words.data());
    mpz_ui_pow_ui(denom, 2u, (unsigned long)bits);

    std::string out;
    out.reserve(digits);
    for (uint32_t start = 0; start < digits; start += 9u) {
        mpz_mul_ui(numer, numer, 1000000000u);
        mpz_fdiv_qr(q, r, numer, denom);
        mpz_set(numer, r);

        const unsigned long chunk = mpz_get_ui(q);
        char buf[10];
        snprintf(buf, sizeof(buf), "%09lu", chunk);
        const uint32_t chunk_digits = std::min<uint32_t>(9u, digits - start);
        out.append(buf, buf + chunk_digits);
    }

    mpz_clears(numer, denom, q, r, NULL);
    return out;
}

uint32_t max_zero_nine_run(const std::string & digits) {
    uint32_t best = 0u;
    uint32_t cur = 0u;
    char prev = '\0';
    for (char ch : digits) {
        if ((ch == '0' || ch == '9') && ch == prev) {
            ++cur;
        } else if (ch == '0' || ch == '9') {
            cur = 1u;
        } else {
            cur = 0u;
        }
        prev = ch;
        best = std::max(best, cur);
    }
    return best;
}

bool run_correctness_case(
    BatchMPContext * ctx,
    const char * label,
    uint32_t total_digits,
    uint32_t input_bits,
    uint64_t seed
) {
    const uint32_t levels = levels_for_shape(total_digits, kLeafDigits);
    const uint32_t limbs = ceil_div_u32(input_bits, 32u);
    std::vector<uint32_t> host_words(limbs);
    std::string expected;

    for (int attempt = 0; attempt < 8; ++attempt) {
        fill_random_fraction(host_words, input_bits, seed + (uint64_t)attempt * 0x9e3779b97f4a7c15ull);
        expected = exact_decimal_digits(host_words, input_bits, total_digits);
        if (max_zero_nine_run(expected) <= 12u) {
            break;
        }
    }

    ConvertDecimalPowerTable powers{};
    CUDA_CHECK(convert_decimal_precompute_powers(ctx, &powers, kLeafDigits, levels));

    ArrayOwner input{batch_mp_array_create(1u, limbs)};
    if (input.array.data == nullptr) {
        fprintf(stderr, "%s: input allocation failed\n", label);
        return false;
    }
    CUDA_CHECK(cudaMemcpy(
        input.array.data, host_words.data(), host_words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice
    ));

    char * d_digits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_digits, total_digits));
    CUDA_CHECK(convert_decimal_equal_split(
        ctx, input.array, input_bits, total_digits, kLeafDigits, &powers, d_digits
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::string got(total_digits, '?');
    CUDA_CHECK(cudaMemcpy(got.data(), d_digits, total_digits, cudaMemcpyDeviceToHost));

    cudaFree(d_digits);
    convert_decimal_release_powers(&powers);

    if (got != expected) {
        fprintf(stderr, "%s failed\n", label);
        fprintf(stderr, "expected prefix: %.64s\n", expected.c_str());
        fprintf(stderr, "got      prefix: %.64s\n", got.c_str());
        fprintf(stderr, "expected suffix: %.64s\n", expected.c_str() + std::max<int>(0, (int)expected.size() - 64));
        fprintf(stderr, "got      suffix: %.64s\n", got.c_str() + std::max<int>(0, (int)got.size() - 64));
        return false;
    }

    printf("%s: PASSED (digits=%u bits=%u max_run_0_9=%u)\n",
           label, total_digits, input_bits, max_zero_nine_run(expected));
    fflush(stdout);
    return true;
}

__global__ void fill_words_kernel(uint32_t * data, size_t count, uint64_t seed) {
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < count;
         idx += (size_t)blockDim.x * gridDim.x) {
        uint64_t x = seed + 0x9e3779b97f4a7c15ull * (idx + 1u);
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebull;
        x ^= x >> 31;
        data[idx] = (uint32_t)x;
    }
}

void run_full_benchmark() {
    BatchMPContext * ctx = batch_mp_init();
    if (ctx == nullptr) {
        fprintf(stderr, "batch_mp_init failed for full benchmark\n");
        exit(1);
    }

    printf("full benchmark setup\n");
    printf("  limbs=%u\n", kFullLimbs);
    printf("  digits=%u\n", kFullDigits);
    printf("  levels=%u leaf_digits=%u\n", kFullLevels, kLeafDigits);
    fflush(stdout);

    ConvertDecimalPowerTable powers{};
    cudaEvent_t precompute_start = nullptr;
    cudaEvent_t precompute_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&precompute_start));
    CUDA_CHECK(cudaEventCreate(&precompute_stop));
    CUDA_CHECK(cudaEventRecord(precompute_start));
    CUDA_CHECK(convert_decimal_precompute_powers(ctx, &powers, kLeafDigits, kFullLevels));
    CUDA_CHECK(cudaEventRecord(precompute_stop));
    CUDA_CHECK(cudaEventSynchronize(precompute_stop));
    float precompute_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&precompute_ms, precompute_start, precompute_stop));
    CUDA_CHECK(cudaEventDestroy(precompute_start));
    CUDA_CHECK(cudaEventDestroy(precompute_stop));
    printf("  power_table_ms=%.3f\n", precompute_ms);
    fflush(stdout);

    ArrayOwner input{batch_mp_array_create(1u, kFullLimbs)};
    if (input.array.data == nullptr) {
        fprintf(stderr, "full benchmark input allocation failed\n");
        exit(1);
    }
    const size_t word_count = (size_t)kFullLimbs;
    const uint32_t blocks = 65535u;
    fill_words_kernel<<<blocks, 256>>>(input.array.data, word_count, 0x123456789abcdef0ull);
    CUDA_CHECK(cudaGetLastError());

    char * d_digits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_digits, (size_t)kFullDigits));

    CUDA_CHECK(convert_decimal_equal_split(
        ctx, input.array, kFullLimbs * 32u, kFullDigits, kLeafDigits, &powers, d_digits
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(convert_decimal_equal_split(
        ctx, input.array, kFullLimbs * 32u, kFullDigits, kLeafDigits, &powers, d_digits
    ));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    char prefix[65];
    char suffix[65];
    memset(prefix, 0, sizeof(prefix));
    memset(suffix, 0, sizeof(suffix));
    CUDA_CHECK(cudaMemcpy(prefix, d_digits, 64u, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(suffix, d_digits + (size_t)kFullDigits - 64u, 64u, cudaMemcpyDeviceToHost));

    const double seconds = elapsed_ms * 1e-3;
    const double digits_g_s = (double)kFullDigits / seconds / 1e9;
    printf("full benchmark result\n");
    printf("  elapsed_ms=%.3f\n", elapsed_ms);
    printf("  digits_G_per_s=%.3f\n", digits_g_s);
    printf("  prefix=%s\n", prefix);
    printf("  suffix=%s\n", suffix);
    fflush(stdout);

    cudaFree(d_digits);
    convert_decimal_release_powers(&powers);
    batch_mp_destroy(ctx);
}

}  // namespace

int main(int argc, char ** argv) {
    bool skip_full = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--skip-full") == 0) {
            skip_full = true;
        }
    }

    BatchMPContext * ctx = batch_mp_init();
    if (ctx == nullptr) {
        fprintf(stderr, "batch_mp_init failed\n");
        return 1;
    }

    bool ok = true;
    ok &= run_correctness_case(ctx, "correctness_477", 477u, bits_for_digits(477u) + 64u, 0x1234u);
    ok &= run_correctness_case(ctx, "correctness_954", 954u, bits_for_digits(954u) + 64u, 0x2345u);
    ok &= run_correctness_case(ctx, "correctness_1908", 1908u, bits_for_digits(1908u) + 64u, 0x3456u);
    ok &= run_correctness_case(ctx, "correctness_3816", 3816u, bits_for_digits(3816u) + 64u, 0x4567u);

    batch_mp_destroy(ctx);

    if (!ok) {
        return 1;
    }
    printf("All convert_decimal correctness tests passed.\n");
    fflush(stdout);

    if (!skip_full) {
        run_full_benchmark();
    }
    return 0;
}
