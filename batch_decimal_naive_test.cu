#include <cuda_runtime.h>
#include <curand.h>
#include <gmp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "batch_decimal_naive.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

#define CURAND_CHECK(call) do { \
    curandStatus_t _err = (call); \
    if (_err != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, (int)_err); \
        exit(1); \
    } \
} while (0)

namespace {

constexpr uint64_t kTargetLimbWork = 100000000ull;
constexpr int kWarmupIters = 1;
constexpr int kTimedIters = 3;
constexpr uint32_t kLimbValues[] = {31u, 63u, 127u, 255u, 511u, 1023u, 2047u};

void import_words(mpz_t out, const uint32_t * words, uint32_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

std::string expected_decimal_digits(const uint32_t * words, uint32_t L_limb, int L_dec) {
    mpz_t numer;
    mpz_t denom;
    mpz_t q;
    mpz_t r;
    mpz_inits(numer, denom, q, r, NULL);

    import_words(numer, words, L_limb);
    mpz_ui_pow_ui(denom, 2u, (unsigned long)L_limb * 32ul);

    std::string out;
    out.reserve((size_t)L_dec);
    for (int start_idx = 0; start_idx < L_dec; start_idx += 9) {
        mpz_mul_ui(numer, numer, 1000000000u);
        mpz_fdiv_qr(q, r, numer, denom);
        mpz_set(numer, r);

        const unsigned long chunk = mpz_get_ui(q);
        char buf[10];
        snprintf(buf, sizeof(buf), "%09lu", chunk);
        const int chunk_len = std::min(9, L_dec - start_idx);
        out.append(buf, buf + chunk_len);
    }

    mpz_clears(numer, denom, q, r, NULL);
    std::reverse(out.begin(), out.end());
    return out;
}

bool run_case(
    const char * label,
    const std::vector<uint32_t> & h_A,
    uint32_t N,
    int L_limb,
    uint32_t stride_A,
    int L_dec,
    uint32_t stride_B
) {
    std::vector<char> h_B((size_t)N * stride_B, '?');
    std::vector<char> expected((size_t)N * stride_B, '?');

    for (uint32_t i = 0; i < N; ++i) {
        const std::string digits = expected_decimal_digits(h_A.data() + (size_t)i * stride_A, L_limb, L_dec);
        for (int j = 0; j < L_dec; ++j) {
            expected[(size_t)i * stride_B + j] = digits[(size_t)j];
        }
    }

    uint32_t * d_A = nullptr;
    char * d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(char)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(char), cudaMemcpyHostToDevice));

    batch_decimal_naive(d_A, d_B, N, L_limb, stride_A, L_dec, stride_B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, h_B.size() * sizeof(char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    for (uint32_t i = 0; i < N; ++i) {
        for (int j = 0; j < L_dec; ++j) {
            const char got = h_B[(size_t)i * stride_B + j];
            const char want = expected[(size_t)i * stride_B + j];
            if (got != want) {
                printf("%s failed at row %u digit %d: expected=%c actual=%c\n",
                       label, i, j, want, got);
                printf("  expected: %.*s\n", L_dec, expected.data() + (size_t)i * stride_B);
                printf("  actual  : %.*s\n", L_dec, h_B.data() + (size_t)i * stride_B);
                return false;
            }
        }
        for (uint32_t j = (uint32_t)L_dec; j < stride_B; ++j) {
            if (h_B[(size_t)i * stride_B + j] != '?') {
                printf("%s overwrote padding at row %u byte %u\n", label, i, j);
                return false;
            }
        }
    }

    printf("%s: PASSED\n", label);
    return true;
}

int decimal_digits_bound(uint32_t L_limb) {
    const double digits = std::ceil((double)L_limb * 32.0 * std::log10(2.0));
    return std::max(1, (int)digits);
}

uint32_t make_test_word(uint32_t row, uint32_t limb, uint32_t L_limb) {
    if (row == 0) {
        return 0u;
    }
    if (row == 1) {
        return (limb == 0) ? 1u : 0u;
    }
    if (row == 2) {
        return (limb + 1 == L_limb) ? 0x80000000u : 0u;
    }
    if (row == 3) {
        return 0xffffffffu;
    }
    if (row == 4) {
        return (limb & 1u) ? 0xaaaaaaaau : 0x55555555u;
    }
    if (row == 5) {
        return (limb & 1u) ? 0xffffffffu : 0u;
    }

    uint32_t x = 0x9e3779b9u * (row + 1u);
    x ^= 0x85ebca6bu * (limb + 1u);
    x ^= 0xc2b2ae35u * (L_limb + 1u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

bool run_generated_case(const char * label, uint32_t N, uint32_t L_limb, int extra_digits) {
    const uint32_t stride_A = L_limb + 1u + (L_limb & 1u);
    const int L_dec = decimal_digits_bound(L_limb) + extra_digits;
    const uint32_t stride_B = (uint32_t)L_dec + 5u;
    std::vector<uint32_t> h_A((size_t)N * stride_A, 0u);

    for (uint32_t row = 0; row < N; ++row) {
        for (uint32_t limb = 0; limb < L_limb; ++limb) {
            h_A[(size_t)row * stride_A + limb] = make_test_word(row, limb, L_limb);
        }
    }

    return run_case(label, h_A, N, (int)L_limb, stride_A, L_dec, stride_B);
}

void fill_random_words(uint32_t * d_A, size_t word_count) {
    curandGenerator_t gen = nullptr;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 123456789ull));
    CURAND_CHECK(curandGenerate(gen, d_A, word_count));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

double benchmark_case(uint32_t N, uint32_t L_limb, uint32_t stride_A, int L_dec, uint32_t stride_B) {
    uint32_t * d_A = nullptr;
    char * d_B = nullptr;
    const size_t words_A = (size_t)N * stride_A;
    const size_t bytes_B = (size_t)N * stride_B;

    CUDA_CHECK(cudaMalloc(&d_A, words_A * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    fill_random_words(d_A, words_A);
    CUDA_CHECK(cudaMemset(d_B, 0, bytes_B));

    for (int iter = 0; iter < kWarmupIters; ++iter) {
        batch_decimal_naive(d_A, d_B, N, (int)L_limb, stride_A, L_dec, stride_B);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kTimedIters; ++iter) {
        batch_decimal_naive(d_A, d_B, N, (int)L_limb, stride_A, L_dec, stride_B);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    return (double)elapsed_ms / (double)kTimedIters;
}

void run_benchmarks() {
    printf("batch_decimal_naive benchmark\n");
    printf("target N*L_limb ~= %llu\n", (unsigned long long)kTargetLimbWork);
    printf("L_dec = ceil(32 * L_limb * log10(2))\n");
    printf("\n");
    printf("%8s %12s %10s %12s %12s %12s %12s\n",
           "L_limb", "N", "L_dec", "time_ms", "in_GiB/s", "out_GiB/s", "digits_G/s");

    for (uint32_t L_limb : kLimbValues) {
        const uint32_t N = std::max<uint32_t>(1u, (uint32_t)(kTargetLimbWork / L_limb));
        const uint32_t stride_A = L_limb;
        const int L_dec = decimal_digits_bound(L_limb);
        const uint32_t stride_B = (uint32_t)L_dec;

        const double elapsed_ms = benchmark_case(N, L_limb, stride_A, L_dec, stride_B);
        const double seconds = elapsed_ms * 1e-3;
        const double input_gib_s = ((double)N * (double)L_limb * sizeof(uint32_t)) / seconds / (1024.0 * 1024.0 * 1024.0);
        const double output_gib_s = ((double)N * (double)L_dec) / seconds / (1024.0 * 1024.0 * 1024.0);
        const double digits_g_s = ((double)N * (double)L_dec) / seconds / 1e9;

        printf("%8u %12u %10d %12.3f %12.3f %12.3f %12.3f\n",
               L_limb, N, L_dec, elapsed_ms, input_gib_s, output_gib_s, digits_g_s);
        fflush(stdout);
    }
}

}  // namespace

int main() {
    bool ok = true;

    {
        struct CorrectnessConfig {
            const char * label;
            uint32_t N;
            uint32_t L_limb;
            int extra_digits;
        };
        const CorrectnessConfig configs[] = {
            {"correctness_n1_l1", 1u, 1u, 0},
            {"correctness_n2_l1", 2u, 1u, 7},
            {"correctness_n4_l2", 4u, 2u, 0},
            {"correctness_n70000_l1", 70000u, 1u, 2},
            {"correctness_n7_l3", 7u, 3u, 5},
            {"correctness_n8_l4", 8u, 4u, 0},
            {"correctness_n5_l5", 5u, 5u, 9},
            {"correctness_n9_l8", 9u, 8u, 3},
            {"correctness_n3_l15", 3u, 15u, 0},
            {"correctness_n6_l31", 6u, 31u, 11},
            {"correctness_n2_l63", 2u, 63u, 0},
            {"correctness_n1_l127", 1u, 127u, 13},
            {"correctness_n4_l255", 4u, 255u, 0},
            {"correctness_n2_l511", 2u, 511u, 17},
            {"correctness_n1_l1023", 1u, 1023u, 0},
            {"correctness_n1_l2047", 1u, 2047u, 19},
        };

        for (const CorrectnessConfig & cfg : configs) {
            ok &= run_generated_case(cfg.label, cfg.N, cfg.L_limb, cfg.extra_digits);
            if (!ok){
                break;
            }
        }
    }

    if (!ok) {
        return 1;
    }
    printf("All batch_decimal_naive tests passed.\n");
    run_benchmarks();
    return 0;
}
/*
  L_limb            N      L_dec      time_ms     in_GiB/s    out_GiB/s   digits_G/s
      31      3225806        299        4.502       82.755      199.545      214.260
      63      1587301        607        8.462       44.025      106.045      113.865
     127       787401       1224       11.777       31.631       76.214       81.834
     255       392156       2457       19.582       19.024       45.826       49.206
     511       195694       4923       35.537       10.483       25.248       27.110
    1023        97751       9855       73.976        5.036       12.128       13.022
    2047        48851      19719      442.652        0.842        2.027        2.176
*/