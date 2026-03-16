#include <cuda_runtime.h>
#include <gmp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "batch_exactdiv_small.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

namespace {

constexpr uint32_t kPadPattern = 0xC3C3C3C3u;

uint32_t sample_u32_bounded(uint32_t max_value, std::mt19937_64 & rng) {
    std::uniform_int_distribution<uint64_t> dist(0u, (uint64_t)max_value);
    return (uint32_t)dist(rng);
}

void words_to_mpz(mpz_t out, const uint32_t * words, size_t count) {
    mpz_import(out, count, -1, sizeof(uint32_t), 0, 0, words);
}

bool verify_padding(const std::vector<uint32_t> & words, uint32_t N, uint32_t used, uint32_t stride) {
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t * row = words.data() + (size_t)i * stride;
        for (uint32_t j = used; j < stride; ++j) {
            if (row[j] != kPadPattern) {
                return false;
            }
        }
    }
    return true;
}

void fill_exact_products(
    std::vector<uint32_t> & dividend,
    std::vector<uint32_t> & quotient,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    uint32_t B,
    std::mt19937_64 & rng
) {
    std::uniform_int_distribution<uint32_t> dist(0u, 0xffffffffu);

    std::fill(dividend.begin(), dividend.end(), kPadPattern);
    std::fill(quotient.begin(), quotient.end(), kPadPattern);

    for (uint32_t row_idx = 0; row_idx < N; ++row_idx) {
        uint32_t * dividend_row = dividend.data() + (size_t)row_idx * stride;
        uint32_t * quotient_row = quotient.data() + (size_t)row_idx * stride;
        uint32_t carry = 0u;

        for (uint32_t limb = 0; limb < L; ++limb) {
            uint32_t q_limb;
            if (limb + 1u == L) {
                const uint32_t max_q = (0xffffffffu - carry) / B;
                q_limb = sample_u32_bounded(max_q, rng);
            } else {
                q_limb = dist(rng);
            }

            const uint64_t prod = (uint64_t)q_limb * (uint64_t)B + (uint64_t)carry;
            quotient_row[limb] = q_limb;
            dividend_row[limb] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
    }
}

void fill_exact_products_in_place(
    std::vector<uint32_t> & dividend,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    uint32_t B,
    std::mt19937_64 & rng
) {
    std::uniform_int_distribution<uint32_t> dist(0u, 0xffffffffu);

    std::fill(dividend.begin(), dividend.end(), kPadPattern);
    for (uint32_t row_idx = 0; row_idx < N; ++row_idx) {
        uint32_t * dividend_row = dividend.data() + (size_t)row_idx * stride;
        uint32_t carry = 0u;

        for (uint32_t limb = 0; limb < L; ++limb) {
            uint32_t q_limb;
            if (limb + 1u == L) {
                const uint32_t max_q = (0xffffffffu - carry) / B;
                q_limb = sample_u32_bounded(max_q, rng);
            } else {
                q_limb = dist(rng);
            }

            const uint64_t prod = (uint64_t)q_limb * (uint64_t)B + (uint64_t)carry;
            dividend_row[limb] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
    }
}

bool verify_rows_with_gmp(
    const std::vector<uint32_t> & dividend_before,
    const std::vector<uint32_t> & quotient_after,
    uint32_t B,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    bool verbose
) {
    for (uint32_t row_idx = 0; row_idx < N; ++row_idx) {
        const uint32_t * dividend_row = dividend_before.data() + (size_t)row_idx * stride;
        const uint32_t * quotient_row = quotient_after.data() + (size_t)row_idx * stride;

        mpz_t dividend_mpz;
        mpz_t quotient_mpz;
        mpz_t expected_mpz;
        mpz_inits(dividend_mpz, quotient_mpz, expected_mpz, NULL);

        words_to_mpz(dividend_mpz, dividend_row, L);
        words_to_mpz(quotient_mpz, quotient_row, L);
        mpz_divexact_ui(expected_mpz, dividend_mpz, B);

        const int cmp = mpz_cmp(expected_mpz, quotient_mpz);
        if (cmp != 0) {
            if (verbose) {
                gmp_printf("  row %u mismatch\n  dividend=%Zx\n  expected=%Zx\n  actual  =%Zx\n",
                           row_idx, dividend_mpz, expected_mpz, quotient_mpz);
            }
            mpz_clears(dividend_mpz, quotient_mpz, expected_mpz, NULL);
            return false;
        }

        mpz_clears(dividend_mpz, quotient_mpz, expected_mpz, NULL);
    }

    return true;
}

bool verify_against_reference(
    const std::vector<uint32_t> & quotient_ref,
    const std::vector<uint32_t> & quotient_after,
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    bool verbose
) {
    for (uint32_t row_idx = 0; row_idx < N; ++row_idx) {
        const uint32_t * ref_row = quotient_ref.data() + (size_t)row_idx * stride;
        const uint32_t * got_row = quotient_after.data() + (size_t)row_idx * stride;
        for (uint32_t limb = 0; limb < L; ++limb) {
            if (ref_row[limb] != got_row[limb]) {
                if (verbose) {
                    printf("  mismatch at row=%u limb=%u expected=%08x actual=%08x\n",
                           row_idx, limb, ref_row[limb], got_row[limb]);
                }
                return false;
            }
        }
    }
    return true;
}

bool test_configuration(
    uint32_t N,
    uint32_t L,
    uint32_t stride,
    uint32_t B,
    std::mt19937_64 & rng,
    bool verbose
) {
    if (verbose) {
        printf("Testing N=%u L=%u stride=%u B=%08x...\n", N, L, stride, B);
    }

    std::vector<uint32_t> h_dividend((size_t)N * stride);
    std::vector<uint32_t> h_dividend_before;
    std::vector<uint32_t> h_quotient_ref((size_t)N * stride);
    fill_exact_products(h_dividend, h_quotient_ref, N, L, stride, B, rng);
    h_dividend_before = h_dividend;

    uint32_t * d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_dividend.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_A, h_dividend.data(), h_dividend.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    batch_exactdiv_small(d_A, B, N, L, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dividend.data(), d_A, h_dividend.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));

    bool pass = verify_against_reference(h_quotient_ref, h_dividend, N, L, stride, verbose);
    if (pass) {
        pass = verify_rows_with_gmp(h_dividend_before, h_dividend, B, std::min<uint32_t>(N, 4u), L, stride, verbose);
    }
    if (pass && !verify_padding(h_dividend, N, L, stride)) {
        if (verbose) {
            printf("  padding was modified\n");
        }
        pass = false;
    }

    if (verbose) {
        printf("  %s\n", pass ? "PASSED" : "FAILED");
    }
    return pass;
}

void benchmark_configuration(uint32_t L, uint32_t B, uint64_t target_words, std::mt19937_64 & rng) {
    const uint32_t stride = L + 3u;
    uint32_t N = (uint32_t)(target_words / std::max<uint64_t>(L, 1u));
    N = std::max<uint32_t>(N, 1u);

    std::vector<uint32_t> h_dividend((size_t)N * stride);
    fill_exact_products_in_place(h_dividend, N, L, stride, B, rng);

    uint32_t * d_A = nullptr;
    cudaError_t err = cudaMalloc(&d_A, h_dividend.size() * sizeof(uint32_t));
    if (err != cudaSuccess) {
        printf("Benchmark L=%3u N=%8u skipped: cudaMalloc failed: %s\n",
               L, N, cudaGetErrorString(err));
        return;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_dividend.data(), h_dividend.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    batch_exactdiv_small(d_A, B, N, L, stride);
    batch_exactdiv_small(d_A, B, N, L, stride);
    batch_exactdiv_small(d_A, B, N, L, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_A, h_dividend.data(), h_dividend.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 200;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        batch_exactdiv_small(d_A, B, N, L, stride);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    const double avg_ms = milliseconds / iterations;
    const double divs_per_sec_m = ((double)N / (avg_ms * 1e-3)) / 1e6;
    const double bandwidth_gb_s = ((double)N * (double)L * 2.0 * sizeof(uint32_t)) / (avg_ms * 1e-3) / 1e9;
    printf("Benchmark L=%3u N=%8u avg=%7.3f ms div/s=%8.3f M BW=%6.2f GB/s\n",
           L, N, avg_ms, divs_per_sec_m, bandwidth_gb_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
}

}  // namespace

int main() {
    std::mt19937_64 rng(0x6fddc0ffee123456ULL);
    bool pass = true;

    const uint32_t divisors[] = {
        1u,
        3u,
        5u,
        0xffffffffu,
        0xfffffffdu,
        0x13579bdfu
    };

    const struct {
        uint32_t N;
        uint32_t L;
        uint32_t stride;
    } cases[] = {
        {1u, 1u, 4u},
        {7u, 2u, 5u},
        {19u, 7u, 11u},
        {33u, 8u, 12u},
        {31u, 31u, 36u},
        {65u, 32u, 37u},
        {17u, 63u, 67u},
        {96u, 64u, 70u},
        {35u, 95u, 101u},
        {70u, 96u, 103u},
        {37u, 127u, 127u},
        {68u, 128u, 128u}
    };

    for (uint32_t B : divisors) {
        for (const auto & cfg : cases) {
            pass &= test_configuration(cfg.N, cfg.L, cfg.stride, B, rng, !pass);
            if (!pass) {
                break;
            }
        }
        if (!pass) {
            break;
        }
    }

    if (!pass) {
        fprintf(stderr, "batch_exactdiv_small tests FAILED\n");
        return 1;
    }

    printf("All batch_exactdiv_small correctness tests PASSED\n");

    const uint32_t bench_divisor = 0xfffffffdu;
    const uint64_t target_words = 100000000ull;
    const uint32_t bench_lengths[] = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 96u, 128u};
    for (uint32_t L : bench_lengths) {
        benchmark_configuration(L, bench_divisor, target_words, rng);
    }

    return 0;
}
