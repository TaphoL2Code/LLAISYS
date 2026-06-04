#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#ifdef USE_SIMD
#include <immintrin.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Fast sigmoid approximation using SIMD: 1/(1+exp(-x)) is the standard sigmoid
// For SIMD we use the SSE/AVX polynomial approximation
#ifdef USE_SIMD
static inline __m256 _mm256_sigmoid_ps(__m256 x) {
    // Sigmoid approximation: 0.5 + 0.5 * tanh(x/2)
    // tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
    // But simpler: use exp approximation or just compute directly
    // For precision, we compute exp(-x) then 1/(1+exp(-x))
    // exp(-x) approximation using AVX2: this is complex, so we use a polynomial
    // Using: sigmoid(x) ≈ 1/(1+exp(-x))
    // Simple AVX2 exp: not available natively, use scalar fallback or approximate
    // For now, use a cubic polynomial sigmoid approximation:
    // sigmoid(x) ≈ 0.5 + x * (0.25 - 0.020833 * x^2) for |x| < 5
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 p1 = _mm256_set1_ps(0.25f);
    __m256 p3 = _mm256_set1_ps(0.020833f);
    __m256 term = _mm256_fnmadd_ps(x2, p3, p1); // 0.25 - 0.020833*x^2
    return _mm256_fmadd_ps(x, term, half);       // 0.5 + x * term
}
#endif

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
#ifdef USE_SIMD
    int64_t i = 0;
    for (; i + 8 <= (int64_t)numel; i += 8) {
        __m256 g, u;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            g = _mm256_set_ps(
                llaisys::utils::cast<float>(gate[i + 7]),
                llaisys::utils::cast<float>(gate[i + 6]),
                llaisys::utils::cast<float>(gate[i + 5]),
                llaisys::utils::cast<float>(gate[i + 4]),
                llaisys::utils::cast<float>(gate[i + 3]),
                llaisys::utils::cast<float>(gate[i + 2]),
                llaisys::utils::cast<float>(gate[i + 1]),
                llaisys::utils::cast<float>(gate[i + 0])
            );
            u = _mm256_set_ps(
                llaisys::utils::cast<float>(up[i + 7]),
                llaisys::utils::cast<float>(up[i + 6]),
                llaisys::utils::cast<float>(up[i + 5]),
                llaisys::utils::cast<float>(up[i + 4]),
                llaisys::utils::cast<float>(up[i + 3]),
                llaisys::utils::cast<float>(up[i + 2]),
                llaisys::utils::cast<float>(up[i + 1]),
                llaisys::utils::cast<float>(up[i + 0])
            );
        } else {
            g = _mm256_loadu_ps(reinterpret_cast<const float*>(&gate[i]));
            u = _mm256_loadu_ps(reinterpret_cast<const float*>(&up[i]));
        }
        __m256 sig = _mm256_sigmoid_ps(g);
        __m256 result = _mm256_mul_ps(u, _mm256_mul_ps(g, sig));
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float tmp[8];
            _mm256_storeu_ps(tmp, result);
            for (int k = 0; k < 8; k++) {
                out[i + k] = llaisys::utils::cast<T>(tmp[k]);
            }
        } else {
            _mm256_storeu_ps(reinterpret_cast<float*>(&out[i]), result);
        }
    }
    // Tail loop
    for (; i < (int64_t)numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float g = llaisys::utils::cast<float>(gate[i]);
            float u = llaisys::utils::cast<float>(up[i]);
            float sigmoid = 1.0f / (1.0f + std::exp(-g));
            out[i] = llaisys::utils::cast<T>(u * g * sigmoid);
        } else {
            float sigmoid = 1.0f / (1.0f + std::exp(-static_cast<float>(gate[i])));
            out[i] = up[i] * gate[i] * sigmoid;
        }
    }
#else
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < (int64_t)numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float g = llaisys::utils::cast<float>(gate[i]);
            float u = llaisys::utils::cast<float>(up[i]);
            float sigmoid = 1.0f / (1.0f + std::exp(-g));
            out[i] = llaisys::utils::cast<T>(u * g * sigmoid);
        } else {
            float sigmoid = 1.0f / (1.0f + std::exp(-static_cast<float>(gate[i])));
            out[i] = up[i] * gate[i] * sigmoid;
        }
    }
#endif
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu