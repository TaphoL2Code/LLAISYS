#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#ifdef USE_SIMD
#include <immintrin.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < (int64_t)rows; i++) {
        // Compute mean of squares
        float sum_sq = 0.0f;
#ifdef USE_SIMD
        size_t j = 0;
        __m256 sum8 = _mm256_setzero_ps();
        for (; j + 8 <= cols; j += 8) {
            __m256 inv;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                inv = _mm256_set_ps(
                    llaisys::utils::cast<float>(in[i * cols + j + 7]),
                    llaisys::utils::cast<float>(in[i * cols + j + 6]),
                    llaisys::utils::cast<float>(in[i * cols + j + 5]),
                    llaisys::utils::cast<float>(in[i * cols + j + 4]),
                    llaisys::utils::cast<float>(in[i * cols + j + 3]),
                    llaisys::utils::cast<float>(in[i * cols + j + 2]),
                    llaisys::utils::cast<float>(in[i * cols + j + 1]),
                    llaisys::utils::cast<float>(in[i * cols + j + 0])
                );
            } else {
                inv = _mm256_loadu_ps(reinterpret_cast<const float*>(&in[i * cols + j]));
            }
            sum8 = _mm256_fmadd_ps(inv, inv, sum8);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, sum8);
        sum_sq += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; j < cols; j++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * cols + j]);
            } else {
                val = static_cast<float>(in[i * cols + j]);
            }
            sum_sq += val * val;
        }
#else
        for (size_t j = 0; j < cols; j++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * cols + j]);
            } else {
                val = static_cast<float>(in[i * cols + j]);
            }
            sum_sq += val * val;
        }
#endif
        float rms = std::sqrt(sum_sq / static_cast<float>(cols) + eps);
#ifdef USE_SIMD
        // Use _mm256_rsqrt_ps for approximate reciprocal sqrt then scale
        float inv_rms = 1.0f / rms;
        __m256 inv_rms8 = _mm256_set1_ps(inv_rms);

        j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 inv, wv;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                inv = _mm256_set_ps(
                    llaisys::utils::cast<float>(in[i * cols + j + 7]),
                    llaisys::utils::cast<float>(in[i * cols + j + 6]),
                    llaisys::utils::cast<float>(in[i * cols + j + 5]),
                    llaisys::utils::cast<float>(in[i * cols + j + 4]),
                    llaisys::utils::cast<float>(in[i * cols + j + 3]),
                    llaisys::utils::cast<float>(in[i * cols + j + 2]),
                    llaisys::utils::cast<float>(in[i * cols + j + 1]),
                    llaisys::utils::cast<float>(in[i * cols + j + 0])
                );
                wv = _mm256_set_ps(
                    llaisys::utils::cast<float>(weight[j + 7]),
                    llaisys::utils::cast<float>(weight[j + 6]),
                    llaisys::utils::cast<float>(weight[j + 5]),
                    llaisys::utils::cast<float>(weight[j + 4]),
                    llaisys::utils::cast<float>(weight[j + 3]),
                    llaisys::utils::cast<float>(weight[j + 2]),
                    llaisys::utils::cast<float>(weight[j + 1]),
                    llaisys::utils::cast<float>(weight[j + 0])
                );
            } else {
                inv = _mm256_loadu_ps(reinterpret_cast<const float*>(&in[i * cols + j]));
                wv = _mm256_loadu_ps(reinterpret_cast<const float*>(&weight[j]));
            }
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(inv, wv), inv_rms8);
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float tmp_r[8];
                _mm256_storeu_ps(tmp_r, result);
                for (int r = 0; r < 8; r++) {
                    out[i * cols + j + r] = llaisys::utils::cast<T>(tmp_r[r]);
                }
            } else {
                _mm256_storeu_ps(reinterpret_cast<float*>(&out[i * cols + j]), result);
            }
        }
        for (; j < cols; j++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * cols + j]);
            } else {
                val = static_cast<float>(in[i * cols + j]);
            }
            float w_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                w_val = llaisys::utils::cast<float>(weight[j]);
            } else {
                w_val = static_cast<float>(weight[j]);
            }
            out[i * cols + j] = llaisys::utils::cast<T>(w_val * val * inv_rms);
        }
#else
        // Normalize and apply weight
        for (size_t j = 0; j < cols; j++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * cols + j]);
            } else {
                val = static_cast<float>(in[i * cols + j]);
            }
            float w_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                w_val = llaisys::utils::cast<float>(weight[j]);
            } else {
                w_val = static_cast<float>(weight[j]);
            }
            out[i * cols + j] = llaisys::utils::cast<T>(w_val * val / rms);
        }
#endif
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu