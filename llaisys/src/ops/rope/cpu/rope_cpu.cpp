#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#ifdef USE_SIMD
#include <immintrin.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;

    // Precompute freq table: theta^(-2*j/head_dim) for j in [0, half_dim)
    float *freq_base = new float[half_dim];
    for (size_t j = 0; j < half_dim; j++) {
        freq_base[j] = 1.0f / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    }

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t s = 0; s < (int64_t)seq_len; s++) {
        for (int64_t h = 0; h < (int64_t)n_heads; h++) {
            float pos = static_cast<float>(pos_ids[s]);
            size_t base = s * n_heads * head_dim + h * head_dim;

#ifdef USE_SIMD
            size_t j = 0;
            for (; j + 4 <= half_dim; j += 4) {
                // Compute cos/sin for 4 elements
                float cos_vals[4], sin_vals[4];
                for (int k = 0; k < 4; k++) {
                    float freq = pos * freq_base[j + k];
                    cos_vals[k] = std::cos(freq);
                    sin_vals[k] = std::sin(freq);
                }

                // Load a = in[base + j..j+3], b = in[base + half_dim + j..j+3]
                float aval[4], bval[4];
                for (int k = 0; k < 4; k++) {
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        aval[k] = llaisys::utils::cast<float>(in[base + j + k]);
                        bval[k] = llaisys::utils::cast<float>(in[base + half_dim + j + k]);
                    } else {
                        aval[k] = static_cast<float>(in[base + j + k]);
                        bval[k] = static_cast<float>(in[base + half_dim + j + k]);
                    }
                }

                // Use SSE 128-bit for 4-element rotation
                __m128 a_lo = _mm_set_ps(aval[3], aval[2], aval[1], aval[0]);
                __m128 b_lo = _mm_set_ps(bval[3], bval[2], bval[1], bval[0]);
                __m128 c_lo = _mm_set_ps(cos_vals[3], cos_vals[2], cos_vals[1], cos_vals[0]);
                __m128 s_lo = _mm_set_ps(sin_vals[3], sin_vals[2], sin_vals[1], sin_vals[0]);

                // out_a = a*cos - b*sin
                __m128 out_a_lo = _mm_sub_ps(_mm_mul_ps(a_lo, c_lo), _mm_mul_ps(b_lo, s_lo));
                // out_b = b*cos + a*sin
                __m128 out_b_lo = _mm_add_ps(_mm_mul_ps(b_lo, c_lo), _mm_mul_ps(a_lo, s_lo));

                float tmp_a[4], tmp_b[4];
                _mm_storeu_ps(tmp_a, out_a_lo);
                _mm_storeu_ps(tmp_b, out_b_lo);
                for (int k = 0; k < 4; k++) {
                    out[base + j + k] = llaisys::utils::cast<T>(tmp_a[k]);
                    out[base + half_dim + j + k] = llaisys::utils::cast<T>(tmp_b[k]);
                }
            }
            // Tail loop
            for (; j < half_dim; j++) {
                float freq = pos * freq_base[j];
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[base + j]);
                    b = llaisys::utils::cast<float>(in[base + half_dim + j]);
                } else {
                    a = static_cast<float>(in[base + j]);
                    b = static_cast<float>(in[base + half_dim + j]);
                }
                out[base + j] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                out[base + half_dim + j] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
#else
            for (size_t j = 0; j < half_dim; j++) {
                float freq = pos * freq_base[j];
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[base + j]);
                    b = llaisys::utils::cast<float>(in[base + half_dim + j]);
                } else {
                    a = static_cast<float>(in[base + j]);
                    b = static_cast<float>(in[base + half_dim + j]);
                }

                out[base + j] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                out[base + half_dim + j] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
#endif
        }
    }

    delete[] freq_base;
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    const int64_t *pids = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pids, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu