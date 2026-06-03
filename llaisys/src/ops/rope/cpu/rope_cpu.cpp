#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                float freq = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                size_t base = s * n_heads * head_dim + h * head_dim;
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
        }
    }
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