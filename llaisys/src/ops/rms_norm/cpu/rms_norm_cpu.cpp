#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t i = 0; i < rows; i++) {
        // Compute mean of squares
        float sum_sq = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * cols + j]);
            } else {
                val = static_cast<float>(in[i * cols + j]);
            }
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(cols) + eps);

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