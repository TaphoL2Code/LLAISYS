#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    // out = [m, n], in = [m, k], weight = [n, k], bias = [n]
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(in[i * k + p]) * llaisys::utils::cast<float>(weight[j * k + p]);
                } else {
                    sum += static_cast<float>(in[i * k + p]) * static_cast<float>(weight[j * k + p]);
                }
            }
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(bias[j]);
                } else {
                    sum += static_cast<float>(bias[j]);
                }
            }
            out[i * n + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t k, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias), m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias), m, k, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu