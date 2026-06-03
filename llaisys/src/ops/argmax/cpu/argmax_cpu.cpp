#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    T best_val = vals[0];
    size_t best_idx = 0;
    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float cur = llaisys::utils::cast<float>(vals[i]);
            float best = llaisys::utils::cast<float>(best_val);
            if (cur > best) {
                best_val = vals[i];
                best_idx = i;
            }
        } else {
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_idx = i;
            }
        }
    }
    max_val[0] = best_val;
    max_idx[0] = static_cast<int64_t>(best_idx);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu