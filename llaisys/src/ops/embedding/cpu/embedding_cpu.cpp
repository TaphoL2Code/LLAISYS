#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t idx_len, size_t embed_dim) {
    for (size_t i = 0; i < idx_len; i++) {
        int64_t row = index[i];
        const T *src = weight + row * embed_dim;
        T *dst = out + i * embed_dim;
        for (size_t j = 0; j < embed_dim; j++) {
            dst[j] = src[j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type,
               size_t idx_len, size_t embed_dim, size_t /*vocab_size*/) {
    const int64_t *idx = reinterpret_cast<const int64_t *>(index);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx, reinterpret_cast<const float *>(weight), idx_len, embed_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx, reinterpret_cast<const llaisys::bf16_t *>(weight), idx_len, embed_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), idx, reinterpret_cast<const llaisys::fp16_t *>(weight), idx_len, embed_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu