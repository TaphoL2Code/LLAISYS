#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                  size_t idx_len, size_t embed_dim, size_t vocab_size) {
    int64_t i = blockIdx.x; // which token
    if (i >= (int64_t)idx_len) return;

    int64_t token_id = index[i];
    if (token_id < 0 || token_id >= (int64_t)vocab_size) {
        // Out-of-range: set to zero
        for (int64_t j = threadIdx.x; j < (int64_t)embed_dim; j += blockDim.x) {
            out[i * embed_dim + j] = static_cast<T>(0);
        }
        return;
    }

    for (int64_t j = threadIdx.x; j < (int64_t)embed_dim; j += blockDim.x) {
        out[i * embed_dim + j] = weight[token_id * embed_dim + j];
    }
}

#undef embedding_dispatch
#define embedding_dispatch(T)                                                \
    do {                                                                     \
        const int threads = 256;                                             \
        const int blocks = (int)idx_len;                                     \
        embedding_kernel<<<blocks, threads>>>(                               \
            reinterpret_cast<T *>(out),                                      \
            reinterpret_cast<const int64_t *>(index),                        \
            reinterpret_cast<const T *>(weight),                             \
            idx_len, embed_dim, vocab_size);                                 \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[ERROR] embedding kernel: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("embedding kernel failed");             \
        }                                                                    \
    } while (0)

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type,
               size_t idx_len, size_t embed_dim, size_t vocab_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_dispatch(float);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia