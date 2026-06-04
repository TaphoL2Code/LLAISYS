#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                             size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t s = blockIdx.x;
    size_t h = blockIdx.y;
    if (s >= seq_len || h >= n_heads) return;

    size_t half_dim = head_dim / 2;
    float pos = static_cast<float>(pos_ids[s]);

    size_t base_offset = s * n_heads * head_dim + h * head_dim;

    for (size_t j = threadIdx.x; j < half_dim; j += blockDim.x) {
        float freq = 1.0f / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
        float angle = pos * freq;
        float cos_val = std::cos(angle);
        float sin_val = std::sin(angle);

        float x0 = static_cast<float>(in[base_offset + j]);
        float x1 = static_cast<float>(in[base_offset + j + half_dim]);

        out[base_offset + j] = static_cast<T>(x0 * cos_val - x1 * sin_val);
        out[base_offset + j + half_dim] = static_cast<T>(x0 * sin_val + x1 * cos_val);
    }
}

#undef rope_dispatch
#define rope_dispatch(T)                                                     \
    do {                                                                     \
        dim3 blocks((unsigned int)seq_len, (unsigned int)n_heads);           \
        const int threads = 128;                                             \
        rope_kernel<<<blocks, threads>>>(                                    \
            reinterpret_cast<T *>(out),                                      \
            reinterpret_cast<const T *>(in),                                 \
            reinterpret_cast<const int64_t *>(pos_ids),                      \
            seq_len, n_heads, head_dim, theta);                              \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[ERROR] rope kernel: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("rope kernel failed");                  \
        }                                                                    \
    } while (0)

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_dispatch(float);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia