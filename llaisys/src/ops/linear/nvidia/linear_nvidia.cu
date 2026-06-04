#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                               size_t m, size_t k, size_t n) {
    // 2D grid: blockIdx.x = output row, blockIdx.y = output col group
    int row = blockIdx.x;
    int col_start = blockIdx.y * blockDim.x;
    int tid = threadIdx.x;

    if (row >= (int)m) return;

    int col = col_start + tid;
    if (col >= (int)n) return;

    T sum = static_cast<T>(0);
    for (size_t j = 0; j < k; ++j) {
        sum += in[row * k + j] * weight[col * k + j];
    }

    if (bias != nullptr) {
        sum += bias[col];
    }

    out[row * n + col] = sum;
}

#undef linear_dispatch
#define linear_dispatch(T)                                                   \
    do {                                                                     \
        const int threads = 256;                                             \
        const int rows = (int)m;                                             \
        const int cols_blocks = ((int)n + threads - 1) / threads;            \
        dim3 blocks(rows, cols_blocks);                                      \
        linear_kernel<<<blocks, threads>>>(                                  \
            reinterpret_cast<T *>(out),                                      \
            reinterpret_cast<const T *>(in),                                 \
            reinterpret_cast<const T *>(weight),                             \
            reinterpret_cast<const T *>(bias),                               \
            m, k, n);                                                        \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[ERROR] linear kernel: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("linear kernel failed");                \
        }                                                                    \
    } while (0)

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t k, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_dispatch(float);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia