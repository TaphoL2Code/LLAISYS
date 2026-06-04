#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                 size_t rows, size_t cols, float eps) {
    int row = blockIdx.x;
    if (row >= (int)rows) return;

    // Step 1: compute RMS in shared memory
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int col = tid;

    float sum_sq = 0.0f;
    for (; col < (int)cols; col += blockDim.x) {
        float val = static_cast<float>(in[row * cols + col]);
        sum_sq += val * val;
    }

    // Block-level reduction for sum_sq
    smem[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        smem[0] = std::sqrt(smem[0] / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    float rms = smem[0];

    // Step 2: normalize
    col = tid;
    for (; col < (int)cols; col += blockDim.x) {
        float normed = static_cast<float>(in[row * cols + col]) / rms;
        float w = static_cast<float>(weight[col]);
        out[row * cols + col] = static_cast<T>(normed * w);
    }
}

#undef rms_norm_dispatch
#define rms_norm_dispatch(T)                                                 \
    do {                                                                     \
        const int threads = 256;                                             \
        const int blocks = (int)rows;                                        \
        size_t smem = threads * sizeof(float);                               \
        rms_norm_kernel<<<blocks, threads, smem>>>(                          \
            reinterpret_cast<T *>(out),                                      \
            reinterpret_cast<const T *>(in),                                 \
            reinterpret_cast<const T *>(weight),                             \
            rows, cols, eps);                                                \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[ERROR] rms_norm kernel: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("rms_norm kernel failed");              \
        }                                                                    \
    } while (0)

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_dispatch(float);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia