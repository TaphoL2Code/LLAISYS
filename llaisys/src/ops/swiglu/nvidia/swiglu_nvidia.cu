#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)numel) return;

    float g = static_cast<float>(gate[i]);
    float u = static_cast<float>(up[i]);

    // SwiGLU: gate * sigmoid(gate) * up
    float sig = 1.0f / (1.0f + std::exp(-g));
    out[i] = static_cast<T>(g * sig * u);
}

#undef swiglu_dispatch
#define swiglu_dispatch(T)                                                   \
    do {                                                                     \
        const int threads = 256;                                             \
        const int blocks = ((int)numel + threads - 1) / threads;             \
        swiglu_kernel<<<blocks, threads>>>(                                  \
            reinterpret_cast<T *>(out),                                      \
            reinterpret_cast<const T *>(gate),                               \
            reinterpret_cast<const T *>(up), numel);                         \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[ERROR] swiglu kernel: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("swiglu kernel failed");                \
        }                                                                    \
    } while (0)

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_dispatch(float);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia