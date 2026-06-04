#include "argmax_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

namespace llaisys::ops::nvidia {

// Phase 1: Each block reduces its portion to find local max
template <typename T>
__global__ void argmax_block_reduce_kernel(
    int64_t *block_max_idx, T *block_max_val,
    const T *vals, size_t numel) {

    extern __shared__ char smem[];
    T *smax_val = reinterpret_cast<T *>(smem);
    int64_t *smax_idx = reinterpret_cast<int64_t *>(smax_val + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < (int64_t)numel) {
        smax_val[tid] = vals[gid];
        smax_idx[tid] = gid;
    } else {
        smax_val[tid] = -FLT_MAX;
        smax_idx[tid] = -1;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax_val[tid + s] > smax_val[tid]) {
                smax_val[tid] = smax_val[tid + s];
                smax_idx[tid] = smax_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max_val[blockIdx.x] = smax_val[0];
        block_max_idx[blockIdx.x] = smax_idx[0];
    }
}

#undef argmax_dispatch
#define argmax_dispatch(T)                                                        \
    do {                                                                          \
        const int threads = 256;                                                  \
        const int blocks = ((int)numel + threads - 1) / threads;                  \
        size_t smem = threads * (sizeof(T) + sizeof(int64_t));                    \
                                                                                  \
        /* Allocate per-block output buffers on device */                         \
        int64_t *d_block_max_idx;                                                 \
        T *d_block_max_val;                                                       \
        cudaError_t err = cudaMalloc(&d_block_max_idx, blocks * sizeof(int64_t)); \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error("argmax: cudaMalloc failed for block_max_idx"); \
        }                                                                         \
        err = cudaMalloc(&d_block_max_val, blocks * sizeof(T));                   \
        if (err != cudaSuccess) {                                                 \
            cudaFree(d_block_max_idx);                                            \
            throw std::runtime_error("argmax: cudaMalloc failed for block_max_val"); \
        }                                                                         \
                                                                                  \
        /* Launch kernel */                                                       \
        argmax_block_reduce_kernel<<<blocks, threads, smem>>>(                    \
            d_block_max_idx, d_block_max_val,                                     \
            reinterpret_cast<const T *>(vals), numel);                            \
        err = cudaGetLastError();                                                 \
        if (err != cudaSuccess) {                                                 \
            cudaFree(d_block_max_idx);                                            \
            cudaFree(d_block_max_val);                                            \
            std::cerr << "[ERROR] argmax kernel: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("argmax kernel failed");                     \
        }                                                                         \
        cudaDeviceSynchronize();                                                  \
                                                                                  \
        /* Copy block results to host and do final reduction */                   \
        std::vector<int64_t> h_block_max_idx(blocks);                             \
        std::vector<T> h_block_max_val(blocks);                                   \
        cudaMemcpy(h_block_max_idx.data(), d_block_max_idx,                       \
                   blocks * sizeof(int64_t), cudaMemcpyDeviceToHost);             \
        cudaMemcpy(h_block_max_val.data(), d_block_max_val,                       \
                   blocks * sizeof(T), cudaMemcpyDeviceToHost);                   \
        cudaFree(d_block_max_idx);                                                \
        cudaFree(d_block_max_val);                                                \
                                                                                  \
        /* Final host-side reduction */                                           \
        int64_t best_idx = -1;                                                    \
        T best_val = -FLT_MAX;                                                    \
        for (int i = 0; i < blocks; i++) {                                        \
            if (h_block_max_val[i] > best_val) {                                  \
                best_val = h_block_max_val[i];                                    \
                best_idx = h_block_max_idx[i];                                    \
            }                                                                     \
        }                                                                         \
        if (best_idx >= 0) {                                                      \
            cudaMemcpy(max_idx, &best_idx, sizeof(int64_t), cudaMemcpyHostToDevice);   \
            cudaMemcpy(max_val, &best_val, sizeof(T), cudaMemcpyHostToDevice);          \
        }                                                                         \
    } while (0)

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_dispatch(float);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia