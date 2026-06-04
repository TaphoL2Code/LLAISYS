#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace llaisys::device::nvidia {

namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

void setDevice(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaSetDevice(" << device_id << ") failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaSetDevice failed");
    }
}

void deviceSynchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaDeviceSynchronize failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaDeviceSynchronize failed");
    }
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamCreate failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaStreamCreate failed");
    }
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        cudaStreamDestroy((cudaStream_t)stream);
    }
}

void streamSynchronize(llaisysStream_t stream) {
    if (stream != nullptr) {
        cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                      << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("cudaStreamSynchronize failed");
        }
    }
}

void *mallocDevice(size_t size) {
    if (size == 0) return nullptr;
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc(" << size << ") failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMalloc failed");
    }
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void *mallocHost(size_t size) {
    if (size == 0) return nullptr;
    void *ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMallocHost(" << size << ") failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMallocHost failed");
    }
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

static cudaMemcpyKind mapMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: return cudaMemcpyDefault;
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaError_t err = cudaMemcpy(dst, src, size, mapMemcpyKind(kind));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpy(" << size << " bytes) failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpy failed");
    }
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, mapMemcpyKind(kind), (cudaStream_t)stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpyAsync(" << size << " bytes) failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpyAsync failed");
    }
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync
};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia