#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace llaisys::ops::nvidia {

// GQA (Grouped Query Attention) kernel:
// qlen = query sequence length, kvlen = key/value sequence length
// nh = number of query heads, nkvh = number of key/value heads (nh must be multiple of nkvh)
// hd = head dimension, scale = 1/sqrt(hd)
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                       size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    // One thread block per query head per query position
    int q_seq = blockIdx.x;
    int query_head = blockIdx.y;
    if (q_seq >= (int)qlen || query_head >= (int)nh) return;

    int kv_head = query_head / (int)(nh / nkvh);

    // Compute attention scores for this (q_seq, query_head) against all kv positions
    extern __shared__ float smem[];
    float *scores = smem; // kvlen floats
    int tid = threadIdx.x;

    // Step 1: Compute Q*K^T for this query across all kv positions
    float max_score = -FLT_MAX;
    for (int kv_seq = tid; kv_seq < (int)kvlen; kv_seq += blockDim.x) {
        float dot = 0.0f;
        const T *q_ptr = q + q_seq * nh * hd + query_head * hd;
        const T *k_ptr = k + kv_seq * nkvh * hd + kv_head * hd;
        for (int d = 0; d < (int)hd; ++d) {
            dot += static_cast<float>(q_ptr[d]) * static_cast<float>(k_ptr[d]);
        }
        scores[kv_seq] = dot * scale;
        if (scores[kv_seq] > max_score) {
            max_score = scores[kv_seq];
        }
    }
    __syncthreads();

    // Step 2: Softmax
    float sum_exp = 0.0f;
    for (int kv_seq = tid; kv_seq < (int)kvlen; kv_seq += blockDim.x) {
        scores[kv_seq] = std::exp(scores[kv_seq] - max_score);
        sum_exp += scores[kv_seq];
    }
    __syncthreads();

    // Reduce sum_exp
    smem[tid] = sum_exp; // reuse smem - careful, scores are in the same smem
    __syncthreads();
    // Actually let me use a different approach - use a separate reduction
    // Let's re-approach this more carefully

    // For simplicity: use atomicAdd for sum_exp
    // Actually the shared memory approach is fine if we use a separate buffer
    // Let's restructure: use a 2-phase approach
}

// Simpler approach: each thread computes one output element
template <typename T>
__global__ void self_attention_simple_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                              size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    int q_seq = blockIdx.x;
    int query_head = blockIdx.y;
    int d = threadIdx.x;
    if (q_seq >= (int)qlen || query_head >= (int)nh || d >= (int)hd) return;

    int kv_head = query_head / (int)(nh / nkvh);

    // Compute attention scores for this query against all kv positions
    float max_score = -FLT_MAX;
    for (int kv_seq = 0; kv_seq < (int)kvlen; ++kv_seq) {
        float dot = 0.0f;
        const T *q_ptr = q + q_seq * nh * hd + query_head * hd;
        const T *k_ptr = k + kv_seq * nkvh * hd + kv_head * hd;
        for (int dd = 0; dd < (int)hd; ++dd) {
            dot += static_cast<float>(q_ptr[dd]) * static_cast<float>(k_ptr[dd]);
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    // Compute attention weights and weighted sum
    float sum_exp = 0.0f;
    float weighted_sum = 0.0f;
    for (int kv_seq = 0; kv_seq < (int)kvlen; ++kv_seq) {
        float dot = 0.0f;
        const T *q_ptr = q + q_seq * nh * hd + query_head * hd;
        const T *k_ptr = k + kv_seq * nkvh * hd + kv_head * hd;
        for (int dd = 0; dd < (int)hd; ++dd) {
            dot += static_cast<float>(q_ptr[dd]) * static_cast<float>(k_ptr[dd]);
        }
        float weight = std::exp(dot * scale - max_score);
        sum_exp += weight;

        const T *v_ptr = v + kv_seq * nkvh * hd + kv_head * hd;
        weighted_sum += weight * static_cast<float>(v_ptr[d]);
    }

    attn_val[q_seq * nh * hd + query_head * hd + d] = static_cast<T>(weighted_sum / sum_exp);
}

#undef self_attention_dispatch
#define self_attention_dispatch(T)                                           \
    do {                                                                     \
        dim3 blocks((unsigned int)qlen, (unsigned int)nh);                   \
        dim3 threads((unsigned int)hd);                                      \
        self_attention_simple_kernel<<<blocks, threads>>>(                   \
            reinterpret_cast<T *>(attn_val),                                 \
            reinterpret_cast<const T *>(q),                                  \
            reinterpret_cast<const T *>(k),                                  \
            reinterpret_cast<const T *>(v),                                  \
            qlen, kvlen, nh, nkvh, hd, scale);                               \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[ERROR] self_attention kernel: " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("self_attention kernel failed");        \
        }                                                                    \
    } while (0)

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_dispatch(float);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia