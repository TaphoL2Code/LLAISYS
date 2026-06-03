#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    size_t n_groups = nh / nkvh;

    // Allocate temporary storage for attention scores
    std::vector<float> scores(qlen * kvlen);
    std::vector<float> softmax_scores(qlen * kvlen);
    std::vector<float> attn_out(qlen * nh * hd);

    for (size_t g = 0; g < nkvh; g++) {
        for (size_t gi = 0; gi < n_groups; gi++) {
            size_t h = g * n_groups + gi;

            // Compute Q*K^T for this head
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t kj = 0; kj < kvlen; kj++) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < hd; d++) {
                        float qv, kv;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            qv = llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d]);
                            kv = llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d]);
                        } else {
                            qv = static_cast<float>(q[qi * nh * hd + h * hd + d]);
                            kv = static_cast<float>(k[kj * nkvh * hd + g * hd + d]);
                        }
                        sum += qv * kv;
                    }
                    scores[qi * kvlen + kj] = sum * scale;
                }
            }

            // Causal mask
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t kj = 0; kj < kvlen; kj++) {
                    // Allow positions where kj <= qi + (kvlen - qlen)
                    // For causal mask with prefill: positions where (kvlen - qlen + qi) >= kj
                    if (static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen) + static_cast<int64_t>(qi) < static_cast<int64_t>(kj)) {
                        scores[qi * kvlen + kj] = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            // Softmax
            for (size_t qi = 0; qi < qlen; qi++) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t kj = 0; kj < kvlen; kj++) {
                    max_val = std::max(max_val, scores[qi * kvlen + kj]);
                }
                float sum_exp = 0.0f;
                for (size_t kj = 0; kj < kvlen; kj++) {
                    softmax_scores[qi * kvlen + kj] = std::exp(scores[qi * kvlen + kj] - max_val);
                    sum_exp += softmax_scores[qi * kvlen + kj];
                }
                for (size_t kj = 0; kj < kvlen; kj++) {
                    softmax_scores[qi * kvlen + kj] /= sum_exp;
                }
            }

            // Compute attention output: softmax(A) * V
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t d = 0; d < hd; d++) {
                    float sum = 0.0f;
                    for (size_t kj = 0; kj < kvlen; kj++) {
                        float vv;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            vv = llaisys::utils::cast<float>(v[kj * nkvh * hd + g * hd + d]);
                        } else {
                            vv = static_cast<float>(v[kj * nkvh * hd + g * hd + d]);
                        }
                        sum += softmax_scores[qi * kvlen + kj] * vv;
                    }
                    attn_out[qi * nh * hd + h * hd + d] = sum;
                }
            }
        }
    }

    // Write back to attn_val
    for (size_t i = 0; i < qlen * nh * hd; i++) {
        attn_val[i] = llaisys::utils::cast<T>(attn_out[i]);
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu