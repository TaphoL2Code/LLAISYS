#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef USE_SIMD
#include <immintrin.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    size_t n_groups = nh / nkvh;

    // Allocate temporary storage for attention scores
    std::vector<float> scores(qlen * kvlen);
    std::vector<float> softmax_scores(qlen * kvlen);
    std::vector<float> attn_out(qlen * nh * hd);

    for (size_t g = 0; g < nkvh; g++) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int64_t gi = 0; gi < (int64_t)n_groups; gi++) {
            size_t h = g * n_groups + gi;

            // Compute Q*K^T for this head with SIMD
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t kj = 0; kj < kvlen; kj++) {
                    float sum = 0.0f;
#ifdef USE_SIMD
                    size_t d = 0;
                    __m256 sum8 = _mm256_setzero_ps();
                    for (; d + 8 <= hd; d += 8) {
                        __m256 qv, kv;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            qv = _mm256_set_ps(
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 7]),
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 6]),
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 5]),
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 4]),
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 3]),
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 2]),
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 1]),
                                llaisys::utils::cast<float>(q[qi * nh * hd + h * hd + d + 0])
                            );
                            kv = _mm256_set_ps(
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 7]),
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 6]),
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 5]),
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 4]),
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 3]),
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 2]),
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 1]),
                                llaisys::utils::cast<float>(k[kj * nkvh * hd + g * hd + d + 0])
                            );
                        } else {
                            qv = _mm256_loadu_ps(reinterpret_cast<const float*>(&q[qi * nh * hd + h * hd + d]));
                            kv = _mm256_loadu_ps(reinterpret_cast<const float*>(&k[kj * nkvh * hd + g * hd + d]));
                        }
                        sum8 = _mm256_fmadd_ps(qv, kv, sum8);
                    }
                    float tmp[8];
                    _mm256_storeu_ps(tmp, sum8);
                    sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                    for (; d < hd; d++) {
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
#else
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
#endif
                    scores[qi * kvlen + kj] = sum * scale;
                }
            }

            // Causal mask
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t kj = 0; kj < kvlen; kj++) {
                    if (static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen) + static_cast<int64_t>(qi) < static_cast<int64_t>(kj)) {
                        scores[qi * kvlen + kj] = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            // Softmax with SIMD
            for (size_t qi = 0; qi < qlen; qi++) {
                float max_val = -std::numeric_limits<float>::infinity();
#ifdef USE_SIMD
                {
                    size_t kj = 0;
                    __m256 max8 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                    for (; kj + 8 <= kvlen; kj += 8) {
                        __m256 s = _mm256_loadu_ps(&scores[qi * kvlen + kj]);
                        max8 = _mm256_max_ps(max8, s);
                    }
                    float tmp[8];
                    _mm256_storeu_ps(tmp, max8);
                    max_val = tmp[0];
                    for (int r = 1; r < 8; r++) max_val = std::max(max_val, tmp[r]);
                    for (; kj < kvlen; kj++) {
                        max_val = std::max(max_val, scores[qi * kvlen + kj]);
                    }
                }
#else
                for (size_t kj = 0; kj < kvlen; kj++) {
                    max_val = std::max(max_val, scores[qi * kvlen + kj]);
                }
#endif

                float sum_exp = 0.0f;
#ifdef USE_SIMD
                {
                    size_t kj = 0;
                    __m256 mv = _mm256_set1_ps(max_val);
                    __m256 sum8 = _mm256_setzero_ps();
                    for (; kj + 8 <= kvlen; kj += 8) {
                        __m256 s = _mm256_loadu_ps(&scores[qi * kvlen + kj]);
                        s = _mm256_sub_ps(s, mv);
                        // Compute exp: use scalar approximation since AVX2 has no native exp
                        float tmp[8];
                        _mm256_storeu_ps(tmp, s);
                        for (int r = 0; r < 8; r++) {
                            tmp[r] = std::exp(tmp[r]);
                        }
                        __m256 e = _mm256_loadu_ps(tmp);
                        _mm256_storeu_ps(&softmax_scores[qi * kvlen + kj], e);
                        sum8 = _mm256_add_ps(sum8, e);
                    }
                    float tmp[8];
                    _mm256_storeu_ps(tmp, sum8);
                    sum_exp = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                    for (; kj < kvlen; kj++) {
                        softmax_scores[qi * kvlen + kj] = std::exp(scores[qi * kvlen + kj] - max_val);
                        sum_exp += softmax_scores[qi * kvlen + kj];
                    }
                }
#else
                for (size_t kj = 0; kj < kvlen; kj++) {
                    softmax_scores[qi * kvlen + kj] = std::exp(scores[qi * kvlen + kj] - max_val);
                    sum_exp += softmax_scores[qi * kvlen + kj];
                }
#endif

                // Normalize with SIMD
#ifdef USE_SIMD
                {
                    size_t kj = 0;
                    __m256 inv_sum = _mm256_set1_ps(1.0f / sum_exp);
                    for (; kj + 8 <= kvlen; kj += 8) {
                        __m256 ss = _mm256_loadu_ps(&softmax_scores[qi * kvlen + kj]);
                        ss = _mm256_mul_ps(ss, inv_sum);
                        _mm256_storeu_ps(&softmax_scores[qi * kvlen + kj], ss);
                    }
                    for (; kj < kvlen; kj++) {
                        softmax_scores[qi * kvlen + kj] /= sum_exp;
                    }
                }
#else
                for (size_t kj = 0; kj < kvlen; kj++) {
                    softmax_scores[qi * kvlen + kj] /= sum_exp;
                }
#endif
            }

            // Compute attention output: softmax(A) * V with SIMD
            for (size_t qi = 0; qi < qlen; qi++) {
                for (size_t d = 0; d < hd; d++) {
                    float sum = 0.0f;
#ifdef USE_SIMD
                    size_t kj = 0;
                    __m256 sum8 = _mm256_setzero_ps();
                    __m256 vv;
                    for (; kj + 8 <= kvlen; kj += 8) {
                        __m256 ss = _mm256_loadu_ps(&softmax_scores[qi * kvlen + kj]);
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            vv = _mm256_set_ps(
                                llaisys::utils::cast<float>(v[(kj + 7) * nkvh * hd + g * hd + d]),
                                llaisys::utils::cast<float>(v[(kj + 6) * nkvh * hd + g * hd + d]),
                                llaisys::utils::cast<float>(v[(kj + 5) * nkvh * hd + g * hd + d]),
                                llaisys::utils::cast<float>(v[(kj + 4) * nkvh * hd + g * hd + d]),
                                llaisys::utils::cast<float>(v[(kj + 3) * nkvh * hd + g * hd + d]),
                                llaisys::utils::cast<float>(v[(kj + 2) * nkvh * hd + g * hd + d]),
                                llaisys::utils::cast<float>(v[(kj + 1) * nkvh * hd + g * hd + d]),
                                llaisys::utils::cast<float>(v[(kj + 0) * nkvh * hd + g * hd + d])
                            );
                        } else {
                            // Gather V values from strided memory - must be scalar for non-contiguous
                            float vtmp[8];
                            for (int r = 0; r < 8; r++) {
                                vtmp[r] = static_cast<float>(v[(kj + r) * nkvh * hd + g * hd + d]);
                            }
                            vv = _mm256_loadu_ps(vtmp);
                        }
                        sum8 = _mm256_fmadd_ps(ss, vv, sum8);
                    }
                    float tmp[8];
                    _mm256_storeu_ps(tmp, sum8);
                    sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                    for (; kj < kvlen; kj++) {
                        float vv;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            vv = llaisys::utils::cast<float>(v[kj * nkvh * hd + g * hd + d]);
                        } else {
                            vv = static_cast<float>(v[kj * nkvh * hd + g * hd + d]);
                        }
                        sum += softmax_scores[qi * kvlen + kj] * vv;
                    }
#else
                    for (size_t kj = 0; kj < kvlen; kj++) {
                        float vv;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            vv = llaisys::utils::cast<float>(v[kj * nkvh * hd + g * hd + d]);
                        } else {
                            vv = static_cast<float>(v[kj * nkvh * hd + g * hd + d]);
                        }
                        sum += softmax_scores[qi * kvlen + kj] * vv;
                    }
#endif
                    attn_out[qi * nh * hd + h * hd + d] = sum;
                }
            }
        }
    }

    // Write back to attn_val
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < (int64_t)(qlen * nh * hd); i++) {
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