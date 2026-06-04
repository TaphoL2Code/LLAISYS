#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#ifdef USE_SIMD
#include <immintrin.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    // out = [m, n], in = [m, k], weight = [n, k], bias = [n]
#ifdef USE_OPENBLAS
    // Use OpenBLAS cblas_sgemm for F32
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)m, (int)n, (int)k,
                    1.0f, in, (int)k, weight, (int)k,
                    0.0f, out, (int)n);
        if (bias != nullptr) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
            for (int64_t i = 0; i < (int64_t)m; i++) {
#ifdef USE_SIMD
                size_t j = 0;
                __m256 bv;
                for (; j + 8 <= n; j += 8) {
                    bv = _mm256_loadu_ps(&bias[j]);
                    __m256 ov = _mm256_loadu_ps(&out[i * n + j]);
                    ov = _mm256_add_ps(ov, bv);
                    _mm256_storeu_ps(&out[i * n + j], ov);
                }
                for (; j < n; j++) {
                    out[i * n + j] += bias[j];
                }
#else
                for (size_t j = 0; j < n; j++) {
                    out[i * n + j] += bias[j];
                }
#endif
            }
        }
        return;
    }
#endif

    // Hand-written GEMM with SIMD + OpenMP
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < (int64_t)m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
#ifdef USE_SIMD
            size_t p = 0;
            __m256 sum8 = _mm256_setzero_ps();
            for (; p + 8 <= k; p += 8) {
                __m256 inv, wv;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    inv = _mm256_set_ps(
                        llaisys::utils::cast<float>(in[i * k + p + 7]),
                        llaisys::utils::cast<float>(in[i * k + p + 6]),
                        llaisys::utils::cast<float>(in[i * k + p + 5]),
                        llaisys::utils::cast<float>(in[i * k + p + 4]),
                        llaisys::utils::cast<float>(in[i * k + p + 3]),
                        llaisys::utils::cast<float>(in[i * k + p + 2]),
                        llaisys::utils::cast<float>(in[i * k + p + 1]),
                        llaisys::utils::cast<float>(in[i * k + p + 0])
                    );
                    wv = _mm256_set_ps(
                        llaisys::utils::cast<float>(weight[j * k + p + 7]),
                        llaisys::utils::cast<float>(weight[j * k + p + 6]),
                        llaisys::utils::cast<float>(weight[j * k + p + 5]),
                        llaisys::utils::cast<float>(weight[j * k + p + 4]),
                        llaisys::utils::cast<float>(weight[j * k + p + 3]),
                        llaisys::utils::cast<float>(weight[j * k + p + 2]),
                        llaisys::utils::cast<float>(weight[j * k + p + 1]),
                        llaisys::utils::cast<float>(weight[j * k + p + 0])
                    );
                } else {
                    inv = _mm256_loadu_ps(reinterpret_cast<const float*>(&in[i * k + p]));
                    wv = _mm256_loadu_ps(reinterpret_cast<const float*>(&weight[j * k + p]));
                }
                sum8 = _mm256_fmadd_ps(inv, wv, sum8);
            }
            // Horizontal sum of 8 floats
            float tmp[8];
            _mm256_storeu_ps(tmp, sum8);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            // Tail loop
            for (; p < k; p++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(in[i * k + p]) * llaisys::utils::cast<float>(weight[j * k + p]);
                } else {
                    sum += static_cast<float>(in[i * k + p]) * static_cast<float>(weight[j * k + p]);
                }
            }
#else
            for (size_t p = 0; p < k; p++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(in[i * k + p]) * llaisys::utils::cast<float>(weight[j * k + p]);
                } else {
                    sum += static_cast<float>(in[i * k + p]) * static_cast<float>(weight[j * k + p]);
                }
            }
#endif
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(bias[j]);
                } else {
                    sum += static_cast<float>(bias[j]);
                }
            }
            out[i * n + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t k, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias), m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias), m, k, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu