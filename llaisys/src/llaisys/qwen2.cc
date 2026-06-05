#include "llaisys/models/qwen2.h"

#include "../models/qwen2/qwen2.hpp"

#include <new>

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        int device_id = (ndevice > 0) ? device_ids[0] : 0;
        auto *model = new (std::nothrow) llaisys::models::Qwen2Model(*meta, device, device_id);
        return reinterpret_cast<struct LlaisysQwen2Model *>(model);
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete reinterpret_cast<llaisys::models::Qwen2Model *>(model);
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        auto *m = reinterpret_cast<llaisys::models::Qwen2Model *>(model);
        return m->weights();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        auto *m = reinterpret_cast<llaisys::models::Qwen2Model *>(model);
        return m->infer(token_ids, ntoken);
    }

    void llaisysQwen2ModelForward(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, float *logits_out) {
        auto *m = reinterpret_cast<llaisys::models::Qwen2Model *>(model);
        m->forward(token_ids, ntoken, logits_out);
    }

    void llaisysQwen2ModelResetKV(struct LlaisysQwen2Model *model) {
        auto *m = reinterpret_cast<llaisys::models::Qwen2Model *>(model);
        m->reset_kv_cache();
    }
}