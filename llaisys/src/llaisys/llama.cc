#include "llaisys/models/llama.h"

#include "../models/llama/llama.hpp"

#include <new>

__C {
    struct LlaisysLlamaModel *llaisysLlamaModelCreate(
        const LlaisysLlamaMeta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        int device_id = (ndevice > 0) ? device_ids[0] : 0;
        auto *model = new (std::nothrow) llaisys::models::LlamaModel(*meta, device, device_id);
        return reinterpret_cast<struct LlaisysLlamaModel *>(model);
    }

    void llaisysLlamaModelDestroy(struct LlaisysLlamaModel *model) {
        delete reinterpret_cast<llaisys::models::LlamaModel *>(model);
    }

    struct LlaisysLlamaWeights *llaisysLlamaModelWeights(struct LlaisysLlamaModel *model) {
        auto *m = reinterpret_cast<llaisys::models::LlamaModel *>(model);
        return m->weights();
    }

    int64_t llaisysLlamaModelInfer(struct LlaisysLlamaModel *model, int64_t *token_ids, size_t ntoken) {
        auto *m = reinterpret_cast<llaisys::models::LlamaModel *>(model);
        return m->infer(token_ids, ntoken);
    }

    void llaisysLlamaModelForward(struct LlaisysLlamaModel *model, int64_t *token_ids, size_t ntoken, float *logits_out) {
        auto *m = reinterpret_cast<llaisys::models::LlamaModel *>(model);
        m->forward(token_ids, ntoken, logits_out);
    }

    void llaisysLlamaModelResetKV(struct LlaisysLlamaModel *model) {
        auto *m = reinterpret_cast<llaisys::models::LlamaModel *>(model);
        m->reset_kv_cache();
    }
}