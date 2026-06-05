#ifndef LLAISYS_MODELS_LLAMA_H
#define LLAISYS_MODELS_LLAMA_H

#include "../tensor.h"

__C {
    struct LlaisysLlamaMeta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysLlamaWeights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysLlamaModel;

    __export struct LlaisysLlamaModel *llaisysLlamaModelCreate(const LlaisysLlamaMeta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysLlamaModelDestroy(struct LlaisysLlamaModel * model);

    __export struct LlaisysLlamaWeights *llaisysLlamaModelWeights(struct LlaisysLlamaModel * model);

    __export int64_t llaisysLlamaModelInfer(struct LlaisysLlamaModel * model, int64_t * token_ids, size_t ntoken);

    __export void llaisysLlamaModelForward(struct LlaisysLlamaModel * model, int64_t * token_ids, size_t ntoken, float * logits_out);

    __export void llaisysLlamaModelResetKV(struct LlaisysLlamaModel * model);
}
#endif // LLAISYS_MODELS_LLAMA_H