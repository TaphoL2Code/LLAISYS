#pragma once

#include "llaisys/models/llama.h"

#include "../../llaisys/llaisys_tensor.hpp"
#include "../../tensor/tensor.hpp"
#include "../../core/llaisys_core.hpp"

#include <vector>
#include <memory>

namespace llaisys::models {

class LlamaModel {
public:
    LlamaModel(const LlaisysLlamaMeta &meta, llaisysDeviceType_t device, int device_id);
    ~LlamaModel();

    LlaisysLlamaWeights *weights();
    int64_t infer(const int64_t *token_ids, size_t ntoken);

    // Forward pass returning logits (for sampling)
    void forward(const int64_t *token_ids, size_t ntoken, float *logits_out);

    // KV-cache management for multi-turn conversations
    void reset_kv_cache();
    size_t get_kv_cache_length() const { return _cur_seq_len; }

    const LlaisysLlamaMeta &meta() const { return _meta; }

private:
    LlaisysLlamaMeta _meta;
    llaisysDeviceType_t _device;
    int _device_id;

    LlaisysLlamaWeights _weights;

    // KV Cache: per-layer K and V caches
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    size_t _cur_seq_len = 0;

    // Intermediate tensors for forward pass
    tensor_t _create_tensor(const std::vector<size_t> &shape) const;

    void _transformer_block(
        size_t layer,
        tensor_t x,          // [ntoken, hs]
        tensor_t pos_ids,    // [ntoken]
        tensor_t attn_out,   // [ntoken, hs] output of attention
        tensor_t ffn_out     // [ntoken, hs] output of FFN
    );

    void _copy_logits_to_host(tensor_t logits, float *logits_out, size_t voc);

    void _prefill(const int64_t *token_ids, size_t ntoken);
    int64_t _decode(int64_t token_id);
};

} // namespace llaisys::models