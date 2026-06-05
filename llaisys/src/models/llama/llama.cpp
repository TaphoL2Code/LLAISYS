#include "llama.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cstring>
#include <cmath>

namespace llaisys::models {

LlamaModel::LlamaModel(const LlaisysLlamaMeta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id), _cur_seq_len(0) {
    // Initialize weights struct to zero
    std::memset(&_weights, 0, sizeof(_weights));

    // Allocate per-layer weight arrays (LLaMA has no bias, so fewer arrays)
    _weights.attn_norm_w = new llaisysTensor_t[meta.nlayer]();
    _weights.attn_q_w    = new llaisysTensor_t[meta.nlayer]();
    _weights.attn_k_w    = new llaisysTensor_t[meta.nlayer]();
    _weights.attn_v_w    = new llaisysTensor_t[meta.nlayer]();
    _weights.attn_o_w    = new llaisysTensor_t[meta.nlayer]();
    _weights.mlp_norm_w  = new llaisysTensor_t[meta.nlayer]();
    _weights.mlp_gate_w  = new llaisysTensor_t[meta.nlayer]();
    _weights.mlp_up_w    = new llaisysTensor_t[meta.nlayer]();
    _weights.mlp_down_w  = new llaisysTensor_t[meta.nlayer]();

    // Allocate KV Cache tensors
    for (size_t i = 0; i < meta.nlayer; i++) {
        _k_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device, device_id));
        _v_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device, device_id));
    }
}

LlamaModel::~LlamaModel() {
    delete[] _weights.attn_norm_w;
    delete[] _weights.attn_q_w;
    delete[] _weights.attn_k_w;
    delete[] _weights.attn_v_w;
    delete[] _weights.attn_o_w;
    delete[] _weights.mlp_norm_w;
    delete[] _weights.mlp_gate_w;
    delete[] _weights.mlp_up_w;
    delete[] _weights.mlp_down_w;
}

LlaisysLlamaWeights *LlamaModel::weights() {
    return &_weights;
}

tensor_t LlamaModel::_create_tensor(const std::vector<size_t> &shape) const {
    return Tensor::create(shape, _meta.dtype, _device, _device_id);
}

// =========================================================================
// Helper: copy data from src tensor into a slice of cache
// =========================================================================
static void _copy_to_cache_slice(tensor_t cache, tensor_t src, size_t start, size_t ntoken) {
    auto cache_slice = cache->slice(0, start, start + ntoken);
    auto src_contig = src->contiguous();
    size_t nbytes = ntoken * src->shape()[1] * src->shape()[2] * src->elementSize();
    std::memcpy(cache_slice->data(), src_contig->data(), nbytes);
}

// =========================================================================
// Single Transformer Block forward pass
// LLaMA differs from Qwen2: no bias in Q/K/V/O projections
// =========================================================================
void LlamaModel::_transformer_block(
    size_t layer,
    tensor_t x,          // [ntoken, hs]
    tensor_t pos_ids,    // [ntoken]
    tensor_t attn_out,   // [ntoken, hs] output
    tensor_t ffn_out     // [ntoken, hs] output
) {
    size_t ntoken = x->shape()[0];
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;

    // --- RMS Norm (pre-attention) ---
    auto norm_x = _create_tensor({ntoken, hs});
    ops::rms_norm(norm_x, x, _weights.attn_norm_w[layer]->tensor, _meta.epsilon);

    // --- Q/K/V projections (LLaMA: no bias) ---
    auto q_2d = _create_tensor({ntoken, nh * dh});
    ops::linear(q_2d, norm_x, _weights.attn_q_w[layer]->tensor, nullptr);
    auto q = q_2d->view({ntoken, nh, dh});

    auto k_2d = _create_tensor({ntoken, nkvh * dh});
    ops::linear(k_2d, norm_x, _weights.attn_k_w[layer]->tensor, nullptr);
    auto k = k_2d->view({ntoken, nkvh, dh});

    auto v_2d = _create_tensor({ntoken, nkvh * dh});
    ops::linear(v_2d, norm_x, _weights.attn_v_w[layer]->tensor, nullptr);
    auto v = v_2d->view({ntoken, nkvh, dh});

    // --- RoPE on Q and K ---
    auto q_rope = _create_tensor({ntoken, nh, dh});
    auto k_rope = _create_tensor({ntoken, nkvh, dh});
    ops::rope(q_rope, q, pos_ids, _meta.theta);
    ops::rope(k_rope, k, pos_ids, _meta.theta);

    // --- KV Cache: store new K/V ---
    size_t cache_start = _cur_seq_len;
    _copy_to_cache_slice(_k_cache[layer], k_rope, cache_start, ntoken);
    _copy_to_cache_slice(_v_cache[layer], v, cache_start, ntoken);

    size_t total_seq_len = cache_start + ntoken;

    // --- Self-Attention ---
    auto attn = _create_tensor({ntoken, nh, dh});
    auto k_full = _k_cache[layer]->slice(0, 0, total_seq_len);
    auto v_full = _v_cache[layer]->slice(0, 0, total_seq_len);
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    ops::self_attention(attn, q_rope, k_full, v_full, scale);

    // --- O projection (LLaMA: no bias) ---
    auto attn_2d = attn->view({ntoken, nh * dh});
    ops::linear(attn_out, attn_2d, _weights.attn_o_w[layer]->tensor, nullptr);

    // --- Residual: x + attn_out → attn_out ---
    ops::add(attn_out, x, attn_out);

    // --- RMS Norm (pre-FFN) ---
    auto norm_attn = _create_tensor({ntoken, hs});
    ops::rms_norm(norm_attn, attn_out, _weights.mlp_norm_w[layer]->tensor, _meta.epsilon);

    // --- FFN: Gate + Up → SwiGLU → Down ---
    auto gate = _create_tensor({ntoken, _meta.di});
    auto up   = _create_tensor({ntoken, _meta.di});
    ops::linear(gate, norm_attn, _weights.mlp_gate_w[layer]->tensor, nullptr);
    ops::linear(up,   norm_attn, _weights.mlp_up_w[layer]->tensor,   nullptr);

    auto swiglu_out = _create_tensor({ntoken, _meta.di});
    ops::swiglu(swiglu_out, gate, up);

    auto ffn = _create_tensor({ntoken, hs});
    ops::linear(ffn, swiglu_out, _weights.mlp_down_w[layer]->tensor, nullptr);

    // --- Residual: attn_out + ffn → ffn_out ---
    ops::add(ffn_out, attn_out, ffn);
}

// =========================================================================
// Prefill + Decode: process input tokens and return next token
// =========================================================================
int64_t LlamaModel::infer(const int64_t *token_ids, size_t ntoken) {
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t dh = _meta.dh;
    size_t voc = _meta.voc;

    if (_cur_seq_len == 0) {
        // === First call: prefill all input tokens ===
        auto idx = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
        std::memcpy(idx->data(), token_ids, ntoken * sizeof(int64_t));

        auto x = _create_tensor({ntoken, hs});
        ops::embedding(x, idx, _weights.in_embed->tensor);

        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
        auto *pos_data = reinterpret_cast<int64_t *>(pos_ids->data());
        for (size_t i = 0; i < ntoken; i++) {
            pos_data[i] = static_cast<int64_t>(i);
        }

        auto attn_out = _create_tensor({ntoken, hs});
        auto ffn_out  = _create_tensor({ntoken, hs});
        for (size_t layer = 0; layer < _meta.nlayer; layer++) {
            _transformer_block(layer, x, pos_ids, attn_out, ffn_out);
            x = ffn_out;
        }
        _cur_seq_len = ntoken;

        auto last_hidden = x->slice(0, ntoken - 1, ntoken);
        auto final_norm = _create_tensor({1, hs});
        ops::rms_norm(final_norm, last_hidden, _weights.out_norm_w->tensor, _meta.epsilon);

        auto logits = _create_tensor({1, voc});
        ops::linear(logits, final_norm, _weights.out_embed->tensor, nullptr);

        auto max_val = _create_tensor({1});
        auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
        ops::argmax(max_idx, max_val, logits);

        return reinterpret_cast<const int64_t *>(max_idx->data())[0];
    } else {
        // === Subsequent call: decode one new token ===
        int64_t new_token = token_ids[_cur_seq_len];

        auto idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
        reinterpret_cast<int64_t *>(idx->data())[0] = new_token;

        auto x = _create_tensor({1, hs});
        ops::embedding(x, idx, _weights.in_embed->tensor);

        auto pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
        reinterpret_cast<int64_t *>(pos_ids->data())[0] = static_cast<int64_t>(_cur_seq_len);

        auto attn_out = _create_tensor({1, hs});
        auto ffn_out  = _create_tensor({1, hs});
        for (size_t layer = 0; layer < _meta.nlayer; layer++) {
            _transformer_block(layer, x, pos_ids, attn_out, ffn_out);
            x = ffn_out;
        }
        _cur_seq_len++;

        auto final_norm = _create_tensor({1, hs});
        ops::rms_norm(final_norm, x, _weights.out_norm_w->tensor, _meta.epsilon);

        auto logits = _create_tensor({1, voc});
        ops::linear(logits, final_norm, _weights.out_embed->tensor, nullptr);

        auto max_val = _create_tensor({1});
        auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
        ops::argmax(max_idx, max_val, logits);

        return reinterpret_cast<const int64_t *>(max_idx->data())[0];
    }
}

// =========================================================================
// Forward pass returning raw logits (for sampling)
// =========================================================================
void LlamaModel::forward(const int64_t *token_ids, size_t ntoken, float *logits_out) {
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t dh = _meta.dh;
    size_t voc = _meta.voc;

    if (_cur_seq_len == 0) {
        auto idx = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
        std::memcpy(idx->data(), token_ids, ntoken * sizeof(int64_t));

        auto x = _create_tensor({ntoken, hs});
        ops::embedding(x, idx, _weights.in_embed->tensor);

        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
        auto *pos_data = reinterpret_cast<int64_t *>(pos_ids->data());
        for (size_t i = 0; i < ntoken; i++) {
            pos_data[i] = static_cast<int64_t>(i);
        }

        auto attn_out = _create_tensor({ntoken, hs});
        auto ffn_out  = _create_tensor({ntoken, hs});
        for (size_t layer = 0; layer < _meta.nlayer; layer++) {
            _transformer_block(layer, x, pos_ids, attn_out, ffn_out);
            x = ffn_out;
        }
        _cur_seq_len = ntoken;

        auto last_hidden = x->slice(0, ntoken - 1, ntoken);
        auto final_norm = _create_tensor({1, hs});
        ops::rms_norm(final_norm, last_hidden, _weights.out_norm_w->tensor, _meta.epsilon);

        auto logits = _create_tensor({1, voc});
        ops::linear(logits, final_norm, _weights.out_embed->tensor, nullptr);

        _copy_logits_to_host(logits, logits_out, voc);
    } else {
        int64_t new_token = token_ids[_cur_seq_len];

        auto idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
        reinterpret_cast<int64_t *>(idx->data())[0] = new_token;

        auto x = _create_tensor({1, hs});
        ops::embedding(x, idx, _weights.in_embed->tensor);

        auto pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
        reinterpret_cast<int64_t *>(pos_ids->data())[0] = static_cast<int64_t>(_cur_seq_len);

        auto attn_out = _create_tensor({1, hs});
        auto ffn_out  = _create_tensor({1, hs});
        for (size_t layer = 0; layer < _meta.nlayer; layer++) {
            _transformer_block(layer, x, pos_ids, attn_out, ffn_out);
            x = ffn_out;
        }
        _cur_seq_len++;

        auto final_norm = _create_tensor({1, hs});
        ops::rms_norm(final_norm, x, _weights.out_norm_w->tensor, _meta.epsilon);

        auto logits = _create_tensor({1, voc});
        ops::linear(logits, final_norm, _weights.out_embed->tensor, nullptr);

        _copy_logits_to_host(logits, logits_out, voc);
    }
}

// =========================================================================
// Copy logits from device tensor to host buffer
// =========================================================================
void LlamaModel::_copy_logits_to_host(tensor_t logits, float *logits_out, size_t voc) {
    size_t logits_size = voc * sizeof(float);
    if (_device == LLAISYS_DEVICE_CPU) {
        std::memcpy(logits_out, logits->data(), logits_size);
    } else {
        auto *api = llaisys::core::context().runtime().api();
        if (api && api->memcpy_sync) {
            api->memcpy_sync(logits_out, logits->data(), logits_size, LLAISYS_MEMCPY_D2H);
        }
    }
}

// =========================================================================
// Reset KV-cache for new conversation
// =========================================================================
void LlamaModel::reset_kv_cache() {
    _cur_seq_len = 0;
}

} // namespace llaisys::models