from typing import Sequence, List, Dict
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.llama import LlaisysLlamaMeta
from ..tensor import Tensor
from ..sampler import sample

from pathlib import Path
import json
import safetensors
import numpy as np
from ctypes import (
    c_void_p,
    c_size_t,
    c_int,
    c_int64,
    c_float,
    POINTER,
    cast,
    pointer,
    byref,
)


def _dtype_name_to_llaisys(dtype_name: str) -> int:
    """Convert torch dtype name to llaisysDataType_t."""
    dtype_name = str(dtype_name).lower()
    if "bfloat16" in dtype_name:
        return DataType.BF16
    if "float16" in dtype_name:
        return DataType.F16
    if "float32" in dtype_name:
        return DataType.F32
    return DataType.F16


class Llama:
    """LLaMA model implementation.

    LLaMA architecture is similar to Qwen2 but without bias in attention
    projections (Q, K, V, O are all bias-free).
    """

    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # --- Read config.json ---
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        hs = config["hidden_size"]
        nlayer = config["num_hidden_layers"]
        nh = config["num_attention_heads"]
        nkvh = config.get("num_key_value_heads", nh)
        dh = hs // nh
        di = config["intermediate_size"]
        maxseq = config.get("max_position_embeddings", 32768)
        voc = config["vocab_size"]
        epsilon = config.get("rms_norm_eps", 1e-6)
        theta = config.get("rope_theta", 1000000.0)
        end_token = config.get("eos_token_id", 2)
        dtype = _dtype_name_to_llaisys(str(config.get("torch_dtype", "bfloat16")))

        # CPU kernels don't support BF16 yet, fall back to FP32
        if dtype == DataType.BF16 and device == DeviceType.CPU:
            dtype = DataType.F32

        # --- Create C++ model ---
        meta = LlaisysLlamaMeta(
            dtype=dtype,
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=di,
            maxseq=maxseq,
            voc=voc,
            epsilon=c_float(epsilon),
            theta=c_float(theta),
            end_token=c_int64(end_token),
        )

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysLlamaModelCreate(
            byref(meta),
            device,
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create LLaMA model")

        self._nlayer = nlayer
        self._dtype = dtype
        self._end_token = end_token
        self._voc_size = voc

        # --- Load weights ---
        weights_ptr = LIB_LLAISYS.llaisysLlamaModelWeights(self._model)

        # Map parameter names to weights struct fields
        # Single tensors (same as Qwen2)
        weight_fields = {
            "model.embed_tokens.weight": "in_embed",
            "lm_head.weight": "out_embed",
            "model.norm.weight": "out_norm_w",
        }

        # Per-layer tensor fields (LLaMA: no bias in attention)
        per_layer_fields = {
            "model.layers.{i}.input_layernorm.weight": "attn_norm_w",
            "model.layers.{i}.self_attn.q_proj.weight": "attn_q_w",
            "model.layers.{i}.self_attn.k_proj.weight": "attn_k_w",
            "model.layers.{i}.self_attn.v_proj.weight": "attn_v_w",
            "model.layers.{i}.self_attn.o_proj.weight": "attn_o_w",
            "model.layers.{i}.post_attention_layernorm.weight": "mlp_norm_w",
            "model.layers.{i}.mlp.gate_proj.weight": "mlp_gate_w",
            "model.layers.{i}.mlp.up_proj.weight": "mlp_up_w",
            "model.layers.{i}.mlp.down_proj.weight": "mlp_down_w",
        }

        # Load all safetensors files
        import torch
        self._tensors = []  # Keep Python Tensor objects alive
        for file in sorted(model_path.glob("*.safetensors")):
            f = safetensors.safe_open(str(file), framework="pt", device="cpu")
            for name_ in f.keys():
                tensor_pt = f.get_tensor(name_)
                shape = tuple(tensor_pt.shape)

                if tensor_pt.dtype == torch.bfloat16:
                    if dtype == DataType.F32:
                        tensor_data = tensor_pt.float().numpy()
                        t_dtype = DataType.F32
                    else:
                        tensor_data = tensor_pt.view(torch.uint16).numpy()
                        t_dtype = DataType.BF16
                elif tensor_pt.dtype == torch.float16:
                    if dtype == DataType.F32:
                        tensor_data = tensor_pt.float().numpy()
                        t_dtype = DataType.F32
                    else:
                        tensor_data = tensor_pt.view(torch.uint16).numpy()
                        t_dtype = DataType.F16
                elif tensor_pt.dtype == torch.float32:
                    tensor_data = tensor_pt.numpy()
                    t_dtype = DataType.F32
                else:
                    tensor_data = tensor_pt.float().numpy()
                    t_dtype = DataType.F32

                # Create tensor
                t = Tensor(shape, dtype=t_dtype, device=device)
                t.load(tensor_data)
                self._tensors.append(t)  # Keep alive

                handle = t.lib_tensor()

                # Check if this is a single tensor
                if name_ in weight_fields:
                    field_name = weight_fields[name_]
                    setattr(weights_ptr.contents, field_name, handle)
                    continue

                # Check if this is a per-layer tensor
                for pattern, field_name in per_layer_fields.items():
                    prefix = pattern.split("{i}")[0]
                    suffix = pattern.split("{i}")[1] if "{i}" in pattern else ""
                    if name_.startswith(prefix) and name_.endswith(suffix):
                        # Extract layer index
                        middle = name_[len(prefix):]
                        if suffix:
                            middle = middle[:-len(suffix)]
                        try:
                            layer_idx = int(middle)
                        except ValueError:
                            continue

                        # Get the array pointer
                        arr_ptr = getattr(weights_ptr.contents, field_name)
                        # Cast to array of llaisysTensor_t (c_void_p)
                        ArrType = c_void_p * nlayer
                        arr = ArrType.from_address(arr_ptr)
                        arr[layer_idx] = handle
                        break

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 0.8,
    ) -> List[int]:
        """Generate tokens with sampling support."""
        token_ids = list(inputs)
        ntoken = len(token_ids)
        arr_type = c_int64 * ntoken
        arr = arr_type(*token_ids)

        logits = None

        for _ in range(max_new_tokens):
            ntoken = len(token_ids)

            if temperature <= 0:
                next_token = LIB_LLAISYS.llaisysLlamaModelInfer(
                    self._model, arr, c_size_t(ntoken)
                )
            else:
                if logits is None or len(logits) != self._voc_size:
                    logits = np.zeros(self._voc_size, dtype=np.float32)
                logits_ptr = logits.ctypes.data_as(POINTER(c_float))
                LIB_LLAISYS.llaisysLlamaModelForward(
                    self._model, arr, c_size_t(ntoken), logits_ptr
                )
                next_token = sample(logits, temperature, top_k, top_p)

                arr_type = c_int64 * (ntoken + 1)
                new_arr = arr_type(*token_ids, next_token)
                arr = new_arr

            token_ids.append(next_token)

            if next_token == self._end_token:
                break

        return token_ids

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 0.8,
    ):
        """Generator that yields tokens one by one."""
        token_ids = list(inputs)
        ntoken = len(token_ids)
        arr_type = c_int64 * ntoken
        arr = arr_type(*token_ids)

        logits = None

        for _ in range(max_new_tokens):
            ntoken = len(token_ids)

            if temperature <= 0:
                next_token = LIB_LLAISYS.llaisysLlamaModelInfer(
                    self._model, arr, c_size_t(ntoken)
                )
            else:
                if logits is None or len(logits) != self._voc_size:
                    logits = np.zeros(self._voc_size, dtype=np.float32)
                logits_ptr = logits.ctypes.data_as(POINTER(c_float))
                LIB_LLAISYS.llaisysLlamaModelForward(
                    self._model, arr, c_size_t(ntoken), logits_ptr
                )
                next_token = sample(logits, temperature, top_k, top_p)

                arr_type = c_int64 * (ntoken + 1)
                new_arr = arr_type(*token_ids, next_token)
                arr = new_arr

            token_ids.append(next_token)
            yield next_token

            if next_token == self._end_token:
                break

    def chat(
        self,
        messages: List[Dict[str, str]],
        tokenizer,
        max_new_tokens: int = 128,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 0.8,
        stream: bool = False,
    ):
        """Generate a response for a chat conversation."""
        from .chat_format import format_chat_prompt

        prompt = format_chat_prompt(messages, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt)

        if stream:
            return self._stream_chat_response(inputs, tokenizer, max_new_tokens, top_k, top_p, temperature)
        else:
            output_ids = self.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            return tokenizer.decode(output_ids, skip_special_tokens=True)

    def _stream_chat_response(
        self,
        inputs: List[int],
        tokenizer,
        max_new_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
    ):
        """Stream chat response yielding text chunks."""
        generated_ids = list(inputs)
        for token_id in self.generate_stream(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ):
            generated_ids.append(token_id)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            yield text

    def reset_kv(self):
        """Reset KV-cache for a new conversation."""
        LIB_LLAISYS.llaisysLlamaModelResetKV(self._model)

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysLlamaModelDestroy(self._model)
            self._model = None