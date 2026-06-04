from ctypes import (
    c_void_p,
    c_size_t,
    c_int,
    c_int64,
    c_float,
    POINTER,
    Structure,
)

from .tensor import llaisysTensor_t
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", c_void_p),  # llaisysTensor_t*
        ("attn_q_w", c_void_p),
        ("attn_q_b", c_void_p),
        ("attn_k_w", c_void_p),
        ("attn_k_b", c_void_p),
        ("attn_v_w", c_void_p),
        ("attn_v_b", c_void_p),
        ("attn_o_w", c_void_p),
        ("mlp_norm_w", c_void_p),
        ("mlp_gate_w", c_void_p),
        ("mlp_up_w", c_void_p),
        ("mlp_down_w", c_void_p),
    ]


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64