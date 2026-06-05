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


class LlaisysLlamaMeta(Structure):
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


class LlaisysLlamaWeights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", c_void_p),  # llaisysTensor_t*
        ("attn_q_w", c_void_p),
        ("attn_k_w", c_void_p),
        ("attn_v_w", c_void_p),
        ("attn_o_w", c_void_p),
        ("mlp_norm_w", c_void_p),
        ("mlp_gate_w", c_void_p),
        ("mlp_up_w", c_void_p),
        ("mlp_down_w", c_void_p),
    ]


def load_llama(lib):
    lib.llaisysLlamaModelCreate.argtypes = [
        POINTER(LlaisysLlamaMeta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysLlamaModelCreate.restype = c_void_p

    lib.llaisysLlamaModelDestroy.argtypes = [c_void_p]
    lib.llaisysLlamaModelDestroy.restype = None

    lib.llaisysLlamaModelWeights.argtypes = [c_void_p]
    lib.llaisysLlamaModelWeights.restype = POINTER(LlaisysLlamaWeights)

    lib.llaisysLlamaModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
    lib.llaisysLlamaModelInfer.restype = c_int64

    lib.llaisysLlamaModelForward.argtypes = [c_void_p, POINTER(c_int64), c_size_t, POINTER(c_float)]
    lib.llaisysLlamaModelForward.restype = None

    lib.llaisysLlamaModelResetKV.argtypes = [c_void_p]
    lib.llaisysLlamaModelResetKV.restype = None