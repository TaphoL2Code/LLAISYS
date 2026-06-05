"""Sampling strategies for token generation."""

import numpy as np
from typing import Optional


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature. Lower temperature = more deterministic."""
    if temperature <= 0:
        return logits
    return logits / temperature


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax with numerical stability."""
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep only the top-k logits, set others to -inf."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    # Find the k-th largest value
    threshold = np.partition(logits, -k, axis=-1)[..., -k]
    # Set values below threshold to -inf
    mask = logits < threshold[..., np.newaxis]
    return np.where(mask, -np.inf, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep tokens with cumulative probability <= p."""
    if p >= 1.0 or p <= 0:
        return logits
    probs = softmax(logits)
    # Sort in descending order
    sorted_indices = np.argsort(probs, axis=-1)[..., ::-1]
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
    cumsum = np.cumsum(sorted_probs, axis=-1)
    # Create mask: True for tokens to keep
    keep_mask = cumsum <= p
    # Always keep at least the first token
    keep_mask[..., 0] = True
    # Convert sorted mask back to original order
    unsort_indices = np.argsort(sorted_indices, axis=-1)
    keep_mask_original = np.take_along_axis(keep_mask, unsort_indices, axis=-1)
    return np.where(keep_mask_original, logits, -np.inf)


def sample(
    logits: np.ndarray,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: Raw logits of shape (vocab_size,) or (1, vocab_size).
        temperature: Temperature for scaling. 0 = argmax (greedy).
        top_k: Keep only top-k tokens. 0 = disabled.
        top_p: Keep tokens with cumulative probability <= p. 1.0 = disabled.

    Returns:
        Sampled token id (int).
    """
    logits = np.asarray(logits, dtype=np.float32).flatten()

    # Temperature = 0: greedy (argmax)
    if temperature <= 0:
        return int(np.argmax(logits))

    # Temperature scaling
    logits = temperature_scale(logits, temperature)

    # Top-K filtering
    if top_k > 0:
        logits = top_k_filter(logits[np.newaxis, :], top_k)[0]

    # Top-P filtering
    if top_p < 1.0:
        logits = top_p_filter(logits[np.newaxis, :], top_p)[0]

    # Softmax
    probs = softmax(logits)

    # Random sampling
    return int(np.random.choice(len(probs), p=probs))


def argmax_sample(logits: np.ndarray) -> int:
    """Greedy sampling: return the token with highest logit."""
    return int(np.argmax(logits))