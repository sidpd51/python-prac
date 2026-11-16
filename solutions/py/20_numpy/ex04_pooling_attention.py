"""
Exercise 20.4 - Mean pooling, softmax and batched attention          level: ★★★
topics: numpy.axis, numpy.broadcasting, numpy.matmul, numpy.reshaping, numpy.math,
        numpy.aggregation

Problem
-------
What sentence-transformers and a transformer layer do, in NumPy. Shapes:
B = batch, T = query tokens, S = key tokens, D = hidden size, H = heads.
No Python loops over the batch, tokens or heads.

1. mean_pool(token_embs, attention_mask) -> (B, D)
   token_embs (B, T, D); attention_mask (B, T) of 0/1 (or bools), 1 = real token,
   0 = padding. Average only the real tokens of each sequence. A sequence with no real
   tokens pools to a zero vector (no NaN, no warnings).

2. softmax(x, axis=-1) -> same shape
   Numerically stable (subtract the max along `axis` first): softmax([1000, 1000]) must
   be [0.5, 0.5], not NaN.

3. attention(q, k, v, key_mask=None) -> (output (B, T, Dv), weights (B, T, S))
   Scaled dot-product attention: weights = softmax(q @ k^T / sqrt(D)) over the S axis,
   output = weights @ v. q (B, T, D), k (B, S, D), v (B, S, Dv).
   key_mask (B, S): 1/True = attend, 0/False = padding key -> weight exactly 0.
   If ALL keys of a sequence are masked, its weights (and output) are all zeros.

4. split_heads(x, n_heads) -> (B, H, T, D // H)
   Reshape the hidden dimension into heads and move the head axis before T.
   D not divisible by n_heads -> ValueError.
   merge_heads(x) -> (B, T, H * Dh) is the exact inverse.

Interviewers look for
---------------------
* Picking the right axis and keeping dims ((B, T, 1) mask broadcasting over D).
* Why max-subtraction makes softmax stable (exp overflows above ~709 in float64, ~88 in
  float32) and doesn't change the result.
* `@` / np.matmul broadcasts over leading (batch) dims; transpose only the last two axes
  (k.swapaxes(-1, -2)), not k.T which reverses ALL axes.
* reshape vs transpose: reshape reinterprets memory order, transpose permutes axes; you
  need both to split heads correctly.
"""

import numpy as np


def mean_pool(token_embs: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(token_embs.dtype)[..., None]  # (B, T, 1)
    summed = (token_embs * mask).sum(axis=1)  # (B, D)
    counts = mask.sum(axis=1)  # (B, 1)
    return np.divide(summed, counts, out=np.zeros_like(summed), where=counts > 0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, key_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    d = q.shape[-1]
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)  # (B, T, S)
    if key_mask is not None:
        scores = np.where(key_mask.astype(bool)[:, None, :], scores, -np.inf)
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)  # fully-masked rows
    e = np.exp(scores - row_max)  # exp(-inf) == 0 for masked keys
    total = e.sum(axis=-1, keepdims=True)
    weights = np.divide(e, total, out=np.zeros_like(e), where=total > 0)
    return weights @ v, weights


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    b, t, d = x.shape
    if n_heads <= 0 or d % n_heads:
        raise ValueError(f"hidden size {d} not divisible by n_heads={n_heads}")
    return x.reshape(b, t, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    b, h, t, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * dh)
