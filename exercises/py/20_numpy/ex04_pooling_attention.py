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
    raise NotImplementedError


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    raise NotImplementedError


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, key_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    raise NotImplementedError


def merge_heads(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError
