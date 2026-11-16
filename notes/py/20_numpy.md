# NumPy

NumPy is the foundation under pandas, scikit-learn, PyTorch-style tensors and every embedding pipeline. An
embedding is a row of a `(n_vectors, dim)` array, cosine similarity is a matrix product, and top-k
retrieval is a sort over one axis. Interviews test whether you understand **memory layout** (views vs copies),
**broadcasting**, and **axis** semantics, and whether you can replace a Python loop with a whole-array
expression. All examples here were run on NumPy 2.5.

## The ndarray and why it beats a list
<!-- topics: arrays, ndarray, vs_lists -->

An `ndarray` is a **fixed-type, fixed-size block of contiguous memory** plus metadata describing how to read it.

```text
Python list of floats                     ndarray float32
+----+                                    one contiguous buffer, 4 bytes per number
| *  |--> float object (24 B + header)     +-----+-----+-----+-----+
| *  |--> float object                     | 1.0 | 2.0 | 3.0 | 4.0 |   + metadata: shape, dtype, strides
| *  |--> float object                     +-----+-----+-----+-----+
+----+  (pointers 8 B each, objects scattered)
```

Consequences: compact (a list of 1000 ints costs about 36 KB with its int objects, the array 8 KB), cache-friendly,
and operations run as **one C loop** over raw numbers instead of Python bytecode per element.

| | `list` | `ndarray` |
|---|---|---|
| Element types | mixed | one `dtype` |
| Size / append | grows cheaply | fixed; "append" copies the whole thing |
| `a * 2` | repeats/concatenates (lists) | multiplies every element |
| Math speed | slow (per-element objects) | 10-100x faster (C, SIMD, BLAS) |

```python
import numpy as np

print([1, 2] * 2)            # prints: [1, 2, 1, 2]
print(np.array([1, 2]) * 2)  # prints: [2 4]
```

> **Remember** NumPy wins by moving the loop out of Python. If you write `for` over array elements you have
> lost the benefit.

> **AI use** Embedding APIs return Python lists of floats. Convert once: `np.asarray(rows, dtype=np.float32)`.
> 1M vectors of 768 dims in float32 is 3.07 GB; in Python lists it would be several times that.

## Creating arrays and attributes
<!-- topics: creation, attributes -->

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(a.shape, a.ndim, a.size)       # prints: (2, 3) 2 6
print(a.dtype, a.itemsize, a.nbytes) # prints: float32 4 24
print(a.strides)                     # prints: (12, 4)
```

`strides` are bytes to step to the next element along each axis: 12 bytes to the next *row* (3 numbers x 4 B),
4 bytes to the next *column*. Everything about views, transpose and reshape follows from strides.

Constructors you should know cold:

```python
import numpy as np

print(np.zeros((2, 2), dtype=int))     # 2x2 of 0
print(np.full(3, 7))                   # prints: [7 7 7]
print(np.arange(0, 1, 0.25))           # prints: [0.   0.25 0.5  0.75]
print(np.linspace(0, 1, 5))            # prints: [0.   0.25 0.5  0.75 1.  ]
print(np.eye(2))                       # identity matrix
print(np.array([1, 2, 3]).dtype, np.array([1, 2.0]).dtype)   # prints: int64 float64
```

`arange(start, stop, step)` excludes `stop` and is unreliable with float steps (use `linspace` for a known count).
`np.empty` returns uninitialised memory; `zeros_like` / `ones_like` copy shape and dtype.

> **Gotcha** **dtype is decided at creation and silently controls results.** Integer arrays overflow
> instead of raising: `np.array([127], dtype=np.int8) + 1` gives `[-128]`. Mixed numeric input upcasts; `np.array([1, "a"])` becomes
> a string array. Pass `dtype=np.float32` explicitly for embeddings (default float64 doubles memory). Floats are approximate (`0.1 + 0.2 != 0.3`): compare with `np.isclose`.

`astype` converts (always copies). Default integer is `int64`, default float `float64` on 64-bit Linux/macOS.

## Indexing, slicing: views vs copies
<!-- topics: indexing -->

Basic indexing (integers, slices) on an array returns a **view**: a new array object that **shares the same
buffer** (new shape/strides, same memory). Advanced indexing (index arrays, boolean masks) returns a **copy**.

```python
import numpy as np

x = np.arange(10)
v = x[2:5]            # slice -> view
v[0] = 99             # writes through to x
print(x)              # prints: [ 0  1 99  3  4  5  6  7  8  9]
print(v.base is x, np.shares_memory(x, v))   # prints: True True

f = x[[1, 2]]         # list of indices ("fancy") -> copy
f[0] = -1
print(x[1])           # prints: 1
```

2-D indexing is `a[rows, cols]`, with slices per axis:

```python
import numpy as np

m = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(m[1])            # prints: [4 5 6 7]        (row)
print(m[:, 1])         # prints: [1 5 9]          (column)
print(m[1:, ::2])      # prints: [[ 4  6]
                       #          [ 8 10]]
print(m[[0, 2], [1, 3]])   # prints: [ 1 11]      (pairs (0,1) and (2,3), NOT a 2x2 block)
print(m[np.ix_([0, 2], [1, 3])])   # prints: [[ 1  3]
                                   #          [ 9 11]]   (the 2x2 block)
```

> **Gotcha** Views mean a function that modifies a slice modifies the caller's array. Ask `arr.base` or
> `np.shares_memory(a, b)`; take `.copy()` when you need independence. The opposite bug: assuming `a[idx] = ...`
> with a fancy index "copies": assignment through an index writes into `a`, it is *reading* with fancy indexing that copies.
> Also `x[1:3]` on a **list** is a copy; on an array it is a view.

## Reshaping
<!-- topics: reshaping -->

`reshape` changes the shape without changing the data; the total element count must match, and one dimension may be `-1`
(inferred). It returns a **view when the memory layout allows it**, otherwise a copy.

```python
import numpy as np

a = np.arange(6)
b = a.reshape(2, 3)
print(b)                       # prints: [[0 1 2]
                               #          [3 4 5]]
print(a.reshape(-1, 2).shape)  # prints: (3, 2)
print(np.shares_memory(a, b))  # prints: True
print(b.T.shape, b.T.strides, b.strides)   # prints: (3, 2) (8, 24) (24, 8)
```

`.T` (transpose) is free: it only swaps strides, so a transposed array is **not C-contiguous**, and reshaping it
may need a copy. `ravel()` returns a view if possible; `flatten()` always copies. Related tools: `np.newaxis` /
`None` to add an axis (`v[:, None]` turns shape `(n,)` into `(n, 1)`), `np.squeeze`, `np.stack` / `np.concatenate`,
`np.transpose(axes)` for N-d.

```python
import numpy as np

v = np.array([1, 2, 3])
print(v[:, None].shape, v[None, :].shape)   # prints: (3, 1) (1, 3)
print(np.concatenate([v, v]).shape, np.stack([v, v]).shape)   # prints: (6,) (2, 3)
```

## Boolean indexing
<!-- topics: boolean_indexing -->

A comparison produces a boolean array; using it as an index selects the `True` positions (always a **copy**).

```python
import numpy as np

m = np.arange(12).reshape(3, 4)
print(m[m % 2 == 0])                 # prints: [ 0  2  4  6  8 10]
print(np.where(m > 5, 1, 0))         # elementwise if/else
# [[0 0 0 0]
#  [0 0 1 1]
#  [1 1 1 1]]
m2 = m.copy()
m2[m2 > 8] = 0                       # masked assignment writes in place
print(m2[2])                         # prints: [8 0 0 0]
```

Combine conditions with `&`, `|`, `~` and **parentheses** (`and`/`or` do not work on arrays):

```python
import numpy as np

s = np.array([0.9, 0.2, 0.7, 0.4])
mask = (s > 0.3) & (s < 0.8)
print(mask, s[mask])   # prints: [False False  True  True] [0.7 0.4]
```

> **Gotcha** `s > 0.3 and s < 0.8` raises "truth value of an array is ambiguous", and `s > 0.3 & s < 0.8`
> parses as `s > (0.3 & s) < 0.8` because `&` binds tighter than comparison. Always parenthesise each comparison.

> **AI use** Metadata filtering before ranking: `scores[(lang == "en") & (age_days < 30)]`, or set masked-out scores to
> `-inf` so they can never reach the top-k.

## Broadcasting
<!-- topics: broadcasting -->

**Broadcasting** lets arrays of different shapes combine in one operation without copying data. The rule, applied
**from the right**, dimension by dimension:

1. Align shapes at the **trailing** end, padding the shorter one with 1s on the left.
2. For each dimension the sizes must be **equal**, or **one of them is 1** (a size-1 dimension is stretched).
3. The result dimension is the larger of the two; otherwise `ValueError`.

```text
a: (3, 1)        b:    (4,)       ->  pad b:  (1, 4)
        3  1                                   1  4
        ^  ^   -> each pair: (3,1) ok, (1,4) ok  -> result (3, 4)

(3, 2) + (3,)     pad: (3, 2) vs (1, 3)   trailing: 2 vs 3  -> mismatch -> error
(3, 2) + (3, 1)   trailing: 2 vs 1 ok, 3 vs 3 ok          -> (3, 2)
```

```python
import numpy as np

a = np.arange(3).reshape(3, 1)
b = np.arange(4)
print((a + b).shape)      # prints: (3, 4)
print(a + b)
# [[0 1 2 3]
#  [1 2 3 4]
#  [2 3 4 5]]

try:
    np.ones((3, 2)) + np.ones(3)
except ValueError as e:
    print(e)    # prints: operands could not be broadcast together with shapes (3,2) (3,)
```

The practical pattern is **centering / normalising by row or column statistics**:

```python
import numpy as np

X = np.array([[1.0, 2, 3], [4, 5, 6]])
print(X - X.mean(axis=0))                      # column means, shape (3,) broadcasts across rows
# [[-1.5 -1.5 -1.5]
#  [ 1.5  1.5  1.5]]
print(X / X.sum(axis=1, keepdims=True))        # row sums need shape (2, 1)
# [[0.16666667 0.33333333 0.5       ]
#  [0.26666667 0.33333333 0.4       ]]
```

> **Gotcha** `X / X.sum(axis=1)` fails (shapes `(2, 3)` and `(2,)`: trailing `3` vs `2`). The fix is
> `keepdims=True` or `[:, None]`, which keeps the reduced axis as size 1. A "wrong but runs" version arises when the
> sizes happen to match (a square matrix divides by the wrong axis silently).

<details markdown="1"><summary>Check yourself: shapes `(5, 1, 3)` and `(4, 1)`. Result?</summary>

Pad `(4, 1)` to `(1, 4, 1)`. Compare right to left: 3 vs 1 -> 3, 1 vs 4 -> 4, 5 vs 1 -> 5. Result `(5, 4, 3)`.

</details>

## Vectorization, math functions and floating point
<!-- topics: vectorization, math -->

**Vectorisation** means expressing work as whole-array operations (**ufuncs** such as `+`, `np.sqrt`, `np.exp`,
`np.maximum`) so the loop runs in compiled C. Rewrite loops by thinking "what is the operation on the whole array?".

```python
import numpy as np

x = np.array([4.0, 9.0, 16.0])
print(np.sqrt(x), x ** 2, np.exp(0.0), np.log(np.e))   # prints: [2. 3. 4.] [ 16.  81. 256.] 1.0 1.0
print(np.clip([-1, 5, 10], 0, 6))                      # prints: [0 5 6]
print(np.round(2.5), np.round(3.5))                    # prints: 2.0 4.0   (round half to even)
```

Floating-point realities: `NaN != NaN`, `inf` appears on division by zero (with a `RuntimeWarning`, not an exception),
and results are approximate.

```python
import numpy as np

n = np.array([np.nan, 1.0, 2.0])
print(n.sum(), np.nansum(n))        # prints: nan 3.0
print(np.nan == np.nan, np.isnan(n))   # prints: False [ True False False]
with np.errstate(divide="ignore", invalid="ignore"):
    print(np.array([1.0, 0.0]) / np.array([0.0, 0.0]))   # prints: [inf nan]
```

Vectorisation is fast because there is one Python-level call, a tight C loop with SIMD, and no per-element objects. It stops helping with many tiny arrays or big temporaries (`a * b + c` allocates intermediates; `out=` avoids that).

> **Remember** `np.vectorize` is **not** vectorisation: it is a Python loop in disguise (convenience, not speed).
> Real speed needs ufuncs, `np.where`, boolean masks, or `einsum` / matmul.

## Aggregation and the axis argument
<!-- topics: aggregation, axis -->

Aggregations (`sum`, `mean`, `std`, `min`, `max`, `argmax`, `cumsum`, ...) reduce an array. **`axis` names the
dimension that disappears**: you collapse along it.

```text
X = [[1, 2, 3],        axis=0 collapses rows  (go DOWN each column)   -> [5, 7, 9]       shape (3,)
     [4, 5, 6]]        axis=1 collapses columns (go ACROSS each row)  -> [6, 15]         shape (2,)
                       axis=None (default): everything                -> 21
```

```python
import numpy as np

X = np.array([[1.0, 2, 3], [4, 5, 6]])
print(X.sum(), X.sum(axis=0), X.sum(axis=1))   # prints: 21.0 [5. 7. 9.] [ 6. 15.]
print(X.mean(axis=0), X.argmax(axis=1))        # prints: [2.5 3.5 4.5] [2 2]
print(X.cumsum(axis=1))
# [[ 1.  3.  6.]
#  [ 4.  9. 15.]]
print(X.sum(axis=1, keepdims=True).shape)      # prints: (2, 1)
```

For N-d, `axis` can be a negative number or a tuple: with shape `(B, T, D)`, `axis=-1` reduces `D`, and
`axis=(0, 2)` leaves `(T,)`:

```python
import numpy as np

Z = np.zeros((2, 3, 4))
print(Z.sum(axis=-1).shape, Z.sum(axis=(0, 2)).shape)   # prints: (2, 3) (3,)
```

> **Gotcha** `std` defaults to population std (`ddof=0`), pandas defaults to sample std (`ddof=1`). `argmax` returns the
> *first* maximum, and an index into the flattened array when `axis=None`. NaNs poison `sum`/`mean`; use `nansum`, `nanmean`.

## Random numbers
<!-- topics: random -->

Use the modern **Generator API**: `rng = np.random.default_rng(seed)`. It is reproducible, independent per
instance (no hidden global state), and statistically better than the legacy `np.random.seed` / `np.random.rand`.

```python
import numpy as np

rng = np.random.default_rng(0)
print(rng.random(3))            # prints: [0.63696169 0.26978671 0.04097352]
print(rng.integers(0, 10, 4))   # prints: [0 0 1 8]
print(rng.normal(size=2))       # prints: [0.36159505 1.30400005]
print(np.random.default_rng(0).random(3)[0])   # prints: 0.6369616873214543   (same seed -> same stream)
```

Handy: `rng.choice(n, size, replace=False)` for sampling, `rng.permutation(n)` / `rng.shuffle` for shuffles,
`rng.standard_normal((n, d))` for fake embeddings. Pass the `rng` object into functions rather than seeding globally.

> **AI use** Seed everything that creates eval splits, negatives or synthetic data; record the seed with the
> result. Fake embeddings for tests: `rng.standard_normal((100, 8)).astype(np.float32)`.

## Linear algebra and matrix multiplication
<!-- topics: linalg, matmul -->

Elementwise `*` is **not** matrix multiplication. Use `@` (`np.matmul`):

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
print(A * B)    # elementwise
# [[0 2]
#  [3 0]]
print(A @ B)    # matrix product: rows of A dot columns of B
# [[2 1]
#  [4 3]]
print(np.dot([1, 2], [3, 4]), np.linalg.norm([3, 4]))   # prints: 11 5.0
```

Shape rule: `(m, k) @ (k, n) -> (m, n)`: inner dimensions must match. For stacks, leading dimensions broadcast:
`(B, T, D) @ (B, D, S) -> (B, T, S)`. `@` on a 1-D vector treats it as a row/column as needed (`(n,) @ (n,)` is a scalar).

`np.linalg` basics: `norm(x, axis=...)`, `inv`, `det`, `solve(A, b)` (prefer this to `inv(A) @ b`), `eig`, `svd`.

```python
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
print(np.linalg.solve(A, [5.0, 6.0]))        # prints: [-4.   4.5]
print(np.linalg.norm(A, axis=1))             # prints: [2.23606798 5.        ]
```

Matrix products dispatch to optimised BLAS (multi-threaded, no Python GIL involved), which is why one `@` beats a loop
of dot products by orders of magnitude.

> **Gotcha** `A @ B.T` vs `A.T @ B` are the shape mistakes interviewers like. Write the shapes in a comment before
> multiplying, and remember `*` never means matrix product on ndarrays.

## Embeddings: cosine similarity and top-k
<!-- topics: vectorization, matmul, axis, boolean_indexing, indexing -->

Cosine similarity between vectors is the dot product of their **L2-normalised** versions. Normalise the corpus
once, normalise each query, then one matrix product scores everything.

```python
import numpy as np

def l2_normalize(x, axis=-1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)   # zero vectors stay zero

docs = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
query = np.array([1, 0.2, 0], dtype=np.float32)

scores = l2_normalize(docs) @ l2_normalize(query)       # (5, 3) @ (3,) -> (5,)
print(scores.round(3), scores.dtype)   # prints: [ 0.981  0.902  0.     0.196 -0.981] float32
```

```text
docs (N, D) @ query (D,) = scores (N,)           queries (Q, D) @ docs.T (D, N) = scores (Q, N)
cosine = dot(unit_a, unit_b)                      unit vectors: |a| = |b| = 1, so dot = cos(angle)
```

For top-k you do not need a full sort. `np.argpartition` finds the k best in O(n) (but unordered); sort just those k:

```python
import numpy as np

scores = np.array([0.981, 0.902, 0.0, 0.196, -0.981])
k = 2
part = np.argpartition(-scores, k - 1)[:k]        # indices of the k largest, unordered
top = part[np.argsort(-scores[part])]             # order only those k
print(top, scores[top])                            # prints: [0 1] [0.981 0.902]

batch = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.8]])    # (Q, N): top-1 per query
print(np.argmax(batch, axis=1))                    # prints: [1 2]
print(np.argsort(-batch, axis=1)[:, :2])           # prints: [[1 2]
                                                   #          [2 0]]
```

Using `-scores` avoids the descending-sort copy; `np.argsort(scores)[::-1]` also works but is a view with
negative stride. Filtering and thresholds are boolean masks (`scores >= 0.5`), and `np.where(valid, scores, -np.inf)` hides
masked items before the top-k.

> **Gotcha** Zero vectors give `0/0 = NaN` in a naive normalisation (hence the `where=norm > 0` guard). Also
> dot product equals cosine **only if both sides are normalised**: many embedding models return unit vectors, some do not;
> check `np.linalg.norm(vectors, axis=1)`.

> **Interview** "Brute-force vs ANN?" Exact search is one matmul: O(N x D) per query, perfectly fine up to
> ~1M vectors on a laptop with float32 and BLAS. ANN indexes (HNSW, IVF in FAISS/pgvector) trade recall for speed beyond
> that, and you evaluate them against this exact baseline (recall@k).

## Pooling, softmax and attention in NumPy
<!-- topics: broadcasting, reshaping, axis, matmul -->

Transformer models output one vector per **token**: shape `(B, T, D)`. A sentence embedding is a **masked mean** over the
token axis: padding tokens (mask 0) must not count.

```python
import numpy as np

rng = np.random.default_rng(0)
tokens = rng.random((2, 4, 3))                  # (B=2, T=4, D=3)
mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])   # (B, T): 1 = real token

summed = (tokens * mask[..., None]).sum(axis=1)       # mask (B,T,1) broadcasts over D
pooled = summed / mask.sum(axis=1, keepdims=True)     # divide by true length per row
print(pooled.shape)                                    # prints: (2, 3)
print(np.allclose(pooled[1], tokens[1, :2].mean(axis=0)))   # prints: True
```

**Softmax** turns scores into probabilities along an axis. The naive `exp(x)` overflows for big values; subtract
the max first (the result is unchanged, because softmax is shift-invariant):

```python
import numpy as np

def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

print(softmax(np.array([1000.0, 1001.0])))   # prints: [0.26894142 0.73105858]
print(softmax(np.array([[1.0, 2, 3], [1, 1, 1]])).round(3))
# [[0.09  0.245 0.665]
#  [0.333 0.333 0.333]]
```

**Attention** scores are `softmax(Q K^T / sqrt(d)) V`, batched with matmul and transposes of the last two axes:

```python
import numpy as np

rng = np.random.default_rng(0)
Q = rng.random((2, 3, 8))     # (B, T queries, D)
K = rng.random((2, 5, 8))     # (B, S keys, D)
V = rng.random((2, 5, 8))
scores = Q @ K.transpose(0, 2, 1) / np.sqrt(8)    # (B,T,D) @ (B,D,S) -> (B,T,S)
w = np.exp(scores - scores.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
out = w @ V                                        # (B,T,S) @ (B,S,D) -> (B,T,D)
print(scores.shape, out.shape, np.allclose(w.sum(-1), 1))   # prints: (2, 3, 5) (2, 3, 8) True
```

Write the shape of every intermediate in a comment; prefer `axis=-1` so the code works for 1-D, 2-D and batched input.

<details markdown="1"><summary>Check yourself: why is dividing by `mask.sum(axis=1, keepdims=True)` needed rather than dividing by `T`?</summary>

Padding positions are zeroed in the numerator but would still be counted in the denominator, shrinking every
short sequence's mean. Dividing by the real token count fixes that; `keepdims` keeps shape `(B, 1)` so it broadcasts.

</details>

## Quick recap
- `ndarray` = one contiguous typed buffer plus shape, dtype and strides; lists are pointer arrays to objects.
- Slicing/`reshape`/`.T`/`ravel` give views (shared memory); fancy and boolean indexing, `astype`, `flatten` copy. Check `np.shares_memory`.
- Broadcasting: align trailing dims; each pair equal or one is 1; add axes with `[:, None]` or `keepdims=True`.
- `axis` is the dimension that disappears; result shape = input shape minus that axis.
- Vectorise with ufuncs, `np.where`, masks and `@`; `np.vectorize` is just a loop. Combine masks with `&`, `|`, `~` and parentheses.
- dtype is chosen at creation: integer overflow wraps silently, floats are approximate, use `float32` for embeddings, `isclose` to compare.
- Use `np.random.default_rng(seed)`; pass the generator around.
- `A @ B` needs matching inner dimensions; `solve` beats `inv`; batched matmul broadcasts leading dims.
- Cosine = dot of L2-normalised vectors (guard zero norms); top-k with `argpartition` then sort the k; masks for filters.
- Masked mean pooling and max-subtracted softmax are the standard embedding-model building blocks.
