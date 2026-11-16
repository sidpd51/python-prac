# Python Fundamentals

Everything else in Python sits on a small object model: names point to objects, objects have a type and an
identity, and some objects can change while others cannot. Most "weird" interview snippets (a list that changes
behind your back, `is` returning surprising results, an empty string being falsy) are this model showing through.
In AI engineering the same ideas show up daily: parsing nested JSON from an LLM API, normalising text before
tokenising, shaping config, and not corrupting a shared default dict. Get the model right and the rest is syntax.

## Variables are names, not boxes
<!-- topics: variables -->

In Python a variable is a **name bound to an object**. Assignment never copies; it makes the name point at an
object that already exists (or is created by the right-hand side).

```text
a = [1, 2]       a ──────►  [1, 2]
b = a            b ──────►  (same object)
```

Every object has an **identity** (`id()`, roughly its memory address), a **type**, and a **value**. Types belong to
objects, not names, so a name can be rebound to any type (dynamic typing) but each object is strongly typed:
`"1" + 1` raises `TypeError` instead of guessing.

```python
x = 10
print(type(x).__name__)       # prints: int
x = "ten"                     # same name, new object of another type
print(type(x).__name__)       # prints: str
a, b = 1, 2
a, b = b, a                   # tuple packing/unpacking swap
print(a, b)                   # prints: 2 1
first, *rest = [1, 2, 3, 4]
print(first, rest)            # prints: 1 [2, 3, 4]
```

The core built-in types you must know cold: `int` (unbounded), `float` (IEEE-754 double), `bool` (a subclass of
`int`), `str`, `bytes`, `list`, `tuple`, `dict`, `set`, `NoneType`.

```python
print(isinstance(True, int))   # prints: True
print(True + True)             # prints: 2
print(0.1 + 0.2 == 0.3)        # prints: False
print(2 ** 100)                # prints: 1267650600228229401496703205376
```

> **Gotcha** Floats are binary approximations. Compare with `math.isclose(a, b)`, and use `decimal.Decimal` for money.
> Python ints never overflow, which is why hashing/bit tricks need explicit masks.

> **Interview** "Is Python pass-by-value or pass-by-reference?" Neither: it is *pass-by-object-reference*
> (call by assignment). The parameter name is bound to the same object the caller passed.

## Type conversion
<!-- topics: type_conversion -->

Python almost never converts implicitly (except numeric widening like `int + float`). You convert explicitly with the
type's constructor: `int()`, `float()`, `str()`, `bool()`, `list()`, `tuple()`, `set()`, `dict()`.

```python
print(int("42"), int(3.9), int(-3.9))     # prints: 42 3 -3
print(float("1e3"), str(3.0), bool("0"))  # prints: 1000.0 3.0 True
print(int("ff", 16), int("0b101", 0))     # prints: 255 5
print(list("abc"), set([1, 1, 2]))        # prints: ['a', 'b', 'c'] {1, 2}
print(dict([("a", 1), ("b", 2)]))         # prints: {'a': 1, 'b': 2}
```

`int()` **truncates toward zero**; use `round()` (banker's rounding: `round(2.5) == 2`) or `math.floor/ceil` when you mean
something else. `int("3.5")` and `int("")` raise `ValueError`; `int(None)` raises `TypeError`. Convert at the edges of
your program (parsing env vars, CSV fields, JSON strings) and wrap with `try/except ValueError`.

```python
def to_int(raw: str, default: int = 0) -> int:
    try:
        return int(raw.strip())
    except (ValueError, AttributeError):
        return default

print(to_int(" 8 "), to_int("8.5"), to_int(None))   # prints: 8 0 0
print(round(2.5), round(3.5), round(2.675, 2))      # prints: 2 4 2.67
```

> **Gotcha** `bool("False")` is `True`: any non-empty string is truthy. Parse env flags explicitly:
> `os.environ.get("DEBUG", "").lower() in {"1", "true", "yes"}`.

> **AI use** Env vars and CLI args are always strings. `MAX_TOKENS=1024` must be `int()`-ed before it reaches an API
> call, or the API rejects `"1024"`.

## Mutable vs immutable objects
<!-- topics: mutability -->

An **immutable** object cannot change after creation: `int`, `float`, `bool`, `str`, `bytes`, `tuple`, `frozenset`,
`None`. A **mutable** one can change in place: `list`, `dict`, `set`, `bytearray`, and most user-defined classes.
"Changing" an immutable value really builds a new object and rebinds the name.

```python
s = "abc"
before = id(s)
s += "d"                        # new str, name rebound
print(id(s) == before)          # prints: False

lst = [1]
before = id(lst)
lst += [2]                      # in-place extend, same object
print(id(lst) == before, lst)   # prints: True [1, 2]
```

Why it matters: **aliasing**. Two names for one mutable object means a mutation through either is visible through both.

```python
def add_tag(tag, tags=[]):      # the default list is created ONCE at def time
    tags.append(tag)
    return tags

print(add_tag("a"))             # prints: ['a']
print(add_tag("b"))             # prints: ['a', 'b']

def add_tag_ok(tag, tags=None):
    tags = [] if tags is None else tags
    tags.append(tag)
    return tags

print(add_tag_ok("a"), add_tag_ok("b"))   # prints: ['a'] ['b']
```

Immutability is **shallow**: a tuple cannot change which objects it holds, but a list inside it can still change.

```python
t = ([1], 2)
t[0].append(99)
print(t)                        # prints: ([1, 99], 2)
try:
    t[0] += [5]                 # += mutates the list, THEN fails assigning to t[0]
except TypeError as e:
    print(e)                    # prints: 'tuple' object does not support item assignment
print(t)                        # prints: ([1, 99, 5], 2)
```

Copies: `b = a` aliases; `a.copy()`, `a[:]`, `list(a)` are **shallow** (inner objects shared); `copy.deepcopy(a)` clones
the whole graph.

```python
import copy
a = [[1], [2]]
shallow, deep = a.copy(), copy.deepcopy(a)
a[0].append("x")
print(shallow, deep)            # prints: [[1, 'x'], [2]] [[1], [2]]
rows = [[0] * 2] * 2            # TRAP: two references to the same inner list
rows[0][0] = 1
print(rows)                     # prints: [[1, 0], [1, 0]]
```

> **Remember** Never use a mutable object as a default argument; never build a grid with `[[0]*n]*m`
> (use `[[0]*n for _ in range(m)]`).

<details markdown="1"><summary>Check yourself: why does `t[0] += [5]` both raise and change the tuple?</summary>

`+=` on a list calls `list.__iadd__`, which mutates in place and succeeds; then Python tries to store the result back into
`t[0]`, which a tuple forbids, so `TypeError` fires after the mutation already happened.

</details>

## Identity vs equality
<!-- topics: identity_equality -->

`==` asks "do these have equal values?" (calls `__eq__`). `is` asks "are these the very same object?" (compares
identity, cannot be overridden).

```python
a = [1, 2]
b = [1, 2]
c = a
print(a == b, a is b, a is c)   # prints: True False True
```

Use `is` for singletons: `x is None`, `x is True`, `x is NotImplemented`. Use `==` for everything else.

CPython **caches** small ints (-5..256) and interns some strings, so `is` can look like it works on values and then
break:

```python
x = int("256"); y = int("256")
print(x is y)                   # prints: True   (cached small int)
x = int("1000"); y = int("1000")
print(x is y)                   # prints: False  (two separate objects)
print(x == y)                   # prints: True
```

> **Gotcha** `is` on numbers and strings is an implementation detail, not a language guarantee (and warns
> `SyntaxWarning` when used with a literal). Never rely on it. Also `float('nan') != float('nan')`, so
> `x == x` is not always `True`, but `[nan] == [nan]` can be, because list comparison checks identity first.

> **Interview** "When would `a == b` be True but `a is b` False, and the reverse?" The first is common (equal lists).
> The reverse needs a weird `__eq__` (NaN or a class returning False). Say that `is None` is preferred because
> a class can override `==` to match `None`, but not `is`.

## Truthiness and None
<!-- topics: truthiness, none -->

Every object has a boolean value. **Falsy**: `None`, `False`, zero (`0`, `0.0`, `0j`), and any **empty** container
or string (`""`, `[]`, `()`, `{}`, `set()`, `range(0)`). Everything else is truthy, including `"0"`, `"False"`,
`[0]` and `[None]`. Custom classes are truthy unless they define `__bool__` or `__len__`.

```python
print([bool(v) for v in (0, 0.0, "", " ", [], [0], {}, None, "0")])
# prints: [False, False, False, True, False, True, False, False, True]
```

`None` is the single instance of `NoneType` and means "no value". A function without `return` returns `None`.
Because `0`, `""` and `[]` are *also* falsy, `if not x` conflates "missing" with "empty or zero".

```python
def tokens_left(budget=None):
    if budget is None:          # distinguishes "not given" from 0
        return "unlimited"
    return budget

print(tokens_left(), tokens_left(0))   # prints: unlimited 0
print(None or "fallback", 0 or 5, [] or "x")   # prints: fallback 5 x
```

`or`/`and` return one of their **operands**, not a bool (see operators below). In-place list methods such as
`list.sort()`, `append`, `reverse` return `None`, so `x = x.sort()` silently sets `x` to `None`.

> **Gotcha** `temperature = cfg.get("temperature") or 0.7` turns an explicit `0.0` into `0.7`. Use `is None` checks
> when zero/empty is a valid value.

## Operators
<!-- topics: operators -->

```python
print(7 / 2, 7 // 2, -7 // 2, 7 % 3, -7 % 3, 2 ** 10)   # prints: 3.5 3 -4 1 2 1024
print(divmod(17, 5))                                      # prints: (3, 2)
print(1 < 2 < 3, 1 < 3 < 2)                               # prints: True False
print(5 in [1, 5], "ab" in "cabd", 3 not in {1, 2})       # prints: True True True
print(6 & 3, 6 | 3, 6 ^ 3, 1 << 4, ~0)                    # prints: 2 7 5 16 -1
```

Key points:

* `/` is always true division (float). `//` floors toward negative infinity, so `-7 // 2 == -4`, and `%` has the sign of the
  divisor so that `a == (a // b) * b + a % b` always holds.
* **Comparison chaining**: `a < b < c` means `a < b and b < c`, with `b` evaluated once.
* **Short-circuit**: `and`/`or` stop early and return an operand. `a and b` returns `a` if falsy else `b`; `a or b`
  returns `a` if truthy else `b`.
* `in` uses `__contains__`: O(1) for sets/dicts, O(n) for lists/strings.
* The walrus `:=` assigns inside an expression.

```python
data = {"user": {"name": "Ada"}}
name = data.get("user", {}).get("name") or "anon"
print(name)                                   # prints: Ada
if (n := len("hello")) > 3:
    print(n)                                  # prints: 5
print(1 if [] else 2)                         # prints: 2
```

## Conditionals
<!-- topics: conditionals -->

`if / elif / else` evaluates conditions top to bottom and runs the first truthy branch. The conditional expression is
`a if cond else b`. Since 3.10, `match` does structural pattern matching, which is more than a switch: it can destructure.

```python
def route(msg: dict) -> str:
    match msg:
        case {"role": "tool", "name": name}:
            return f"tool:{name}"
        case {"role": "assistant", "content": str() as text} if text:
            return "assistant text"
        case {"role": "user"}:
            return "user"
        case _:
            return "unknown"

print(route({"role": "tool", "name": "search"}))     # prints: tool:search
print(route({"role": "assistant", "content": ""}))   # prints: unknown
print(route({"role": "user", "content": "hi"}))      # prints: user
```

Mapping patterns match if the listed keys exist (extra keys are fine). Prefer **guard clauses** (early return) over deep
nesting, and a dict dispatch table (`{"add": add}[op]`) when branching only selects a function.

## Loops and loop control
<!-- topics: loops, loop_control -->

`for` iterates over any **iterable** by calling `iter()` and then `next()` until `StopIteration`. `while` repeats while a
condition holds. Use `for` when you have a collection, `while` when you do not know the count.

```python
for i, ch in enumerate("ab", start=1):
    print(i, ch)                        # prints: 1 a  then 2 b
for k, v in {"x": 1}.items():
    print(k, v)                         # prints: x 1
for a, b in zip([1, 2, 3], "ab"):
    print(a, b)                         # prints: 1 a  then 2 b   (zip stops at the shortest)
```

* `break` leaves the innermost loop; `continue` skips to the next iteration; `pass` is a no-op placeholder.
* A loop's **`else`** runs only if the loop finished without `break`.
* The loop variable **survives** after the loop and holds the last value.

```python
for n in range(2, 8):
    for d in range(2, n):
        if n % d == 0:
            break
    else:                               # no break: n is prime
        print(n, end=" ")
print()                                 # prints: 2 3 5 7
print(n)                                # prints: 7

count = 0
while True:
    count += 1
    if count < 3:
        continue
    break
print(count)                            # prints: 3
```

> **Gotcha** Do not add or remove items of a list while iterating over it; items get skipped. Iterate over a copy
> (`for x in lst[:]`) or build a new list. Mutating a dict's size during iteration raises `RuntimeError`.

## Comprehensions: list, dict, set
<!-- topics: list_comprehensions, dict_comprehensions, set_comprehensions -->

A comprehension builds a collection from an iterable in one expression: `[expr for x in it if cond]`.
Read it as the loop it replaces; the `if` filters, a conditional *expression* before `for` transforms.

```python
words = ["Apple", "bob", "Cat", "apple"]
print([w.lower() for w in words if len(w) > 3])         # prints: ['apple', 'apple']
print([w if w.istitle() else "-" for w in words])       # prints: ['Apple', '-', 'Cat', '-']
print({w.lower(): len(w) for w in words})               # prints: {'apple': 5, 'bob': 3, 'cat': 3}
print({w.lower() for w in words})                       # prints: {'apple', 'bob', 'cat'}  (order varies)
print([(x, y) for x in range(2) for y in "ab"])         # prints: [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]
```

Dict comprehension: later duplicate keys overwrite earlier ones, so the example above keeps the last `apple` length.
Two `for` clauses nest left to right, like nested loops.

Scoping: the comprehension has its **own scope** (the loop variable does not leak, unlike a `for` statement).

```python
[i for i in range(3)]
try:
    print(i)
except NameError:
    print("i is not defined")            # prints: i is not defined
inverted = {v: k for k, v in {"a": 1, "b": 2}.items()}
print(inverted)                          # prints: {1: 'a', 2: 'b'}
```

When to use: a comprehension for simple map/filter into a new collection; a `for` loop when you need side effects,
several statements, or `try/except`. If it needs more than one `for` plus a condition, a loop is more readable.
Use a generator expression (parentheses) when you only need to iterate once (`sum(x for x in ...)`).

## Nested data structures
<!-- topics: nested_structures -->

JSON from an LLM API is nested dicts and lists, and real responses have missing keys, `null`s and variant shapes.
Chained indexing explodes on the first gap; defensive access does not.

```python
resp = {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "search"}}]}}]}

try:
    resp["choices"][0]["message"]["content"].strip()
except AttributeError as e:
    print("null content:", e)   # prints: null content: 'NoneType' object has no attribute 'strip'

msg = (resp.get("choices") or [{}])[0].get("message") or {}
calls = msg.get("tool_calls") or []
print([c["function"]["name"] for c in calls])   # prints: ['search']
print((msg.get("content") or "").strip() == "") # prints: True
```

Note `dict.get(k, {})` only helps for a *missing* key; if the key exists with value `None`, you get `None`, so use
`.get(k) or {}`. A small helper makes the path explicit:

```python
resp = {"choices": [{"message": {"tool_calls": [{"function": {"name": "search"}}]}}]}

def dig(obj, *path, default=None):
    for key in path:
        try:
            obj = obj[key]
        except (KeyError, IndexError, TypeError):
            return default
    return default if obj is None else obj

print(dig(resp, "choices", 0, "message", "tool_calls", 0, "function", "name"))   # prints: search
print(dig(resp, "choices", 5, "message", default="n/a"))                         # prints: n/a
```

Merging nested config needs recursion; `dict.update` and `{**a, **b}` are **shallow** and replace whole subtrees.

```python
base = {"llm": {"model": "a", "temp": 0.2}, "retries": 3}
over = {"llm": {"temp": 0.9}}
print({**base, **over})   # prints: {'llm': {'temp': 0.9}, 'retries': 3}

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

print(deep_merge(base, over))   # prints: {'llm': {'model': 'a', 'temp': 0.9}, 'retries': 3}
```

## Strings and f-strings
<!-- topics: strings, fstrings -->

`str` is an immutable sequence of Unicode code points. Every "modification" returns a new string, so build big
strings with `"".join(parts)` rather than `+=` in a loop (quadratic in the worst case).

```python
s = "  Hello, World  "
print(s.strip().lower())                 # prints: hello, world
print("a,b,,c".split(","))               # prints: ['a', 'b', '', 'c']
print("a  b\n c".split())                # prints: ['a', 'b', 'c']
print("-".join(["x", "y", "z"]))         # prints: x-y-z
print("abc".replace("b", "B"), "abc".startswith(("a", "z")))   # prints: aBc True
print("k=v=w".partition("="))            # prints: ('k', '=', 'v=w')
print("Straße".casefold(), "ß".upper())  # prints: strasse SS
print("naïve".encode("utf-8"))           # prints: b'na\xc3\xafve'
```

Key facts: `.split()` with no argument splits on any whitespace and drops empties; `strip(chars)` removes any of the given
*characters*, not a prefix (use `removeprefix`); `len("naïve")` counts code points, not bytes or tokens. `str` vs `bytes`:
decode bytes at the boundary (`b.decode("utf-8")`), encode when sending.

**f-strings** evaluate expressions inline and accept a format spec after `:`.

```python
name, score, n = "Ada", 0.91234, 1234567
print(f"{name!r} {score:.2f} {n:,} {n:>10} {score:.1%}")
# prints: 'Ada' 0.91 1,234,567    1234567 91.2%
print(f"{name=}", f"{3 + 4}", f"{{literal}}")   # prints: name='Ada' 7 {literal}
print(f"{5:03d} {255:#x} {255:08b}")             # prints: 005 0xff 11111111
```

Spec grammar: `[[fill]align][sign][#][0][width][,][.precision][type]`. `!r` calls `repr` (good for logs: shows quotes and
escapes). `f"{x=}"` prints the expression and value, handy for debugging.

> **Gotcha** Do not build SQL or shell commands with f-strings (injection). For logging use
> `logger.info("x=%s", x)` so formatting is skipped when the level is off. In prompts, escape literal braces as `{{ }}`
> when you also call `.format()`.

> **AI use** Prompt templates are just strings. Normalise first (`strip`, collapse whitespace, Unicode `casefold` or NFKC)
> before counting tokens or hashing for a cache key.

## Slicing and indexing
<!-- topics: slicing -->

Indexing is zero-based; negative indices count from the end. A slice `seq[start:stop:step]` is **half-open**: it
includes `start` and excludes `stop`, and **never raises** for out-of-range bounds (indexing does).

```text
 index   0   1   2   3   4
 value   a   b   c   d   e
 neg    -5  -4  -3  -2  -1
```

```python
s = "abcde"
print(s[1:3], s[:2], s[-2:], s[::2], s[::-1])   # prints: bc ab de ace edcba
print(s[10:], s[-100:2])                        # prints:  ab   (first is an empty string)
try:
    s[10]
except IndexError as e:
    print(e)                                    # prints: string index out of range
nums = list(range(10))
print(nums[2:8:3], nums[-3:])                   # prints: [2, 5] [7, 8, 9]
```

A slice returns a **new** sequence (a shallow copy for lists), which is why `lst[:]` copies. Slice assignment can
resize a list: `nums[2:4] = ["x"]`. Overlapping windows are the building block of chunking:

```python
tokens = list(range(10))
size, overlap = 4, 1
chunks = [tokens[i:i + size] for i in range(0, len(tokens), size - overlap)]
print(chunks[:3], len(chunks))
# prints: [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]] 4
```

The last chunk here is `[9]` (a tail fully covered by overlap); real chunkers decide whether to drop or keep it.

> **Gotcha** Slicing a string by characters may cut a multi-byte emoji sequence or a token. For LLM context limits, slice
> *token lists* produced by the tokenizer, not strings. Also `s[-0:]` is `s[0:]`, the whole string, not the last zero items.

## Quick recap
- Names bind to objects; assignment never copies. Mutating through one alias is visible through all.
- Immutable: int, float, str, tuple, frozenset, None. Tuple immutability is shallow.
- Mutable default arguments are created once at `def` time: default to `None`.
- `==` compares values, `is` compares identity; use `is` only for `None`/singletons.
- Falsy: None, 0, empty containers/strings. `x or default` also replaces valid zeros; test `is None`.
- `and`/`or` return operands; comparisons chain; `//` floors; `int()` truncates; `round` is banker's.
- Loop `else` runs only without `break`; a `for` statement's variable survives the loop, a comprehension's does not.
- Navigate nested JSON defensively (`.get(k) or {}`); merge config recursively, not with shallow `update`.
- Build strings with `join`; f-string specs: `:.2f`, `:,`, `:>10`, `!r`, `=`.
- Slices are half-open, forgiving, and copy; `[::-1]` reverses.
