# Type hints

Type hints are annotations that say what types a value is meant to have. Python itself **ignores them at runtime**:
they are metadata for humans, editors and static checkers such as mypy and Pyright. Used well they catch bugs before
you run the code, make large codebases navigable, and are the foundation of pydantic, FastAPI and the typed message
and tool schemas of LLM SDKs. Interviews check that you know they are not enforced, can read generics, `Protocol`,
`TypedDict`, `Optional`, and can explain what a type checker actually does.

## Basic annotations
<!-- topics: basic -->

Syntax: `name: type = value` for variables, `def f(x: type) -> type:` for functions. Python stores the annotation
and does nothing else with it.

```python
count: int = 0
name: str
ratio: float = 0.5

x: int = "not an int"          # no error at runtime!
print(x)                       # prints: not an int
print(__annotations__)         # prints: {'count': <class 'int'>, 'name': <class 'str'>, 'ratio': <class 'float'>, 'x': <class 'int'>}
```

`name: str` with no value only records the annotation; the variable still does not exist. The checker (mypy) would
flag `x: int = "..."`, the interpreter does not.

> **Remember** Annotations are not validation. If you want runtime enforcement, use pydantic or explicit checks.
> Static typing finds bugs in code paths *before* they run; it cannot see the data that arrives from JSON or an LLM.

Basic types: `int`, `float`, `str`, `bool`, `bytes`, `None`. Note `bool` is a subclass of `int`, and mypy accepts an
`int` where `float` is expected (the numeric tower shortcut), but not the reverse.

**Python 3.14 (PEP 649/749):** annotations are no longer evaluated eagerly at definition time; they are evaluated
lazily on first access. This makes forward references (a class naming itself) work without quotes.

```python
# norun
# On 3.14 this works, on 3.13 it raises NameError at definition time:
def clone(node: Node) -> Node: ...
class Node: ...

import annotationlib
print(annotationlib.get_annotations(clone, format=annotationlib.Format.STRING))
# 3.14 prints: {'node': 'Node', 'return': 'Node'}
```

On 3.13 and earlier use a quoted `"Node"` or `from __future__ import annotations` (which stores every annotation as a
string). That import still works on 3.14 but is no longer needed.

## Function annotations and returns
<!-- topics: functions, returns -->

```python
def chunk(text: str, size: int = 500, *, overlap: int = 0) -> list[str]:
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]

print(chunk("abcdef", 4, overlap=1))     # prints: ['abcd', 'def']
print(chunk.__annotations__)
# prints: {'text': <class 'str'>, 'size': <class 'int'>, 'overlap': <class 'int'>, 'return': list[str]}
```

Annotate parameters and the return type. `*args: int` means each extra positional is an `int` (the tuple itself is
`tuple[int, ...]`); `**kwargs: str` means each value is a `str`. Default values go after the annotation with spaces
around `=`.

**Return types**
* `-> None` for functions that return nothing. A function with no `return` and no annotation is "unchecked" by
  default in mypy, so annotate `-> None` explicitly.
* `-> NoReturn` (or `Never`) for functions that never return normally (always raise or `sys.exit`).
* Generators: `-> Iterator[int]` or `Generator[YieldType, SendType, ReturnType]`; async: `async def f() -> int`
  (the coroutine type is implied; do not write `Coroutine`).
* Methods returning the instance: `-> Self` (3.11+, `typing.Self`) rather than naming the class.

```python
from typing import Iterator, NoReturn

def countdown(n: int) -> Iterator[int]:
    while n:
        yield n
        n -= 1

def fail(msg: str) -> NoReturn:
    raise RuntimeError(msg)

print(list(countdown(3)))    # prints: [3, 2, 1]
```

> **Gotcha** Mutable defaults are still a bug with hints: `def f(xs: list[int] = [])`. Use `xs: list[int] | None = None`.

## Optional and Union
<!-- topics: optional, union -->

`X | Y` (3.10+) means "either type". `Optional[X]` is exactly `X | None`. The name is misleading: it does **not**
mean the *argument* is optional (that is a default value), it means the *value* may be `None`.

```python
from typing import Optional, Union

def find(key: str) -> str | None:          # modern spelling of Optional[str]
    return {"a": "x"}.get(key)

def f(x: Optional[int] = None): ...        # equal to int | None
assert Optional[int] == Union[int, None] == (int | None)

print(find("b"))                           # prints: None
```

The checker's job: it forces you to **narrow** before using the value, which removes a whole class of
`AttributeError: 'NoneType' ...` bugs.

```python
def shout(s: str | None) -> str:
    if s is None:
        return ""
    return s.upper()        # here the checker knows s is str

def size(x: int | str) -> int:
    if isinstance(x, int):  # isinstance narrows unions
        return x
    return len(x)

print(shout(None) == "", size("abc"))   # prints: True 3
```

Narrowing works with `is None`, `isinstance`, truthiness, `in`, and `Literal` comparisons. Without narrowing, mypy
errors on `s.upper()` because `None` has no `upper`.

> **Interview** "What is `Optional`?" Shorthand for `X | None`, not "has a default". Pair it with narrowing.
> Returning `None` as an error signal is a common source of unhandled `Optional`s; consider raising instead.

## Collections: list, dict, tuple, set
<!-- topics: collections -->

Since 3.9 use the built-in types directly, subscripted. The old `typing.List`, `Dict`, `Tuple`, `Set` are deprecated
aliases. Prefer abstract types in **parameters** (`Sequence`, `Mapping`, `Iterable`) and concrete types in returns.

```python
from collections.abc import Iterable, Mapping, Sequence

scores: list[float] = [0.1, 0.9]
index: dict[str, list[int]] = {"a": [1, 2]}
point: tuple[float, float] = (1.0, 2.0)        # exactly two floats
vec: tuple[int, ...] = (1, 2, 3)               # any length, all int
tags: set[str] = {"x"}

def total(xs: Iterable[float]) -> float:       # accepts list, tuple, generator, set...
    return sum(xs)

def lookup(m: Mapping[str, int], k: str) -> int:   # read-only view; accepts dict, MappingProxy...
    return m[k]

print(total(x for x in (1.0, 2.0)), lookup({"a": 1}, "a"))   # prints: 3.0 1
```

Why `Iterable` in parameters: it tells callers "any iterable works", while `list[float]` needlessly rejects tuples
and generators. Why `tuple[int, ...]`: a plain `tuple[int]` means a tuple of exactly one int. Invariance trap:
`list[int]` is **not** a `list[float]` (you could append a float to it), so a function taking `list[float]` rejects a
`list[int]`; `Sequence[float]` (read-only, covariant) accepts it.

> **AI use** `list[dict[str, str]]` for chat messages is common, but a `TypedDict` (below) catches key typos.
> Embeddings are `list[float]` or `numpy.typing.NDArray[np.float32]`.

## Any versus object
<!-- topics: any_vs_object -->

Both accept anything, but they mean opposite things to the checker.

| | `Any` | `object` |
|---|---|---|
| Assignable *from* | everything | everything |
| What you may *do* with it | **everything** (checker turns itself off) | only what every object supports |
| Use | escape hatch, untyped third-party data | "anything, but I will check first" |

```python
from typing import Any

def f(x: Any) -> None:
    x.foo()               # checker allows it: Any is unchecked (fails at runtime)

def g(x: object) -> str:
    # x.foo()  <- mypy error: "object" has no attribute "foo"
    if isinstance(x, str):
        return x.upper()  # narrowed to str, safe
    return repr(x)

print(g("hi"), g(3))      # prints: HI 3
```

`Any` is contagious: a value of type `Any` flows through and silences checks downstream. Use `object` (plus
narrowing) or a union when you are really accepting arbitrary input, and keep `Any` for genuinely dynamic
spots such as `json.loads` results, ideally narrowed or validated right away. Unannotated parameters are
implicitly `Any`, which is why partially typed code gives weaker guarantees.

> **Interview** "`Any` vs `object`?" `Any` opts out of checking in both directions; `object` is the top type
> and forces you to narrow before use.

## Callable
<!-- topics: callable -->

`Callable[[ArgTypes], ReturnType]` types a function parameter. Import from `collections.abc`.

```python
from collections.abc import Callable

def apply(fn: Callable[[int, int], int], a: int, b: int) -> int:
    return fn(a, b)

def make_handler(prefix: str) -> Callable[[str], str]:    # returns a function
    return lambda s: prefix + s

print(apply(lambda x, y: x + y, 2, 3))    # prints: 5
print(make_handler("> ")("hi"))           # prints: > hi
```

* `Callable[..., int]`: any arguments, returns `int`.
* `Callable[[], None]`: no arguments (a callback).
* It cannot express keyword arguments or defaults; for that use a **callback `Protocol`** with `__call__`, or
  `ParamSpec` when writing decorators that preserve the wrapped function's signature.
* Async callbacks: `Callable[[str], Awaitable[str]]`.

> **AI use** Tool registries for LLM agents map names to `Callable[..., Any]`; typed decorators with `ParamSpec`
> keep IDE autocompletion on the decorated tool.

## TypeVar and generics
<!-- topics: typevar, generics -->

A **TypeVar** links types together: "whatever type goes in comes out". It lets a function or class be generic
without losing information (which `Any` would lose).

```python
from typing import TypeVar

T = TypeVar("T")

def first(xs: list[T]) -> T:
    return xs[0]

a = first([1, 2])        # checker infers int
b = first(["x"])         # checker infers str
print(a, b)              # prints: 1 x
```

With `Any` the return type would be unknown; with `T` it is exactly the element type. Constrain with
`TypeVar("N", int, float)` (one of these exactly) or a **bound**, `TypeVar("C", bound=Comparable)` (this type or
a subtype).

**Generic classes** parametrise a container by type. Python 3.12+ has the compact syntax (PEP 695):

```python
class Box[T]:                       # 3.12+: no TypeVar declaration needed
    def __init__(self, item: T) -> None:
        self.item = item
    def get(self) -> T:
        return self.item

def pair[K, V](k: K, v: V) -> tuple[K, V]:      # generic function, 3.12+
    return (k, v)

print(Box(3).get(), pair("a", 1))   # prints: 3 ('a', 1)
```

Older equivalent: `class Box(Generic[T])` with `T = TypeVar("T")`. Reading `Box[int]` in an annotation means
"a Box holding ints". At runtime generics are **erased**: `Box[int]()` is just `Box()`, and `list[int]` does not
check its elements.

```python
class Box[T]:
    def __init__(self, item: T) -> None:
        self.item = item

print(type(Box[int](3)).__name__)        # prints: Box   (Box[int] is just Box at runtime)
print(type(Box[bytes](1)).__name__)  # prints: Box   (nothing checked)
```

Variance in one line: producers (`Sequence[T]`, read-only) are covariant, consumers are contravariant, mutable
containers (`list[T]`) are invariant.

<details markdown="1"><summary>Check yourself: why is `def first(xs: list[Any]) -> Any` worse than the TypeVar version?</summary>

The checker loses the connection between input and output: callers get `Any` back and every later use of the result
goes unchecked. With `T`, `first([1, 2])` is known to be `int`.

</details>

## Protocol (structural typing)
<!-- topics: protocol -->

Normal classes use **nominal** typing: B is an A only if it inherits from A. A **Protocol** uses **structural**
typing ("duck typing, checked statically"): anything with the right methods qualifies, no inheritance needed.

```python
from typing import Protocol

class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

class FakeEmbedder:                    # does NOT inherit from Embedder
    def embed(self, text: str) -> list[float]:
        return [float(len(text))]

def index(docs: list[str], model: Embedder) -> list[list[float]]:
    return [model.embed(d) for d in docs]

print(index(["ab", "abc"], FakeEmbedder()))   # prints: [[2.0], [3.0]]
```

The checker accepts `FakeEmbedder` because its shape matches. That makes dependency injection and test doubles easy
and decouples you from third-party class hierarchies. Compare with an ABC, which forces inheritance and is checked
at instantiation time at runtime.

At runtime a plain `Protocol` does nothing; add `@runtime_checkable` to allow `isinstance(x, Embedder)`, which only
checks that the *method names exist* (not their signatures).

> **Interview** "Protocol vs ABC?" Protocol: structural, static, no inheritance. ABC: nominal, runtime-enforced,
> can share implementation. Use Protocols for "anything that can embed/retrieve/call".

## TypedDict
<!-- topics: typeddict -->

`TypedDict` types a **dict with fixed string keys** and a type per key, the shape of JSON and LLM API payloads.
At runtime it is a plain `dict`; the checking is static only.

```python
from typing import TypedDict, NotRequired

class Message(TypedDict):
    role: str
    content: str
    name: NotRequired[str]          # key may be absent

m: Message = {"role": "user", "content": "hi"}
print(m, type(m).__name__)          # prints: {'role': 'user', 'content': 'hi'} dict

bad: Message = {"role": "user"}     # no error at runtime; mypy: missing key "content"
print(bad)                          # prints: {'role': 'user'}
```

| Use | When |
|---|---|
| `TypedDict` | data already is a dict (JSON in/out), zero conversion cost, static help only |
| `dataclass` | internal objects with behaviour; attribute access |
| pydantic `BaseModel` | untrusted input that needs **runtime validation** and parsing |

`total=False` makes all keys optional; `Required`/`NotRequired` mark individual keys; `ReadOnly` (3.13) forbids
mutation. Typo'd keys (`m["rol"]`) become checker errors instead of `KeyError`s in production.

> **AI use** Messages lists `list[Message]` and tool-call JSON schemas are classic TypedDicts. The Anthropic and
> OpenAI SDK request params are TypedDicts for exactly this reason.

## Literal
<!-- topics: literal -->

`Literal["a", "b"]` restricts a value to specific constants. It is the lightweight enum for string options.

```python
from typing import Literal, get_args

Role = Literal["system", "user", "assistant"]

def make(role: Role, content: str) -> dict[str, str]:
    return {"role": role, "content": content}

make("user", "hi")
make("bot", "hi")                    # runs fine, but mypy: Argument 1 has incompatible type
print(get_args(Role))                # prints: ('system', 'user', 'assistant')
```

The checker narrows on equality and supports exhaustiveness: with a chain of `if role == "..."` covering every
literal, `assert_never(role)` in the `else` fails type-checking if you add a new option and forget a branch. Use it
for model names, roles, modes, `Literal[True]`/`Literal[False]` overloads. Pair with `get_args` to reuse the list of
allowed values at runtime. For richer sets with behaviour, use `enum.Enum`.

## Type aliases
<!-- topics: aliases -->

An alias gives a long type a name. Use it to keep signatures readable and give one definition to reuse.

```python
from typing import TypeAlias

Vector: TypeAlias = list[float]                 # works on all 3.10+
Messages = list[dict[str, str]]                 # simple assignment also works
JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None   # recursive

def norm(v: Vector) -> float:
    return sum(x * x for x in v) ** 0.5

print(norm([3.0, 4.0]))                          # prints: 5.0
```

3.12+ has the `type` statement (PEP 695), which is lazily evaluated and so supports recursion cleanly:

```python
type Json = dict[str, Json] | list[Json] | str | int | float | bool | None
type Vector = list[float]

print(Vector)                    # prints: Vector
print(Vector.__value__)          # prints: list[float]
```

Distinguish alias from `NewType("UserId", int)`, which creates a **distinct** type for the checker (an `int` is not
a `UserId`) with zero runtime cost (it is just a function returning its argument). Use `NewType` to stop mixing up
`DocId` and `ChunkId`.

## Static type checking
<!-- topics: static_checking -->

**Static** means "without running the code". A checker reads source, builds the type of every expression and
reports mismatches. It never imports your modules into a running program and never sees runtime values.

```text
source -> parse -> infer & check types per annotations -> report errors
                                                          (no code executed)
```

What it **can** catch: calling with wrong argument types, forgetting to handle `None`, misspelled attributes, wrong
return types, missing `TypedDict` keys, non-exhaustive branches.
What it **cannot** catch: bad runtime data (a JSON field of the wrong type), logic errors, anything hidden behind
`Any` or `# type: ignore`, dynamic attribute magic.

```python
def add_one(n: int) -> int:
    return n + 1

print(add_one(5))              # prints: 6
if False:
    add_one("5")               # mypy reports this line; Python never runs it, so no error
```

Gradual typing: untyped code is allowed, so you adopt types incrementally and tighten over time (`--strict`).
Common escape hatches: `# type: ignore[code]`, `cast(T, x)` (a lie the checker believes; no runtime check), `Any`.
Prefer fixing the types; justify every ignore.

> **Gotcha** `cast` and `# type: ignore` do not change runtime behaviour or verify anything. The bug just moves to
> runtime. Use `isinstance` narrowing or validate with pydantic at the boundary instead.

> **Remember** Types describe what your code *expects*. Validate what the outside world *sends*.

## mypy and Pyright
<!-- topics: mypy_pyright -->

The two main checkers. Both read `pyproject.toml` config and understand the same typing PEPs.

| | mypy | Pyright |
|---|---|---|
| Written in | Python | TypeScript (Node) |
| Speed | slower, has an incremental cache and daemon | fast |
| Editor | plugin or `dmypy` | powers **Pylance** in VS Code |
| Inference | more conservative about unannotated code | stronger inference and narrowing |
| Strictness | `--strict` | `strict` mode / `basic` / `standard` |

```bash
python -m mypy src/             # run on a directory
python -m mypy --strict src/    # enable the strict flag bundle
pyright src/
```

```toml
[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true      # for libs without type information

[tool.pyright]
typeCheckingMode = "standard"
```

Libraries ship types via inline annotations plus a `py.typed` marker file (PEP 561), or as separate `types-*` stub
packages (`types-requests`). Run the checker in CI and your pre-commit hook; a typed project with a green checker
turns refactors from guesswork into mechanical edits. Newer entrants (Astral's `ty`, Meta's `pyrefly`) exist and
are fast, but mypy and Pyright are what interviewers name.

> **Interview** "What does mypy do at runtime?" Nothing: it is not involved when your program runs, and the
> interpreter ignores annotations. Which is why you still need tests and runtime validation.

## Quick recap
- Annotations are metadata ignored by the interpreter; checkers and tools like pydantic use them.
- Annotate parameters and returns; `-> None`, `NoReturn`, `Iterator[T]`, `Self` for common cases.
- `Optional[X]` is `X | None`; narrow with `is None` / `isinstance` before use.
- Use built-in generics (`list[int]`); take abstract types (`Iterable`, `Mapping`) and return concrete ones.
- `Any` turns checking off; `object` is the safe top type that forces narrowing.
- `Callable[[Args], Ret]` for function params; use a callback Protocol or `ParamSpec` for richer signatures.
- `TypeVar` links input and output types; `class Box[T]` is the 3.12+ syntax; generics are erased at runtime.
- `Protocol` is structural (no inheritance); `TypedDict` types dict payloads (static only); `Literal` restricts values.
- Aliases name types (`type X = ...` on 3.12+); `NewType` makes distinct types.
- mypy and Pyright check statically (no code runs); they cannot validate runtime data.
- 3.14 evaluates annotations lazily (PEP 649), so forward references work unquoted.
