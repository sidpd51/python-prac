"""
Exercise 09.4 - dataclasses for model configs and eval results    level: ★★☆
topics: oop.dataclasses, oop.class_methods, oop.slots, oop.init

Problem
-------
The field declarations below are part of the spec - implement the methods.

1. ModelConfig - @dataclass(frozen=True, slots=True): name, temperature=0.7,
   max_tokens=1024, stop=() (a tuple of stop sequences).
   * __post_init__ validates: empty name -> ValueError; temperature outside [0, 2] ->
     ValueError; max_tokens < 1 -> ValueError. It also NORMALISES `stop`: a list (or any
     iterable of str) is converted to a tuple, so configs stay hashable. The instance is
     frozen - find the sanctioned way to set a field inside __post_init__.
   * from_dict(data) - classmethod. Unknown keys -> TypeError listing them sorted (don't let
     a typo like "temprature" be silently ignored). Use dataclasses.fields().
   * from_env(environ, prefix="LLM_") - classmethod, reads e.g.
         LLM_NAME="gpt-x" LLM_TEMPERATURE="0.2" LLM_MAX_TOKENS="256" LLM_STOP="###,END"
     converting types (float / int / comma-separated tuple; empty LLM_STOP -> ()).
     Missing variables use the defaults; missing NAME -> KeyError. `environ` is any
     mapping (pass os.environ in real code; tests pass a dict).
   * with_overrides(**changes) -> a NEW validated ModelConfig (dataclasses.replace).

2. EvalResult - @dataclass(order=True, frozen=True): name, score, latency_ms=0.0, plus a
   `sort_key` field that is not an __init__ parameter, not in the repr and is the ONLY
   field used for comparisons. __post_init__ sets sort_key = (-score, latency_ms, name), so
   sorted(results) puts the best first: highest score, then the fastest, then by name.

3. EvalRun - a regular @dataclass: config: ModelConfig, results: list[EvalResult] and
   tags: dict[str, str], both defaulting to a fresh empty container per run.
   * add(result) appends; leaderboard(n=None) -> the n best results (all if None);
   * mean_score property -> average score; no results -> ValueError.

Interviewers look for
---------------------
* Why `results: list = []` is rejected by dataclasses and `field(default_factory=list)`
  is the fix (shared mutable default).
* frozen=True -> immutability + generated __hash__ (with eq=True): safe dict keys/cache keys.
  object.__setattr__ inside __post_init__ is the escape hatch.
* slots=True: memory + typo protection; order=True compares fields as tuples in
  definition order - control it with field(compare=False) or a sort key.
* classmethod alternative constructors (from_dict / from_env) keep __init__ simple.
* dataclass vs NamedTuple vs pydantic (runtime validation/parsing at boundaries).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        raise NotImplementedError

    @classmethod
    def from_env(cls, environ: Mapping[str, str], prefix: str = "LLM_") -> "ModelConfig":
        raise NotImplementedError

    def with_overrides(self, **changes: Any) -> "ModelConfig":
        raise NotImplementedError


@dataclass(order=True, frozen=True)
class EvalResult:
    sort_key: tuple = field(init=False, repr=False)
    name: str = field(compare=False)
    score: float = field(compare=False)
    latency_ms: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        raise NotImplementedError


@dataclass
class EvalRun:
    config: ModelConfig
    results: list[EvalResult] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def add(self, result: EvalResult) -> None:
        raise NotImplementedError

    def leaderboard(self, n: int | None = None) -> list[EvalResult]:
        raise NotImplementedError

    @property
    def mean_score(self) -> float:
        raise NotImplementedError
