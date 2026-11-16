"""
Exercise 09.2 - An LLM provider hierarchy with an abstract base class   level: ★★☆
topics: oop.abc, oop.abstraction, oop.inheritance, oop.polymorphism, oop.overriding, oop.property, oop.encapsulation

Problem
-------
Application code should depend on an abstraction, not on a vendor SDK.

1. BaseLLM(ABC)
   * __init__(model, temperature=0.0): empty model -> ValueError. Store model, and set
     temperature THROUGH the property (so validation runs in __init__ too).
   * temperature - a property backed by `_temperature`; the setter validates:
     bool or non-number -> TypeError; outside [0.0, 2.0] -> ValueError.
   * abstract methods (subclasses MUST implement; BaseLLM itself can't be instantiated):
         complete(prompt: str) -> str
         count_tokens(text: str) -> int
   * generate(prompt) -> Completion  (concrete "template method", don't override it):
     strip the prompt; empty -> ValueError; otherwise call self.complete(prompt) and return
     Completion(text, model, prompt_tokens=count_tokens(prompt),
                completion_tokens=count_tokens(text)).
   * __repr__ -> "EchoLLM(model='echo-1', temperature=0.0)" (real class name).
   * from_config(config) - classmethod alternative constructor:
       - called on BaseLLM: pick the class from config["provider"] via the PROVIDERS dict
         (unknown provider -> ValueError)
       - called on a concrete subclass: build that class (config["provider"] optional)
       - every other key is passed to the constructor as a keyword argument.

2. EchoLLM(BaseLLM): complete returns the prompt unchanged;
   count_tokens = number of whitespace-separated words.

3. ShoutLLM(BaseLLM): __init__(model, temperature=0.0, suffix="!") - OVERRIDES __init__,
   reuses the base one via super(); complete returns prompt.upper() + suffix;
   count_tokens = ceil(len(text) / 4)  (the "4 chars per token" rule of thumb).

4. run_all(llms, prompt) -> dict[str, str]
   {repr(llm): llm.generate(prompt).text for each llm} - polymorphism: no isinstance
   checks / type switches; it must work with any object that has generate() (duck typing).

Completion and PROVIDERS are given.

Interviewers look for
---------------------
* ABCs enforce an interface at instantiation time (TypeError), not at call time.
* Template method pattern: the base class owns the algorithm (validate, call, count),
  subclasses fill in the steps.
* Encapsulation via @property: validation without changing the public attribute API.
* classmethod constructors that respect `cls` (factory / registry pattern).
* ABC (nominal) vs typing.Protocol (structural / duck typing) - which one to choose when.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__stub_keep__ = ["Completion"]


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class BaseLLM(ABC):
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        self.model = model
        self.temperature = temperature  # goes through the property setter

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("temperature must be a number")
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        self._temperature = float(value)

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's text for `prompt`."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in `text`."""

    def generate(self, prompt: str) -> Completion:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        text = self.complete(prompt)
        return Completion(text, self.model, self.count_tokens(prompt), self.count_tokens(text))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, temperature={self.temperature})"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseLLM":
        kwargs = dict(config)
        provider = kwargs.pop("provider", None)
        target = cls
        if cls is BaseLLM:
            if provider not in PROVIDERS:
                raise ValueError(f"unknown provider: {provider!r}")
            target = PROVIDERS[provider]
        return target(**kwargs)


class EchoLLM(BaseLLM):
    def complete(self, prompt: str) -> str:
        return prompt

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class ShoutLLM(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.0, suffix: str = "!") -> None:
        super().__init__(model, temperature)
        self.suffix = suffix

    def complete(self, prompt: str) -> str:
        return prompt.upper() + self.suffix

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / 4)


PROVIDERS: dict[str, type[BaseLLM]] = {"echo": EchoLLM, "shout": ShoutLLM}


def run_all(llms: Iterable[Any], prompt: str) -> dict[str, str]:
    return {repr(llm): llm.generate(prompt).text for llm in llms}
