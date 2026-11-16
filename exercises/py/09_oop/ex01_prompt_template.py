"""
Exercise 09.1 - PromptTemplate: instance vs class state, three kinds of methods   level: ★☆☆
topics: oop.classes_objects, oop.init, oop.instance_vars, oop.class_vars, oop.instance_methods, oop.class_methods, oop.static_methods

Problem
-------
Write `PromptTemplate`, a small class for reusable LLM prompts:

    qa = PromptTemplate("qa", "Context: {context} Question: {question}", tags=["rag"])
    qa.render(context="...", question="...")

* __init__(name, template, *, tags=None)
  - empty name -> ValueError
  - instance variables: name, template, tags (a NEW list per instance - copy the argument;
    two templates must never share one tags list)
  - count every template ever created in the CLASS variable `PromptTemplate.created`
    (instances of subclasses count too, and they must update PromptTemplate.created - not
    create a separate counter on the subclass or the instance).
* variables(template) - STATIC method -> set of placeholder names in a template string,
  e.g. {"context", "question"}. Use string.Formatter().parse; ignore literal "{{" "}}".
  Callable as PromptTemplate.variables("...") and qa.variables("...").
* render(**values) - instance method -> the formatted string.
  Missing placeholders -> ValueError whose message lists the missing names sorted and
  comma-separated, e.g. "missing variables: context, question". Extra keys are ignored.
* from_dict(data) - CLASS method: build from {"name": ..., "template": ..., "tags": [...]}
  ("tags" optional). Called on a subclass it must return an instance of that subclass.
* __contains__(name) -> True if `name` is a placeholder:  "question" in qa
* __repr__ -> "PromptTemplate(name='qa', variables=['context', 'question'])"
  (variables sorted; use the real class name, so a subclass shows its own name).

Interviewers look for
---------------------
* Class variables are shared by all instances; `self.created += 1` would create an
  INSTANCE attribute shadowing the class one (classic bug). Mutable class-level defaults
  (`tags = []`) are shared by every instance (another classic bug).
* instance method (self) vs classmethod (cls - alternative constructors, works with
  subclasses) vs staticmethod (no self/cls - a utility that belongs with the class).
* __init__ initialises an already-created object; it doesn't create it (__new__ does).
"""

from collections.abc import Iterable
from typing import Any


class PromptTemplate:
    created = 0

    def __init__(self, name: str, template: str, *, tags: Iterable[str] | None = None) -> None:
        raise NotImplementedError

    @staticmethod
    def variables(template: str) -> set[str]:
        raise NotImplementedError

    def render(self, **values: Any) -> str:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        raise NotImplementedError

    def __contains__(self, name: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
