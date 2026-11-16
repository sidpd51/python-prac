"""
Exercise 05.3 - Fix three scoping bugs from code review           level: ★★☆
topics: closures.unboundlocalerror, closures.local_global, closures.lexical_scoping, closures.closures

Problem
-------
Each function below shipped with a scoping bug. Write the fixed version (same name,
same signature, behaviour described under "Expected").

1. record_request(model) - counts LLM requests in module-level REQUEST_COUNTS.
       REQUEST_TOTAL = 0
       def record_request(model):
           REQUEST_COUNTS[model] = REQUEST_COUNTS.get(model, 0) + 1
           REQUEST_TOTAL += 1                # UnboundLocalError!
           return REQUEST_TOTAL
   Expected: increments the module-level REQUEST_TOTAL and the per-model count, returns the
   new REQUEST_TOTAL. Keep using the module globals (that's the point of this one).
   (Why does the dict line work while the int line fails?)

2. pick_model(context_tokens) - chooses a model name.
       def pick_model(context_tokens):
           if context_tokens > LONG_CONTEXT_THRESHOLD:
               DEFAULT_MODEL = "long-context-model"
           return DEFAULT_MODEL              # UnboundLocalError for short prompts!
   Expected: return LONG_CONTEXT_MODEL when context_tokens > LONG_CONTEXT_THRESHOLD,
   otherwise DEFAULT_MODEL. Must NOT modify any global.

3. make_formatters(templates) - one formatter function per template.
       def make_formatters(templates):
           return [lambda text: t.format(text=text) for t in templates]
   Every formatter uses the LAST template (late binding). Expected: formatter i uses
   templates[i]:
       fs = make_formatters(["Q: {text}", "Summarize: {text}"])
       fs[0]("hi") -> "Q: hi" ; fs[1]("hi") -> "Summarize: hi"
   Formatters must take exactly ONE positional parameter (so a caller can't accidentally
   override the template - i.e. don't use the `t=t` default-argument trick here; use a
   factory function instead).

Interviewers look for
---------------------
* The compiler decides a name is local if it is ASSIGNED anywhere in the function body
  (+=, =, for-target, import, def...). Reading it before the assignment -> UnboundLocalError.
* Mutating a global object (dict item) is not rebinding the name, so it needs no `global`.
* Closures capture VARIABLES, not values; the lookup happens when the inner function runs.
* LEGB: Local, Enclosing, Global, Builtins - decided lexically (where code is written).
"""

from collections.abc import Callable

REQUEST_COUNTS: dict[str, int] = {}
REQUEST_TOTAL = 0

DEFAULT_MODEL = "small-fast-model"
LONG_CONTEXT_MODEL = "long-context-model"
LONG_CONTEXT_THRESHOLD = 8000


def record_request(model: str) -> int:
    global REQUEST_TOTAL
    REQUEST_COUNTS[model] = REQUEST_COUNTS.get(model, 0) + 1  # item mutation: no `global` needed
    REQUEST_TOTAL += 1
    return REQUEST_TOTAL


def pick_model(context_tokens: int) -> str:
    # a distinct local name: assigning DEFAULT_MODEL here would make it local everywhere
    model = DEFAULT_MODEL
    if context_tokens > LONG_CONTEXT_THRESHOLD:
        model = LONG_CONTEXT_MODEL
    return model


def make_formatters(templates: list[str]) -> list[Callable[[str], str]]:
    def make_one(template: str) -> Callable[[str], str]:
        # each make_one() call creates a new cell holding THIS template
        def fmt(text: str) -> str:
            return template.format(text=text)

        return fmt

    return [make_one(t) for t in templates]
