# Object-Oriented Programming

Classes bundle data with the behaviour that works on it. In Python they are also ordinary runtime objects,
and a handful of mechanisms (attribute lookup, descriptors, the MRO, dunder methods) explain nearly every
"why does this print that?" question. In AI engineering you use OOP to put several LLM providers behind
one interface, to model messages and configs (dataclasses or pydantic), and to build clients that hold
state such as retry settings, a session or a token budget. Interviewers rarely ask for syntax; they ask for
the *traps*: mutable class variables, `super()` and the MRO, `__eq__` without `__hash__`, shallow copies, and
inheritance versus composition.

## Classes, objects and __init__
<!-- topics: classes_objects, init, instance_vars -->

A **class** is a blueprint and itself an object (an instance of `type`). Calling it creates an **instance**:
`__new__` allocates the object, then `__init__` initialises it. `__init__` is *not* the constructor in the
strict sense (it returns `None` and receives an already created object); `__new__` is rarely overridden (only for
immutable types or singletons).

```python
class Message:
    def __init__(self, role, content):
        self.role = role              # instance variables live in the instance's __dict__
        self.content = content

m = Message("user", "hi")
print(m.__dict__)                     # prints: {'role': 'user', 'content': 'hi'}
print(isinstance(m, Message), type(Message).__name__)   # prints: True type
```

**Instance variables** are created by assigning on `self` (usually in `__init__`) and belong to one
object. Two instances never share them. Python lets you add attributes later (`m.extra = 1`), which is
flexible but typo-prone; `__slots__` or dataclasses tighten this up.

> **Remember** `self` is not a keyword; it is the first parameter, filled in automatically when you call a
> method on an instance. `m.f(x)` is `Message.f(m, x)`.

> **Gotcha** A mutable default argument is shared by every call, including constructor calls.
> `def __init__(self, tags=[])` makes every object share one list. Use `tags=None` and copy (`self.tags = list(tags or [])`).

## Class variables and the lookup rule
<!-- topics: class_vars -->

A **class variable** is assigned in the class body and shared by all instances. Reading `obj.x` looks in the
instance's `__dict__` first, then the class, then base classes. *Writing* `obj.x = ...` always creates or updates an
**instance** attribute and never touches the class.

```python
class Counter:
    created = 0                      # class variable: shared
    def __init__(self):
        Counter.created += 1         # update via the class name

a, b = Counter(), Counter()
print(Counter.created, a.created)    # prints: 2 2

a.created = 99                       # creates an INSTANCE attribute shadowing the class one
print(a.created, b.created, Counter.created)   # prints: 99 2 2
```

The big trap is a **mutable class variable**, which every instance mutates in common:

```python
class Agent:
    history = []                     # shared by ALL agents
    def say(self, text):
        self.history.append(text)    # mutates the shared list, no new attribute created

x, y = Agent(), Agent()
x.say("hello")
print(y.history)                     # prints: ['hello']
```

Fix: create the list in `__init__` (`self.history = []`). Class variables are for constants, defaults of
immutable values and counters. Note `self.count += 1` on an `int` class variable *rebinds* an instance
attribute (reads the class value, writes to the instance), so use `ClassName.count += 1` for a shared counter;
inside a subclass, `type(self).count += 1` would create a separate counter on the subclass.

## Instance, class and static methods
<!-- topics: instance_methods, class_methods, static_methods -->

| Kind | First argument | Typical use |
|---|---|---|
| instance method | `self` (the instance) | behaviour that uses/changes instance state |
| `@classmethod` | `cls` (the class, possibly a subclass) | alternative constructors, class-level state |
| `@staticmethod` | nothing | a related utility function that needs neither |

```python
class PromptTemplate:
    def __init__(self, name, template):
        self.name, self.template = name, template

    def render(self, **values):                  # instance method
        return self.template.format(**values)

    @classmethod
    def from_dict(cls, data):                    # alternative constructor
        return cls(data["name"], data["template"])

    @staticmethod
    def variables(template):                     # no self, no cls
        import string
        return {f for _, f, _, _ in string.Formatter().parse(template) if f}

t = PromptTemplate.from_dict({"name": "qa", "template": "Q: {question}"})
print(t.render(question="why?"))                 # prints: Q: why?
print(PromptTemplate.variables("{a} and {b}") == {"a", "b"})   # prints: True
```

Why `cls` and not the hard-coded class name? A subclass inherits the classmethod and it builds **the subclass**
(`ChatTemplate.from_dict(...)` returns a `ChatTemplate`).

Under the hood, functions are **descriptors**: `obj.method` triggers `function.__get__`, which returns a
*bound method* with `self` pre-filled. `classmethod` and `staticmethod` are descriptor wrappers that bind the
class or nothing.

## Encapsulation and properties
<!-- topics: encapsulation, property -->

Python has no `private`; it relies on convention and a little name mangling.

* `_x`: "internal, do not touch" (convention only; also skipped by `from m import *`).
* `__x`: **name mangled** to `_ClassName__x` inside the class, to avoid accidental clashes in subclasses.
  It is a collision guard, not security.
* No prefix: public API.

```python
class Account:
    def __init__(self):
        self._balance = 0
        self.__secret = "s"

a = Account()
print(a._balance)                        # prints: 0   (allowed, but you were warned)
print(a._Account__secret)                # prints: s   (mangled name still reachable)
print(hasattr(a, "__secret"))            # prints: False
```

Do not write Java-style `get_x()`/`set_x()`. Start with a plain public attribute; if you later need
validation or computation, turn it into a **property** without changing callers' code
(`obj.x` stays `obj.x`).

```python
class LLM:
    def __init__(self, temperature=0.0):
        self.temperature = temperature            # goes through the setter, so it is validated

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("temperature must be a number")
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        self._temperature = float(value)

m = LLM(1)
print(m.temperature)                              # prints: 1.0
try:
    m.temperature = 5
except ValueError as e:
    print(e)                                      # prints: temperature must be in [0, 2]
```

How does it work? `property` is a **data descriptor**: an object stored on the *class* with `__get__` and
`__set__`. When you access `m.temperature`, Python finds the descriptor on the type and calls its `__get__`
before consulting the instance `__dict__`; assignment calls `__set__`. A property with no setter therefore raises `AttributeError` on assignment
(read-only), and it must be defined on the class, not on `self`. Computed read-only values (`area`, `full_name`)
are the other common use.

> **Gotcha** In `__init__`, assign to `self.temperature` (the property), not `self._temperature`, or validation is
> skipped on construction. `functools.cached_property` computes once and stores the result in the instance dict
> (a non-data descriptor, so the instance value wins afterwards).

## Inheritance, overriding and super()
<!-- topics: inheritance, overriding, super -->

A subclass reuses and specialises a base class. It **overrides** a method by defining one with the same name;
lookup finds the subclass version first. `super()` calls the *next* implementation in the MRO (next section), so
the subclass can extend instead of replacing.

```python
class BaseLLM:
    def __init__(self, model):
        self.model = model
    def describe(self):
        return f"model={self.model}"

class OpenAILLM(BaseLLM):
    def __init__(self, model, api_key):
        super().__init__(model)               # initialise the base part
        self.api_key = api_key
    def describe(self):                       # override and extend
        return super().describe() + " provider=openai"

o = OpenAILLM("gpt-x", "k")
print(o.describe())                           # prints: model=gpt-x provider=openai
print(isinstance(o, BaseLLM), issubclass(OpenAILLM, BaseLLM))   # prints: True True
```

If a subclass defines `__init__` and forgets `super().__init__()`, the base's attributes are never set, giving
`AttributeError` later. `super()` takes no arguments inside a method: the compiler supplies the class and
instance (through a hidden `__class__` cell, as in the closures note).

**Liskov substitution**: a subclass instance must be usable wherever the base is expected; if overrides must
narrow inputs or change meaning, inheritance is the wrong tool.

> **Interview** "What does `super()` do?" It does *not* mean "the parent class". It returns a proxy that continues
> the method search at the **next class after the current one in the MRO of `type(self)`**, which is what makes
> cooperative multiple inheritance work.

## MRO and multiple inheritance
<!-- topics: mro -->

With multiple bases, which method wins? Python linearises the class hierarchy with **C3 linearisation**: a
deterministic order, visible as `Class.__mro__`. Rules: a class comes before its bases, and the left-to-right order
of the bases listed is preserved.

```python
class A:
    def who(self): return "A"
class B(A):
    def who(self): return "B>" + super().who()
class C(A):
    def who(self): return "C>" + super().who()
class D(B, C):
    def who(self): return "D>" + super().who()

print([k.__name__ for k in D.__mro__])   # prints: ['D', 'B', 'C', 'A', 'object']
print(D().who())                         # prints: D>B>C>A
```

```text
      A          D's MRO: D, B, C, A, object
     / \         (A comes AFTER both B and C, the "diamond" is resolved once)
    B   C
     \ /
      D
```

Notice `B.who` calls `super().who()` and gets **C**, a class B knows nothing about. `super()` follows the MRO of
the *instance's* class, not B's own parents. That is why all classes in a cooperative hierarchy must call
`super()` and why `A` is run exactly once.

The standard use is **mixins**: small classes adding one capability, put *before* the main base.

```python
class LoggingMixin:
    def call(self, prompt):
        print("calling with", prompt)
        return super().call(prompt)

class Client:
    def call(self, prompt):
        return f"answer to {prompt}"

class LoggedClient(LoggingMixin, Client):
    pass

print(LoggedClient().call("hi"))
# prints: calling with hi
# prints: answer to hi
```

For `__init__` in mixin hierarchies, accept `**kwargs` and pass them on with `super().__init__(**kwargs)`.
An inconsistent base order raises `TypeError: Cannot create a consistent method resolution order`.

> **Gotcha** `Base.method(self)` (calling the parent by name) hard-codes one class and skips the others in a
> diamond, running a shared base twice or not at all. Use `super()` everywhere in a cooperative hierarchy.

## Polymorphism, abstraction and ABCs
<!-- topics: polymorphism, abstraction, abc -->

**Polymorphism** = one call, different behaviour depending on the object. In Python this is mostly **duck typing**:
"if it has the method, it works", with no shared base class required.

```python
class Echo:
    def complete(self, prompt): return prompt
class Shout:
    def complete(self, prompt): return prompt.upper()

for llm in (Echo(), Shout()):
    print(llm.complete("hi"))
# prints: hi
# prints: HI
```

**Abstraction** hides how and exposes what. Application code depends on `complete(prompt)`, not on a vendor SDK.
To *enforce* the contract, use an **abstract base class**: `abc.ABC` with `@abstractmethod`. A class with unimplemented
abstract methods cannot be instantiated, and the error appears at **instantiation time**, not at class definition or
call time.

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...

    def generate(self, prompt):                  # concrete "template method"
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("empty prompt")
        return self.complete(prompt)

class Echo(BaseLLM):
    def complete(self, prompt):
        return f"echo: {prompt}"

try:
    BaseLLM()
except TypeError as e:
    print(e)    # prints: Can't instantiate abstract class BaseLLM without an implementation for abstract method 'complete'
print(Echo().generate("  hi "))   # prints: echo: hi
```

The **template method** pattern (fixed public `generate`, subclasses fill abstract steps) keeps validation and
logging in one place.

`abc.ABC` versus `typing.Protocol`:

| | `ABC` | `Protocol` |
|---|---|---|
| Typing style | nominal: you must inherit | structural: matching methods is enough |
| Shared code | can provide concrete methods and `__init__` | interface only |
| Runtime enforcement | error on instantiation | none (unless `@runtime_checkable`, which checks names only) |
| Choose when | you own the hierarchy and want shared behaviour | you accept third-party or duck-typed objects, easy test fakes |

> **Interview** "Design the classes for several LLM providers." An ABC (or Protocol) with `complete` and
> `count_tokens`; one thin adapter per vendor translating to its SDK; the shared concerns (validation, retries,
> usage accounting) in the base as concrete template methods or injected collaborators; a factory
> `from_config` classmethod picking the adapter; application code typed against the abstraction. Fakes for tests
> fall out for free.

## Composition versus inheritance
<!-- topics: composition -->

Inheritance models **"is a"**, composition models **"has a"**. Prefer composition by default: it is looser
coupled, easier to test and avoids deep hierarchies and the fragile-base-class problem (changing the base breaks
subclasses).

```python
class LLMClient:                           # HAS a backend and a cache; both can be swapped
    def __init__(self, backend, cache):
        self.backend, self.cache = backend, cache
    def ask(self, prompt):
        if prompt not in self.cache:
            self.cache[prompt] = self.backend(prompt)
        return self.cache[prompt]

client = LLMClient(lambda p: p[::-1], {})
print(client.ask("abc"), client.ask("abc"))    # prints: cba cba
```

The alternative, `class CachedRetryingOpenAIClient(OpenAIClient)`, explodes into combinations (cached only?
retry only? Anthropic variant?). With composition you pass the collaborators in (dependency injection), so tests
inject a fake backend. Use inheritance when there is a genuine "is a", a stable interface and you want
polymorphism; use mixins sparingly for orthogonal behaviour.

## Dunder methods
<!-- topics: dunder -->

"Dunder" (double underscore) methods are the hooks the language calls for you: operators, `len()`, `print()`,
`with`, indexing. You rarely call them directly; you implement them so your objects behave like built-ins.

```python
class Vec:
    def __init__(self, *xs):
        self.xs = tuple(xs)
    def __repr__(self):                       # for developers, unambiguous, ideally eval-able
        return f"Vec{self.xs}"
    def __str__(self):                        # for users; falls back to __repr__ if missing
        return "<" + ", ".join(map(str, self.xs)) + ">"
    def __add__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented             # NOT raise: lets Python try the other operand
        return Vec(*(a + b for a, b in zip(self.xs, other.xs)))
    def __eq__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        return self.xs == other.xs
    def __hash__(self):
        return hash(self.xs)

v = Vec(1, 2) + Vec(3, 4)
print(repr(v), str(v))                        # prints: Vec(4, 6) <4, 6>
print(v == Vec(4, 6), v is Vec(4, 6), len({v, Vec(4, 6)}))   # prints: True False 1
try:
    Vec(1) + 5
except TypeError as e:
    print(e)    # prints: unsupported operand type(s) for +: 'Vec' and 'int'
```

Key rules:

* `__repr__` is for debugging and logs (always define it); `__str__` is for display and defaults to `__repr__`.
* Return **`NotImplemented`** (the singleton, not the exception) from binary operators and `__eq__` for unsupported
  types: Python then tries the reflected method on the other operand (`__radd__`) and finally raises `TypeError` itself.
  Raising yourself blocks the other side from handling the operation.
* **`__eq__` and `__hash__` go together.** Defining `__eq__` sets `__hash__` to `None`, making the object
  unhashable (not usable in sets or as dict keys) until you define `__hash__`. Equal objects must have equal hashes,
  and the hash must not change while the object is in a set, so hash only immutable fields.
* Others you will meet: `__bool__`, `__iter__`, `__contains__`, `__call__`, `__enter__`/`__exit__`, `__lt__`
  (with `functools.total_ordering`), `__getattr__`.

## Dataclasses
<!-- topics: dataclasses -->

`@dataclass` generates `__init__`, `__repr__` and `__eq__` from annotated fields, so plain data holders (messages,
configs, results) need no boilerplate.

```python
from dataclasses import dataclass, field, replace

@dataclass
class Message:
    role: str
    content: str
    tags: list[str] = field(default_factory=list)    # NEVER `= []`

m = Message("user", "hi")
print(m)                                  # prints: Message(role='user', content='hi', tags=[])
print(m == Message("user", "hi"))         # prints: True
print(replace(m, content="bye"))          # prints: Message(role='user', content='bye', tags=[])
```

A mutable default like `tags: list = []` is rejected with `ValueError` precisely because it would be shared like the
class-variable trap; use `field(default_factory=list)`.

| Option | Effect |
|---|---|
| `frozen=True` | fields cannot be reassigned (`FrozenInstanceError`); with `eq=True` generates `__hash__`, so usable in sets/dict keys |
| `order=True` | generates `<`, `<=`, `>`, `>=` comparing fields as a tuple in order |
| `slots=True` | generates `__slots__`: less memory, no new attributes |
| `kw_only=True` | all fields keyword-only |
| `__post_init__` | hook run after the generated `__init__` (validation, derived fields) |

```python
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int = 0

print(Version(1, 2) < Version(1, 10))     # prints: True
try:
    Version(1).major = 5
except AttributeError as e:               # FrozenInstanceError is a subclass of AttributeError
    print(type(e).__name__)               # prints: FrozenInstanceError
print(len({Version(1), Version(1)}))      # prints: 1
```

> **AI use** Dataclasses are ideal for chat messages, retrieval results, run configs. Use them for trusted
> internal data; reach for pydantic when you must **validate and parse untrusted input** (LLM JSON output, API
> payloads), since dataclasses do not check types at runtime.

> **Gotcha** `frozen=True` is shallow: a frozen dataclass holding a list can still have the list mutated.

## __slots__
<!-- topics: slots -->

Normally every instance carries a `__dict__` (a hash table) for its attributes. `__slots__` replaces it with fixed,
compact storage: less memory per object, slightly faster attribute access and typo-protection (assigning an unknown
attribute raises `AttributeError`). It pays off when you create millions of small objects, like embedding records
or tokens.

```python
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y

p = Point(1, 2)
print(hasattr(p, "__dict__"))     # prints: False
try:
    p.z = 3
except AttributeError as e:
    print(e)    # prints: 'Point' object has no attribute 'z' and no __dict__ for setting new attributes
```

Gotchas: no dynamic attributes, no `__weakref__` unless listed, and slots are only effective if every class in
the chain defines them (a subclass without `__slots__` gets a `__dict__` again). `@dataclass(slots=True)` gives
both.

## Shallow copy versus deep copy
<!-- topics: copy -->

Assignment never copies: `b = a` makes a second name for the same object. A **shallow copy** (`copy.copy`,
`list(a)`, `a[:]`, `dict(a)`, `a.copy()`) creates a new outer container but its elements are the *same* objects. A
**deep copy** (`copy.deepcopy`) recursively copies everything, handling cycles through a memo dictionary.

```python
import copy

a = {"cfg": {"retries": 3}, "tags": ["x"]}

alias = a
shallow = copy.copy(a)
deep = copy.deepcopy(a)

a["cfg"]["retries"] = 99
print(alias["cfg"]["retries"], shallow["cfg"]["retries"], deep["cfg"]["retries"])   # prints: 99 99 3

```

The shallow copy has its own top-level dict but shares the nested `cfg` dict, so changing nested content leaks
through. The classic bug is `copy.copy(defaults)` for a nested config, then mutating
the copy's inner dict and corrupting the shared defaults.

You can customise copying with `__copy__` / `__deepcopy__(self, memo)`. Deep copy is slow and
can duplicate things you do not want (clients, locks, models, open files), so for configs prefer immutable
structures (`frozen` dataclasses, tuples) or copy only the part you will change.

## Quick recap
- `__new__` creates, `__init__` initialises; instance variables live in `__dict__`; mutable defaults and mutable class variables are shared state.
- Reading `obj.x` goes instance, class, bases; writing creates an instance attribute.
- Instance method gets `self`, classmethod gets `cls` (alternative constructors), staticmethod gets nothing.
- Encapsulation is convention (`_x`, mangled `__x`); `@property` is a data descriptor on the class for validation and computed values.
- `super()` follows the MRO (C3) of the instance's class, not "the parent"; mixins need cooperative `super()` calls.
- Duck typing is polymorphism; `ABC` enforces at instantiation; `Protocol` is the structural alternative.
- Prefer composition and injected collaborators; inherit for genuine "is a".
- Return `NotImplemented` from binary operators; `__eq__` without `__hash__` makes the type unhashable; always define `__repr__`.
- `@dataclass`: `default_factory` for mutables; `frozen`, `order`, `slots`, `__post_init__`; no runtime validation.
- `__slots__` is a memory optimisation; `copy.copy` shares nested objects, `deepcopy` duplicates them.
