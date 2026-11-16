# Pydantic (v2)

Pydantic turns untrusted data (JSON from an HTTP request, text from an LLM, environment variables) into
typed Python objects, or tells you precisely why it cannot. In AI engineering almost every boundary is
untrusted: model output is a string that merely *looks* like JSON, tool arguments come from the model,
config comes from the environment. Pydantic is the standard answer, and the same models double as the
JSON schemas you send to function-calling APIs. Interviews probe whether you know validation versus
coercion, the optional/nullable distinction, validator ordering, and how to survive messy LLM output.

## Models and BaseModel
<!-- topics: models, basemodel -->

A model is a class that inherits `BaseModel`; its annotated attributes are **fields**. Calling the class
runs validation (the core is written in Rust, pydantic-core), so you get either a fully valid instance or
a `ValidationError`. There is no half-valid object.

```python
from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    tags: list[str] = []

a, b = Chunk(text="a"), Chunk(text="b")
a.tags.append("pii")
print(b.tags)          # prints: []   (mutable defaults are copied per instance)
print(a)               # prints: text='a' tags=['pii']
a.text = 123           # assignment is NOT validated by default
print(repr(a.text))    # prints: 123
```

Where it sits among the alternatives:

| | runtime checks | coercion | JSON + schema | use for |
|---|---|---|---|---|
| `TypedDict` | none | no | via TypeAdapter | typing a dict that is already trusted |
| `dataclass` | none | no | no | internal value objects |
| `BaseModel` | full | yes | yes | **boundaries**: HTTP, LLM output, config |

> **Remember** Validate at the edge, then pass trusted objects inward. Re-validating on every internal call is
> wasted work.

> **Gotcha** Unknown input keys are **ignored** by default (`extra="ignore"`), so a typo like `temprature`
> silently vanishes. Use `ConfigDict(extra="forbid")` for request models. Assignment is also unchecked unless
> `validate_assignment=True`; `frozen=True` forbids mutation and makes the model hashable.

## Validation: lax versus strict
<!-- topics: validation -->

Pydantic v2 **validates and, by default, coerces** ("lax mode"): input that is not the right type but
*unambiguously convertible* is converted. Strict mode refuses all conversion.

```python
from pydantic import BaseModel, ValidationError

class M(BaseModel):
    n: int

print(M(n="3").n, M(n=True).n, M(n=4.0).n)    # prints: 3 1 4
try:
    M(n=3.7)
except ValidationError as e:
    err = e.errors()[0]
    print(err["type"], err["loc"])            # prints: int_from_float ('n',)
```

Lax mode accepts numeric strings, `True` as 1, and floats with no fractional part; it **never silently
truncates** (v1 turned 3.7 into 3). Turn on strictness per type (`StrictInt`), per field (`Field(strict=True)`) or per model (`ConfigDict(strict=True)`):

```python
from pydantic import BaseModel, Field, StrictInt, ValidationError

class Usage(BaseModel):
    prompt_tokens: StrictInt
    completion_tokens: int = Field(strict=True)

for bad in ("100", True, 100.0):
    try:
        Usage(prompt_tokens=bad, completion_tokens=1)
    except ValidationError as e:
        print(repr(bad), e.errors()[0]["type"])
# prints: '100' int_type
# prints: True int_type
# prints: 100.0 int_type
```

Errors are structured: `e.errors()` gives a list of dicts with `loc` (path), `type` (stable code),
`msg`, and `input`. Use `type`/`loc` in code and tests, never the message text. Strictness also depends
on the *input source*: JSON has no datetime or enum type, so even in strict mode `model_validate_json` accepts
a string for those, while Python-object input must already be a real `datetime`/enum.

> **Interview** "Is pydantic coercion a bug or a feature?" Feature at the edge (env vars and query strings are
> strings, LLMs return `"3"` for 3). Dangerous where the type carries meaning: use strict for token counts,
> money, IDs. Say you pick per field.

## Required, optional and nullable
<!-- topics: required_optional -->

Two independent questions: *may the key be missing?* (has a default) and *may the value be `None`?*
(type includes `None`). v2 keeps them apart, v1 did not.

```python
from pydantic import BaseModel, ValidationError

class Doc(BaseModel):
    a: int                      # required, not nullable
    b: int = 0                  # optional (default), not nullable
    c: int | None               # REQUIRED but may be None
    d: int | None = None        # optional and nullable

print(Doc(a=1, c=None))         # prints: a=1 b=0 c=None d=None
try:
    Doc(a=1)
except ValidationError as e:
    print([(x["loc"], x["type"]) for x in e.errors()])   # prints: [(('c',), 'missing')]
```

`Optional[X]` is only an alias for `X | None`; it does **not** add a default (v1 did, a classic migration bug).
To tell "absent" from "explicitly null" (PATCH semantics) use `model_fields_set` or `model_dump(exclude_unset=True)`:
`Patch(title=None).model_fields_set` is `{'title'}`, while `Patch().model_fields_set` is `set()`.

> **AI use** LLM structured output: mark fields the model might legitimately not know as `str | None` and
> *require* them. That forces the model to say `null` explicitly instead of omitting or hallucinating.

## Field and constraints
<!-- topics: field, constraints -->

`Field(...)` attaches defaults, constraints, aliases and documentation to a field.

```python
from pydantic import BaseModel, Field, ValidationError

class ChatRequest(BaseModel):
    model: str = Field(min_length=1, description="Model id")
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_tokens: int = Field(gt=0, le=4096)
    stop: list[str] = Field(default_factory=list, max_length=4)
    user: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$")

try:
    ChatRequest(model="", temperature=3, max_tokens=0, stop=list("abcde"), user="Bad Name")
except ValidationError as e:
    for err in e.errors():
        print(err["loc"][0], err["type"])
# prints: model string_too_short
# prints: temperature less_than_equal
# prints: max_tokens greater_than
# prints: stop too_long
# prints: user string_pattern_mismatch
```

All errors are collected in one pass, not just the first. The constraint families:

| Applies to | Constraints |
|---|---|
| numbers | `gt`, `ge`, `lt`, `le`, `multiple_of` |
| str, list, set, dict | `min_length`, `max_length` |
| str | `pattern`, plus `strip_whitespace` via `StringConstraints` |
| any | `strict`, `default`, `default_factory`, `alias`, `description`, `examples`, `exclude` |

Reusable constrained types: `Temperature = Annotated[float, Field(ge=0, le=2)]`, then use `temperature: Temperature = 1.0`
in any model. `StringConstraints(strip_whitespace=True, min_length=1)` strips then checks length.

> **Gotcha** `default_factory` is for values created per instance (`list`, `uuid4`, `datetime.now`). A plain
> `default=[]` happens to be safe in pydantic (it copies), but `default_factory=list` states intent and works
> the same for `datetime.now`, where a plain default would be frozen at import time.

> **AI use** `description=` ends up in the JSON schema the model reads when calling a tool. It is prompt
> engineering: a vague description gives bad tool arguments.

## Custom field validators
<!-- topics: custom_validators -->

When constraints are not enough, write a `@field_validator`. It is a classmethod that receives the value
and returns the (possibly changed) value, or raises `ValueError`/`AssertionError`.

```python
from pydantic import BaseModel, field_validator, ValidationError

class Tags(BaseModel):
    tags: list[str]

    @field_validator("tags")                  # mode="after" is the default
    @classmethod
    def clean(cls, v: list[str]) -> list[str]:
        out = sorted({t.strip().lower() for t in v if t.strip()})
        if not out:
            raise ValueError("need at least one tag")
        return out

print(Tags(tags=[" B", "a", "b", ""]).tags)    # prints: ['a', 'b']
try:
    Tags(tags=[" "])
except ValidationError as e:
    print(e.errors()[0]["msg"])                # prints: Value error, need at least one tag
```

The **mode** decides when it runs relative to pydantic's own type check:

| mode | runs | value you receive | use to |
|---|---|---|---|
| `"after"` (default) | after type validation | already the right type | extra rules, normalising |
| `"before"` | before | raw input, any type | coerce messy input first |
| `"wrap"` | around | raw input + a `handler` | try/fallback around validation |

Use `mode="before"` to repair raw input, e.g. an LLM sends `"doc-1"` where `["doc-1"]` is expected: check
`isinstance(v, str)` and return `[v]`.

Raise `ValueError` (not `TypeError`) inside validators: pydantic converts `ValueError`/`AssertionError` into a
`ValidationError`, while other exceptions escape as raw crashes. Forgetting `@classmethod`, or returning
nothing from an after validator (the field becomes `None`), are the usual bugs.

> **Interview** "before vs after validator?" Before sees raw, untyped input and repairs it; after sees a typed
> value and applies business rules. Mention `Annotated[str, AfterValidator(fn)]` as the reusable form.

## model_validator: cross-field rules
<!-- topics: model_validator -->

A field validator sees one field. When a rule involves several fields, use `@model_validator`.

* `mode="after"`: an instance method, runs after all fields are valid, receives `self`, **must return `self`**.
* `mode="before"`: a classmethod receiving the raw input (usually a dict), returns the dict to validate. Use it
  to reshape input.

```python
from pydantic import BaseModel, model_validator, ValidationError

class Range(BaseModel):
    lo: int
    hi: int

    @model_validator(mode="after")
    def ordered(self):
        if self.lo > self.hi:
            raise ValueError("lo must be <= hi")
        return self

try:
    Range(lo=5, hi=1)
except ValidationError as e:
    print(e.errors()[0]["loc"], e.errors()[0]["msg"])
# prints: () Value error, lo must be <= hi
```

The error `loc` is `()`: it belongs to the model, not a field. A `before` model validator is how you accept
OpenAI-style nesting (`{"function": {"name": ...}}`) by flattening it first; the input may not be a dict at all
(an instance, or garbage), so check `isinstance` first.

> **Gotcha** Assigning fields inside an `after` validator with `validate_assignment=True` recurses. If a field already
> failed, the after validator does not run.

## Nested models
<!-- topics: nested -->

A field typed as another model is validated recursively, from dicts, and errors carry the full path.

```python
from pydantic import BaseModel, ValidationError

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    id: str
    messages: list[Message]
    meta: dict[str, str] = {}

data = {"id": "c1", "messages": [{"role": "user", "content": "hi"}, {"role": "bot"}]}
try:
    Conversation.model_validate(data)
except ValidationError as e:
    print(e.errors()[0]["loc"], e.errors()[0]["type"])
    # prints: ('messages', 1, 'content') missing
```

Dicts became `Message` instances automatically. Instances you pass in are **not** copied by default (same
object, `revalidate_instances="never"`), so mutating it afterwards changes the model's data.

Self-referencing models (`children: list["Node"]`) work via string annotations; their schemas contain `$ref`.

## Parsing: model_validate and model_validate_json
<!-- topics: model_validate -->

`Model(**data)` and `Model.model_validate(data)` both validate a Python object. Prefer `model_validate`: it
takes a non-dict directly and does not need string keys to be valid keyword names.
`model_validate_json(text)` parses **and** validates in one Rust pass.

```python
from pydantic import BaseModel, TypeAdapter

class P(BaseModel):
    x: int

raw = '{"x": "7"}'
print(P.model_validate_json(raw))               # prints: x=7
print(TypeAdapter(list[P]).validate_json('[{"x": 1}, {"x": 2}]'))
# prints: [P(x=1), P(x=2)]
```

`model_validate_json` also turns JSON syntax errors into a `ValidationError` (`json_invalid`); `from_attributes=True`
reads ORM-like objects; `TypeAdapter(T)` validates `list[Model]`, unions and `TypedDict`s without a model class.

> **Gotcha** `json.loads` then `model_validate` is slower and, for strictness, behaves differently from
> `model_validate_json`. Use the JSON entry point when your input is a JSON string. Also,
> `parse_obj`/`parse_raw` are v1 names, deprecated.

## Serialization: model_dump and aliases
<!-- topics: serialization, model_dump -->

Serialization goes model to dict or JSON. `model_dump()` gives Python objects; `model_dump_json()` gives a JSON
string (and knows how to render datetimes, UUIDs, enums). The options are what interviews ask about:

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    email: str | None = None
    api_key: str = Field(default="sk-secret", exclude=True)
    nick: str = "n"

u = User(name="Ada")
print(u.model_dump())                         # prints: {'name': 'Ada', 'email': None, 'nick': 'n'}
print(u.model_dump(exclude_none=True))        # prints: {'name': 'Ada', 'nick': 'n'}
print(u.model_dump(exclude_unset=True))       # prints: {'name': 'Ada'}
print(u.model_dump(exclude_defaults=True))    # prints: {'name': 'Ada'}
print(u.model_dump(include={"name"}))         # prints: {'name': 'Ada'}
print(u.model_dump_json())                    # prints: {"name":"Ada","email":null,"nick":"n"}
```

* `exclude_unset`: drop fields the caller never provided (PATCH payloads).
* `exclude_defaults`: drop fields equal to their default.
* `exclude_none`: drop `None` values (many APIs reject explicit nulls).
* `Field(exclude=True)`: never serialize (secrets).

**Aliases** let the external name differ from the Python name. Three flavours: `alias` (both directions),
`validation_alias` (input only, can be `AliasChoices`), `serialization_alias` (output only).

```python
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

class Req(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    max_tokens: int = Field(
        validation_alias=AliasChoices("maxTokens", "max_completion_tokens", "max_tokens"),
        serialization_alias="maxTokens",
    )

r = Req.model_validate({"max_completion_tokens": 10})
print(r.max_tokens)                          # prints: 10
print(r.model_dump(by_alias=True))           # prints: {'maxTokens': 10}
print(r.model_dump())                        # prints: {'max_tokens': 10}
```

> **Gotcha** `model_dump()` ignores aliases unless `by_alias=True`. For OpenAI-style payloads you usually want
> `model_dump(exclude_none=True, by_alias=True)`.

`@computed_field` adds derived output, `@field_serializer` customises a format, and `model_dump(mode="json")` returns only
JSON-safe types.

## JSON Schema and LLM tool definitions
<!-- topics: json_schema -->

`Model.model_json_schema()` returns the JSON Schema describing the model: field types, constraints, required
keys, descriptions. That is exactly what function-calling APIs want as a tool's `parameters`, so **one model is
both the contract sent to the LLM and the validator for what comes back**. No drift between docs and code.

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get current weather for a city."""
    city: str = Field(description="City name, e.g. Paris")
    unit: str = Field(default="c", pattern="^(c|f)$")

schema = GetWeather.model_json_schema()
print(schema["required"], schema["description"])   # prints: ['city'] Get current weather for a city.
print(schema["properties"]["unit"])
# prints: {'default': 'c', 'pattern': '^(c|f)$', 'title': 'Unit', 'type': 'string'}
```

A tool definition wraps it: `{"type": "function", "function": {"name": ..., "description": ..., "parameters": schema}}`.
The loop is: send schema, receive `arguments` as a JSON string, `model_validate_json` it, execute, and on
`ValidationError` feed the error text back to the model.

Providers are pickier than the standard:

* Nested models produce `$defs` and `$ref`; many providers want them inlined.
* Strict/structured-output modes want `additionalProperties: false` on every object and every field required.
* `title` keywords are noise; a *property* literally called `title` must be kept.

> **AI use** Same trick everywhere: structured output (`response_format`), tool calling, and MCP tool schemas
> are all "a JSON Schema from a model".

## Settings from the environment
<!-- topics: settings -->

`pydantic-settings` provides `BaseSettings`: a model whose fields are read from environment variables (and
optionally a `.env` file), then validated and coerced like any model. This is the twelve-factor way: config in
the environment, typed in code, failing loudly at start-up if wrong.

```python
import os
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")
    openai_api_key: SecretStr
    timeout_s: float = 30.0
    max_retries: int = Field(default=3, ge=0)
    debug: bool = False

os.environ["APP_OPENAI_API_KEY"] = "sk-123"
os.environ["APP_DEBUG"] = "yes"
os.environ["APP_MAX_RETRIES"] = "5"
s = Settings()
print(s.timeout_s, s.max_retries, s.debug)    # prints: 30.0 5 True
print(s.openai_api_key)                       # prints: **********
print(s.openai_api_key.get_secret_value())    # prints: sk-123

os.environ["APP_MAX_RETRIES"] = "-1"
try:
    Settings()
except ValidationError as e:
    print(e.errors()[0]["loc"])               # prints: ('max_retries',)
```

* Env strings are coerced (`"yes"` becomes `True`, `"5"` becomes 5); names are case-insensitive; `env_prefix` namespaces.
* **Priority** (highest first): init kwargs, env vars, `.env` file, secrets dir, defaults. `Settings(timeout_s=1)` in tests.
* `SecretStr` hides the value in `repr`, logs and tracebacks. Nested: `env_nested_delimiter="__"`.

> **Gotcha** Build settings once (module level or `functools.lru_cache`) and inject it; constructing
> `Settings()` all over the codebase re-reads the environment each time, and a missing variable then crashes
> far from start-up.

## Validating API and LLM output
<!-- topics: api_validation -->

LLMs return strings. Even "JSON only" prompts produce fences, prose, trailing commas, wrong types. A solid
pipeline has four steps: **extract** the JSON text, **parse+validate** with the model, **classify** failure,
and **repair** by re-prompting with the error.

```python
import json
from pydantic import BaseModel, Field, ValidationError

class Triage(BaseModel):
    category: str
    priority: int = Field(ge=1, le=5)

def extract_json(text: str) -> str:
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                _, end = dec.raw_decode(text, i)
                return text[i:end]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found")

fence = "`" * 3
reply = f'Sure! Use {{name}} as a placeholder.\n{fence}json\n{{"category": "bug", "priority": "2"}}\n{fence}\nBye'
print(Triage.model_validate_json(extract_json(reply)))
# prints: category='bug' priority=2
```

`raw_decode` finds one complete value starting at an index, which is more robust than regexes for nested braces
and braces inside strings. (It would skip `{name}` because that is not valid JSON.) The repair loop feeds
`str(e)` or `e.errors()` back as the next user message and caps attempts.

**Discriminated unions** handle "one of several shapes" (agent actions, tool calls). Plain unions try each
member in turn and report errors for all of them; a discriminator reads one tag field and picks the right
member directly: faster, and errors point at the right variant.

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

class Search(BaseModel):
    type: Literal["search"]
    query: str
    top_k: int = Field(default=3, ge=1, le=10)

class Final(BaseModel):
    type: Literal["final_answer"]
    answer: str

Action = Annotated[Search | Final, Field(discriminator="type")]
adapter = TypeAdapter(Action)

print(adapter.validate_python({"type": "final_answer", "answer": "42"}))
# prints: type='final_answer' answer='42'
try:
    adapter.validate_python({"type": "search", "query": "x", "top_k": 99})
except ValidationError as e:
    print(len(e.errors()), e.errors()[0]["loc"])   # prints: 1 ('search', 'top_k')
try:
    adapter.validate_python({"type": "delete"})
except ValidationError as e:
    print(e.errors()[0]["type"])                   # prints: union_tag_invalid
```

Every member needs a `Literal` tag. With parallel tool calls, validate each item independently so one bad
action does not discard the others.

> **Interview** "How do you make an LLM reliably return structured data?" Layers: schema-constrained decoding
> or tool calling where the provider supports it; a pydantic model as the single source of truth; tolerant
> extraction; `model_validate_json`; a bounded repair loop sending the validation errors back; finally a
> fallback or a typed failure. Never `eval`, never trust unparsed text.

## Quick recap
- BaseModel validates on construction; no half-valid instances. Assignment is unchecked unless `validate_assignment`.
- Default lax mode coerces (`"3"`, `True`, `4.0` to int) but never truncates 3.7; use strict for meaningful types.
- Required vs nullable are independent: `int | None` has no default. `model_fields_set` shows what was provided.
- `Field` carries defaults, constraints (`ge/le/gt/lt`, `min_length`, `pattern`), aliases, description.
- `field_validator` modes: before (raw input), after (typed, default), wrap. Raise `ValueError`, return the value.
- `model_validator(mode="after")` for cross-field rules and returns `self`; `mode="before"` reshapes raw input.
- Nested models validate recursively; error `loc` shows the full path.
- `model_validate_json` for JSON strings; `TypeAdapter` for non-model types.
- `model_dump`: `exclude_unset/defaults/none`, `by_alias`, `mode="json"`; `Field(exclude=True)` hides secrets.
- `model_json_schema()` makes tool/structured-output schemas from the same model that validates the reply.
- `BaseSettings` reads typed config from env; init kwargs beat env beat `.env` beat defaults; `SecretStr`.
- Parse LLM output as extract, validate, repair; use discriminated unions for multi-shape actions.
