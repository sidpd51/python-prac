"""
Exercise 01.4 - Context-window slicing and usage formatting         level: ★★☆
topics: fundamentals.slicing, fundamentals.fstrings, fundamentals.operators,
        fundamentals.conditionals

Problem
-------
1. last_turns(history: list, n: int) -> list
     * the last n items (a new list); n == 0 -> []; n larger than len -> all items
     * n < 0 -> ValueError
     * this is how a chat app keeps "the last n turns" inside the context window

2. truncate_middle(text: str, max_len: int, marker: str = "...") -> str
     * len(text) <= max_len -> text unchanged
     * otherwise keep the head and the tail with `marker` in between so that the result
       is EXACTLY max_len characters; when the kept characters can't split evenly the
       head gets the extra one
       truncate_middle("abcdefghij", 7) -> "ab...ij"; truncate_middle("abcdefghij", 6) -> "ab...j"
     * max_len < len(marker) -> ValueError
     (used for logging huge prompts/tool outputs: the start AND the end are informative)

3. format_usage(prompt_tokens: int, completion_tokens: int, cost_usd: float) -> str
     * "prompt=1,234 completion=56 total=1,290 cost=$0.0123"
     * thousands separators on all token counts, cost with exactly 4 decimals

4. progress_bar(done: int, total: int, width: int = 20) -> str
     * f"[{'#' * filled}{'.' * (width - filled)}] {percent} ({done}/{total})"
     * filled = done * width // total (floor), percent is done/total as a whole-number
       percentage right-aligned in 4 chars: "  0%", " 25%", "100%"
       progress_bar(5, 20, 10) -> "[##........]  25% (5/20)"
     * total <= 0, width <= 0, or done outside 0..total -> ValueError

Interviewers look for
---------------------
* xs[-n:] with n == 0 is xs[0:] - the WHOLE list. A classic off-by-zero bug.
* Slices clamp to bounds and never raise; they always return a new object (shallow copy).
* Floor division and modulo: `a // b`, `a % b`, ceil via `-(-a // b)`; `/` always gives float.
* Format spec mini-language: `{n:,}`, `{x:.4f}`, `{p:>4.0%}`, `{s:<10}`, nested `{x:{w}}`,
  and `{x=}` for debugging.
"""


def last_turns[T](history: list[T], n: int) -> list[T]:
    raise NotImplementedError


def truncate_middle(text: str, max_len: int, marker: str = "...") -> str:
    raise NotImplementedError


def format_usage(prompt_tokens: int, completion_tokens: int, cost_usd: float) -> str:
    raise NotImplementedError


def progress_bar(done: int, total: int, width: int = 20) -> str:
    raise NotImplementedError
