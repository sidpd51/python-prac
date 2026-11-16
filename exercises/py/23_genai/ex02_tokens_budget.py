"""
Exercise 23.2 - Tokens: counting, truncating to a budget, fitting chat history   level: ★★☆
topics: genai.tokenization, genai.text_processing

Problem
-------
LLM limits and prices are in tokens, not characters. Real tokenizers (tiktoken, sentencepiece)
are BPE models; here we use a deterministic approximation with the same shape.

1. tokenize(text) -> list[str]
   Pre-tokenization: runs of word characters (regex \\w+) and single punctuation/symbol
   characters ([^\\w\\s]); whitespace is dropped.
       tokenize("Hello, world!") == ["Hello", ",", "world", "!"]

2. count_tokens(text) -> int
   Approximate BPE cost: a word piece of length L costs ceil(L / 4) tokens ("~4 characters
   per token" rule of thumb), each punctuation character costs 1.
       count_tokens("Hello, world!") == 2 + 1 + 2 + 1 == 6

3. truncate_to_budget(text, max_tokens) -> str
   The longest prefix of `text` made of WHOLE pre-tokens whose total cost <= max_tokens.
   Cut right after the last fitting token, keeping the original spacing/newlines inside the
   kept prefix (hint: re.finditer + match.end()). A word is never cut in half: if the first
   token doesn't fit, return "". If the whole text fits, return it unchanged.
   max_tokens < 0 -> ValueError.

4. fit_messages(messages, max_tokens) -> list[dict]
   Chat history trimming. messages: [{"role": "system"|"user"|"assistant", "content": str}]
   Each message costs count_tokens(content) + 4 (per-message overhead of chat formats).
   * keep ALL system messages and the LAST message, always
   * drop the OLDEST other messages, one at a time, until the total fits
   * return a new list in the original order (don't mutate the input)
   * if the system messages + last message alone exceed the budget -> ValueError

Interviewers look for
---------------------
* Why len(text) / 4 is only an estimate: tokenizers are model-specific (cl100k vs o200k vs
  Llama); count with the model's own tokenizer when cost/limits matter.
* Truncate on token boundaries, never mid-word or mid-UTF-8 character.
* Context budget = prompt tokens + reserved output tokens (max_tokens of the response).
* Trimming history: keep the system prompt and the latest turn; drop/summarise the middle.
"""

MESSAGE_OVERHEAD = 4


def tokenize(text: str) -> list[str]:
    raise NotImplementedError


def count_tokens(text: str) -> int:
    raise NotImplementedError


def truncate_to_budget(text: str, max_tokens: int) -> str:
    raise NotImplementedError


def fit_messages(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    raise NotImplementedError
