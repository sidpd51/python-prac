"""
Exercise 23.1 - Cleaning scraped text: unicode normalisation, dedup, PII, citations   level: ★☆☆
topics: genai.normalization, genai.text_processing, genai.regex

Problem
-------
Text scraped from PDFs/HTML is full of look-alike characters, so "the same" sentence
hashes differently and your RAG index fills with near-duplicates.

1. normalize_text(text, *, casefold=False) -> str, in this order:
   * unicodedata NFKC  (full-width "ＡＩ" -> "AI", ligature "ﬁ" -> "fi", NBSP -> space)
   * delete zero-width characters: U+200B, U+200C, U+200D, U+2060, U+FEFF
   * smart quotes -> ASCII:  ‘ ’ ‚ ‛ -> '      “ ” „ ‟ -> "      en/em dash – — -> -
   * collapse every run of whitespace (spaces, tabs, newlines...) into ONE space, strip ends
   * casefold=True -> str.casefold() at the end ("Straße" -> "strasse")

2. dedupe_texts(texts) -> list[str]
   Keep the FIRST occurrence of each text, comparing by normalize_text(t, casefold=True).
   Return the ORIGINAL strings (not normalised). Texts that normalise to "" are dropped.

3. redact_pii(text) -> str   (regex-based)
   * emails -> "[EMAIL]"   (local part: letters, digits, . _ % + -; then @; domain with
     at least one dot, e.g. jane.doe+rag@mail.example.co.uk)
   * North-American phone numbers -> "[PHONE]": optional "+" country code (1-3 digits) and
     separator, then a 3-digit area code optionally in parentheses, 3 digits, 4 digits;
     each separator is optional and is one space, "-" or ".".
     Examples: "+1 415-555-0132", "(415) 555-0132", "415.555.0132", "4155550132".
     A phone number is not part of a longer run of digits/letters.
   * redact emails first (an email can contain digits)

4. extract_citations(text) -> list[int]
   LLM answers cite sources as "[1]", "[2, 3]" or ranges "[4-6]" / "[4–6]" (en dash).
   Return the cited numbers, unique, in order of FIRST appearance; ranges expand
   inclusively (a range with start > end is ignored). Brackets that don't contain only
   numbers/commas/ranges ("[a]", "[]", "[1a]", markdown "[link](url)") are not citations.
       extract_citations("A [2]. B [1, 3-4]. C [2][5]") == [2, 1, 3, 4, 5]

Interviewers look for
---------------------
* NFC vs NFKC: NFC only composes (é == e + U+0301); NFKC also folds "compatibility"
  look-alikes - right for matching/dedup keys, too lossy for text you display.
* casefold() vs lower(): casefold handles "ß" -> "ss" and other caseless matching rules.
* Zero-width chars aren't whitespace (`"\\u200b".isspace()` is False) - strip them explicitly.
* Regex PII redaction is a high-recall heuristic; real systems add NER (e.g. Presidio).
* Anchor with lookarounds instead of \\b when the pattern starts with "+" or "("
  (\\b needs a word character next to it). re.compile hot patterns (re also caches).
"""

from collections.abc import Iterable


def normalize_text(text: str, *, casefold: bool = False) -> str:
    raise NotImplementedError


def dedupe_texts(texts: Iterable[str]) -> list[str]:
    raise NotImplementedError


def redact_pii(text: str) -> str:
    raise NotImplementedError


def extract_citations(text: str) -> list[int]:
    raise NotImplementedError
