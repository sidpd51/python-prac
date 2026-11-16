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

import re
import unicodedata
from collections.abc import Iterable
from functools import cache


@cache
def _translation_table() -> dict[int, str | None]:
    return str.maketrans(
        {
            **dict.fromkeys("\u200b\u200c\u200d\u2060\ufeff"),
            **dict.fromkeys("\u2018\u2019\u201a\u201b", "'"),
            **dict.fromkeys("\u201c\u201d\u201e\u201f", '"'),
            **dict.fromkeys("\u2013\u2014", "-"),
        }
    )


def normalize_text(text: str, *, casefold: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_translation_table())
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold() if casefold else text


def dedupe_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        key = normalize_text(text, casefold=True)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


# (re caches compiled patterns, so inline patterns cost no recompilation per call)
def redact_pii(text: str) -> str:
    text = re.sub(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+", "[EMAIL]", text)
    phone = r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"
    return re.sub(phone, "[PHONE]", text)


def extract_citations(text: str) -> list[int]:
    item = r"\s*\d+(?:\s*[-–]\s*\d+)?\s*"
    seen: dict[int, None] = {}  # dict keeps insertion order -> an ordered set
    for match in re.finditer(rf"\[({item}(?:,{item})*)\]", text):
        for part in match.group(1).split(","):
            lo, _, hi = part.replace("–", "-").partition("-")
            start = int(lo)
            end = int(hi) if hi else start
            for n in range(start, end + 1):  # empty when start > end
                seen.setdefault(n)
    return list(seen)
