# Notes guide: how revision notes are written

The learner has already *covered* the syllabus once but does not trust their understanding. The
notes are what they re-read to **really understand** each concept and be ready for interviews at
3-4 years of experience. Reference note: `notes/py/08_generators.md`. Copy its structure and tone.

## Where
`notes/<track>/<NN_module>.md`, same names as `exercises/` and `questions/` (e.g. `notes/py/16_async.md`).

## Shape of a note

```
# Module title
Intro paragraph (3-5 sentences): what this is, why it matters in AI engineering work and interviews.

## Section title
<!-- topics: slug1, slug2 -->
Explanation...

## Quick recap
- You should be able to ...
```

* **Every curriculum topic of the module must be taught in some section.** The line right under the `##`
  heading lists the topic *slugs* (from `curriculum/*.toml`, run `study topics <module>`). `study notes-check <module>`
  fails on a missing/unknown slug. Group related topics into one section (a section = one idea, not one topic).
* 8-14 sections, then `## Quick recap` (no topics line needed) and optionally `## Interview angles`.
* Aim for 250-500 lines of markdown in total. Dense, clear, no padding.

## How to explain (this is what makes it useful)

1. **Build the mental model first**, in plain words and, where it helps, a small ASCII diagram in a ```text block.
   Explain *why* it works that way (what CPython/asyncio/pydantic actually does), not just what to type.
2. **Show it with code.** Every section has at least one short, runnable example. Show the printed result
   in a comment (`# prints: ...` / `# -> ...`) and make sure it is exactly right (run it!).
3. **Contrast**: wrong vs right, list vs generator, sync vs async. Contrasts are what interviews probe.
4. **Name the traps** with a callout (below). Use real traps, not trivia.
5. **Tie to AI engineering** where natural (chunking, embeddings, LLM calls, streaming, RAG), briefly.
6. End complex sections with a `Check yourself` block (below) of 1-2 questions.

## Callouts
Blockquotes starting with a bold keyword render as coloured boxes:

```
> **Remember** one-line rule worth memorising.
> **Gotcha** a trap and how it bites.
> **Interview** how this is asked and what a strong answer includes.
> **AI use** how it shows up building LLM/RAG systems.
```
Each callout is one blockquote (blank `>` lines allowed inside). Keep them short.

## Check yourself (collapsible)
```
<details markdown="1"><summary>Check yourself: why does X print Y?</summary>

Answer in 1-3 sentences.

</details>
```
(The blank lines inside are required.)

## Code blocks
* Language ```python only. Python 3.12+ syntax, typed where it helps, short (under ~25 lines).
* **First line marker (optional):** `# norun` = compile-check only (async main you do not want to execute, a snippet that
  intentionally raises, uses the network); `# nocheck` = skip checks (pseudo-code, intentionally invalid syntax).
  The marker line is hidden in the rendered page. Unmarked blocks are *executed* by `study notes-check --run`, so they must
  run standalone (own imports, no input, finish in seconds, no network, no files outside a temp dir).
* Use ```text for ASCII diagrams and terminal output; ```bash for commands (never executed).

## Accuracy rules
* Run every example (`study notes-check <module> --run` must be clean). The printed results you write must be what Python prints.
* The venv is CPython 3.13; flag behaviour that differs in 3.14 (free-threading, deferred annotations, ...).
* Do not invent library behaviour. Check versions installed in `.venv` (pydantic v2, httpx, numpy 2, pandas 3).
* Tables are fine for comparisons. Do not use emoji. Do not use HTML other than `<details>` and the topics comment.

## Voice
Direct, concrete, second person ("you"), short paragraphs. Explain like a senior colleague who wants the learner to
pass the interview *and* actually understand. No filler ("In this section we will..."). No hype.
