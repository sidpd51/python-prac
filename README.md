# AI Engineering Prep

Revise the concepts, practise interview questions and coding exercises, and see what you're weak at.
Everything runs in your browser. No terminal needed.

## Open it

Double-click **`docs/site/index.html`**. On Windows with WSL, the same files are copied to
`Documents\AI-Prep-Notes`, so double-click `index.html` there.

It works offline, and your progress is saved in that browser. (Use **My topics → Export progress** to back it up
or move it to another browser.)

## Every day

1. **Today**: press *Start today's 10 questions*. Answer each out loud, reveal the key points, grade yourself 0 to 5.
2. **Revise**: open the note for a topic you're weak on, read one section, tick it as revised, press *Quiz me on this*.
3. **Exercises**: do one. Write it in your editor, then compare with the reference solution and rate how it went.

First time: open **My topics** and rate what you already know (about 10 minutes, be honest). That's what lets the
app show you what to work on.

## What's inside

| Page | What it's for |
|------|---------------|
| **Today** | what to do now: due reviews, weakest topics, one-click session |
| **Revise** | a theory note for every module, with runnable examples, traps and interview angles |
| **Quiz** | interview questions at three levels, spaced repetition decides what comes back |
| **Exercises** | coding problems with starter code and a reference solution |
| **My topics** | rate every topic, see mastery, back up your progress |
| **Explainers** | small interactive demos (lazy pipelines, async, retries) |
| **Guide** | the capstone project (DocMind), how the system works, how to add content |

## Optional: terminal helper

You don't need it. It adds two things: automatic test runs for exercises, and rebuilding the pages after you edit a
note or question file.

```bash
./study.sh test generators     # run the tests for one module
./study.sh notes               # rebuild the pages and refresh the Windows copy
```
Setup and every command are in `docs/REFERENCE.md` (also under **Guide**).
