# AI Engineering Prep

Practice material and a tracker to get you interview-ready for AI engineering (Python, LLMs, RAG, agents).
It tells you what you're weak at, what to revise today, and gives you exercises and questions to do it.

## Start in 3 steps

1. Install [uv](https://docs.astral.sh/uv/) once: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. In this folder run: `./study.sh`
3. Rate yourself on the topics you've studied (about 10 minutes). Done.

The first run sets everything up by itself. (Windows: use WSL.)

## Every day

Run `./study.sh` and do what it says. It's usually three things:

| Do this | Command | Time |
|---|---|---|
| Answer a few interview questions | `./study.sh quiz --due` | 15 min |
| Solve one exercise (open the file it names, write the code) | `./study.sh test <module>` | 45 min |
| Note anything you got wrong | `./study.sh mistake "what I got wrong" -t <topic>` | 2 min |

Grade yourself honestly (0 to 5). Topics you do badly on come back sooner.

## Read the notes in your browser

`./study.sh notes` opens an interactive guide with diagrams and small demos. It works offline.
On Windows (WSL) it also copies the notes to `Documents\AI-Prep-Notes`, so you can double-click `index.html` there any time.

## Want more?

- Every command, how scoring works, adding new topics: `docs/REFERENCE.md` (also in the notes)
- The project you build step by step (DocMind): `capstone/ROADMAP.md`
- Stuck or something broke? Run `./study.sh --help`, or see Troubleshooting in `docs/REFERENCE.md`.
