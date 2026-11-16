# Files

Files are where your program meets the outside world: configs, logs, prompt templates, and above all the
corpora that feed RAG and fine-tuning. The Python file API is small, but the traps are real: forgetting to
close handles, text versus bytes, the wrong encoding on Windows, loading a 20 GB file into RAM, and writing a
half-finished file when the process dies. Interviewers ask about `with`, modes, encodings, `pathlib`, and how
you would process a file that does not fit in memory.

Every example below builds its files in a temporary directory so it runs standalone.

## Opening and closing files
<!-- topics: open_close, open -->

`open(path, mode="r", encoding=None, ...)` asks the OS for a **file descriptor** and returns a file object that
wraps it with a buffer. Descriptors are a limited resource, and written data may sit in the buffer until you
flush or close. So every `open` needs a matching `close`.

```python
import tempfile, os

d = tempfile.mkdtemp()
path = os.path.join(d, "a.txt")

f = open(path, "w", encoding="utf-8")
f.write("hello")
print(f.closed)       # prints: False
f.close()             # flushes the buffer and releases the descriptor
print(f.closed)       # prints: True
try:
    f.write("more")
except ValueError as e:
    print(e)          # prints: I/O operation on closed file.
```

Manual `close()` is skipped if anything raises between `open` and `close`. CPython often closes an unreferenced
file when it is garbage collected, but that is an implementation detail (PyPy differs), and you get a
`ResourceWarning`. Use `with` (next sections).

Useful `open` parameters: `encoding` (always pass it for text), `newline` (control line-ending translation),
`buffering` (0 only for binary), `errors` (`"strict"`, `"replace"`, `"ignore"`), `closefd`.

> **Gotcha** A write that "succeeded" can still be lost: data lives in Python's buffer and the OS cache.
> `flush()` empties Python's buffer, `os.fsync(f.fileno())` asks the OS to hit the disk.

## File modes
<!-- topics: modes -->

| Mode | Meaning | Creates? | Truncates? | Position |
|---|---|---|---|---|
| `r` | read (default) | no, error if missing | no | start |
| `w` | write | yes | **yes, immediately** | start |
| `a` | append | yes | no | always end |
| `x` | exclusive create | yes, error if exists | n/a | start |
| `r+` | read and write | no | no | start |
| `w+` | read and write | yes | yes | start |

Add `b` for binary (`rb`, `wb`), `t` (default) for text.

```python
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), "m.txt")

with open(p, "w") as f:
    f.write("one\n")
with open(p, "a") as f:
    f.write("two\n")
with open(p) as f:
    print(f.read().splitlines())    # prints: ['one', 'two']

try:
    open(p, "x")
except FileExistsError:
    print("x refuses to overwrite")   # prints: x refuses to overwrite

with open(p, "w") as f:
    pass                              # opening with "w" already emptied it
print(os.path.getsize(p))             # prints: 0
```

> **Gotcha** `"w"` destroys the existing content the moment you open it, before you write anything. If your
> code then crashes, you have lost the old data too. See atomic writes below.

## Reading and writing
<!-- topics: read_write -->

| Call | Returns | Use |
|---|---|---|
| `f.read()` | whole content as one string | small files |
| `f.read(n)` | up to n characters/bytes | chunked reads |
| `f.readline()` | one line including `\n` | manual loops |
| `f.readlines()` | list of all lines | small files |
| `for line in f` | lines lazily | **default for text** |
| `f.write(s)` | number of chars written | no newline added |
| `f.writelines(seq)` | None | no newlines added either |

```python
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), "r.txt")
with open(p, "w") as f:
    print(f.write("a\nb\nc\n"))        # prints: 6
    f.writelines(["d\n", "e\n"])        # you must supply the \n yourself

with open(p) as f:
    print(f.readline())                 # prints: a (followed by a blank line, the \n is kept)
    print(f.read(3).replace("\n", "|")) # prints: b|c
    print(f.tell())                     # prints: 5
    f.seek(0)
    print([line.strip() for line in f]) # prints: ['a', 'b', 'c', 'd', 'e']
```

The file object is an **iterator** over lines (the same one-pass behaviour as generators): after the loop, it is
exhausted until you `seek(0)`. `print(line)` shows blank lines because the line already carries `\n`; use
`line.rstrip("\n")` or `print(line, end="")`.

> **Remember** `write` and `writelines` never add newlines. `read()` on a huge file loads all of it.

## The with statement for files
<!-- topics: with -->

`with open(...) as f:` closes the file on every exit path: normal completion, `return`, `break` or an
exception. It is `try/finally: f.close()` in one line (the protocol behind it is in the context managers module).

```python
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), "w.txt")

def write_then_fail():
    global f
    with open(p, "w") as f:
        f.write("data")
        raise RuntimeError("boom")

try:
    write_then_fail()
except RuntimeError:
    pass
print(f.closed)           # prints: True
print(open(p).read())     # prints: data
```

The file was flushed and closed even though the function raised. After the block the name `f` is still bound, but the object is closed. Open two files in one statement: `with open(a) as src, open(b, "w") as dst:`.

> **Interview** "Why `with` and not `close()`?" Exception safety and no leaked descriptors. A leak shows up as
> `OSError: Too many open files` in a long-running service.

## Text versus binary and encoding
<!-- topics: text_vs_binary, encoding -->

Disks store **bytes**. **Text mode** decodes those bytes into `str` using an encoding and translates line
endings; **binary mode** gives you raw `bytes` untouched.

```text
disk bytes  --(decode with encoding)-->  str      mode "r"  / "w"   (adds newline translation)
disk bytes  ------------ as is ------->  bytes    mode "rb" / "wb"
```

```python
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), "e.txt")

with open(p, "w", encoding="utf-8") as f:
    f.write("café")

with open(p, "rb") as f:
    raw = f.read()
print(raw)                         # prints: b'caf\xc3\xa9'
print(len("café"), len(raw))       # prints: 4 5
print(raw.decode("utf-8"))         # prints: café
print(raw.decode("latin-1"))       # prints: cafÃ©

try:
    raw.decode("ascii")
except UnicodeDecodeError as e:
    print("decode failed at byte", e.start)   # prints: decode failed at byte 3
```

UTF-8 uses 1 to 4 bytes per character, so characters and bytes are not the same count. Decoding with the wrong
encoding either raises `UnicodeDecodeError` or, worse, produces **mojibake** (`cafÃ©`) silently.

Rules:
* **Always pass `encoding="utf-8"`** for text. The default is locale-dependent (often cp1252 on Windows,
  UTF-8 on modern Linux/macOS). Python 3.15 will make UTF-8 the default everywhere (PEP 686); do not rely on it today.
* `errors="replace"` swaps undecodable bytes for the replacement character, handy for dirty scraped text.
  `"ignore"` drops them silently, which hides data loss.
* `utf-8-sig` strips the BOM Excel likes to add to CSV files.
* Use binary for images, PDFs, pickles, and anything where you must not alter bytes.
* Text mode translates `\r\n` to `\n` on read; pass `newline=""` to disable (required for the `csv` module).

> **AI use** Scraped web text and PDFs are full of bad bytes. A corpus loader that opens with
> `encoding="utf-8", errors="replace"` and logs how many replacements happened beats one that crashes at row 4 million.
> Tokenizers also count tokens, not characters or bytes: all three differ.

<details markdown="1"><summary>Check yourself: why is `len("café")` 4 but the file 5 bytes?</summary>

The `é` encodes to two bytes in UTF-8 (`c3 a9`). `len` of a `str` counts characters (code points); the file stores bytes.

</details>

## CSV and JSON
<!-- topics: csv, json -->

**csv** handles quoting, embedded commas and newlines. Do not split on `","` by hand. Open with `newline=""`.

```python
import csv, io

buf = io.StringIO()
w = csv.writer(buf)
w.writerow(["id", "text"])
w.writerow([1, 'He said "hi", twice'])
print(buf.getvalue().splitlines()[1])      # prints: 1,"He said ""hi"", twice"

buf.seek(0)
for row in csv.DictReader(buf):
    print(row)
# prints: {'id': '1', 'text': 'He said "hi", twice'}
```

Everything read from CSV is a **string**; convert types yourself (or use pandas). `DictReader`/`DictWriter` map
rows to header names.

**json** round-trips dicts, lists, strings, numbers, bools and `None`. `load/dump` work on file objects,
`loads/dumps` on strings.

```python
import json, tempfile, os
p = os.path.join(tempfile.mkdtemp(), "c.json")
cfg = {"model": "m1", "temperature": 0.2, "stop": None, "name": "naïve"}

with open(p, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
with open(p, encoding="utf-8") as f:
    print(json.load(f) == cfg)               # prints: True

print(json.dumps({"a": (1, 2)}))             # prints: {"a": [1, 2]}   tuple becomes list
try:
    json.dumps({"t": {1, 2}})
except TypeError as e:
    print(e)                                 # prints: Object of type set is not JSON serializable
```

Traps: non-string dict keys become strings; tuples come back as lists; `datetime`, `set` and custom classes need
`default=` or a custom encoder; `ensure_ascii=False` keeps non-ASCII readable.

**JSONL** (one JSON object per line) is the standard container for large LLM datasets because you can stream it
and append to it. Do not `json.load` a 10 GB array:

```python
import json, tempfile, os
p = os.path.join(tempfile.mkdtemp(), "data.jsonl")
rows = [{"prompt": "a", "label": 1}, {"prompt": "b", "label": 0}]

with open(p, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

with open(p, encoding="utf-8") as f:
    for line in f:
        print(json.loads(line)["prompt"])
# prints: a
# prints: b
```

## pathlib
<!-- topics: pathlib -->

`pathlib.Path` replaces `os.path` string-juggling with objects. The `/` operator joins parts portably.

```python
from pathlib import Path
import tempfile

root = Path(tempfile.mkdtemp())
p = root / "docs" / "report.final.md"
print(p.name, p.stem, p.suffix)           # prints: report.final.md report.final .md
print(p.with_suffix(".txt").name)         # prints: report.final.txt
print(p.parent.name)                      # prints: docs
print(p.exists())                         # prints: False

p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("# hi\n", encoding="utf-8")
print(p.read_text(encoding="utf-8"))      # prints: # hi
print(p.stat().st_size)                   # prints: 5
print(p.relative_to(root))                # prints: docs/report.final.md
```

| os / open style | pathlib |
|---|---|
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.exists(p)` | `p.exists()` |
| `os.makedirs(p, exist_ok=True)` | `p.mkdir(parents=True, exist_ok=True)` |
| `open(p).read()` | `p.read_text(encoding=...)` |
| `os.path.splitext` | `p.suffix`, `p.stem` |
| `os.listdir`, `glob` | `p.iterdir()`, `p.glob("*.md")`, `p.rglob("*.md")` |

`Path.home()`, `Path.cwd()`, `p.resolve()` (absolute, symlinks resolved), `p.is_file()`, `p.is_dir()` complete the
toolbox. `read_text` loads the whole file, so it is for small files only.

> **Gotcha** `Path("a") / "/etc"` yields `/etc`: an absolute right-hand side discards the left. Never join
> untrusted user input to a base path without checking that `resolve()` stays inside the base (path traversal).

## Directory and file operations
<!-- topics: dir_ops -->

```python
from pathlib import Path
import shutil, tempfile

root = Path(tempfile.mkdtemp())
(root / "a").mkdir()
(root / "a" / "x.md").write_text("x")
(root / "a" / "y.txt").write_text("y")
(root / "b").mkdir()

print(sorted(p.name for p in root.iterdir()))        # prints: ['a', 'b']
print(sorted(p.name for p in root.rglob("*.md")))    # prints: ['x.md']

(root / "a" / "x.md").rename(root / "b" / "x.md")    # move/rename
shutil.copy(root / "a" / "y.txt", root / "b")        # copy file
print(sorted(p.name for p in (root / "b").iterdir()))  # prints: ['x.md', 'y.txt']

(root / "a" / "y.txt").unlink()                      # delete a file
shutil.rmtree(root / "b")                            # delete a tree (dangerous, no undo)
print(sorted(p.name for p in root.iterdir()))        # prints: ['a']
```

`Path.rmdir()` removes only an empty directory. `os.walk(top)` yields `(dirpath, dirnames, filenames)` and
lets you prune the walk; `rglob` is simpler when you just need matching files. `iterdir` and `glob` return
generators in unspecified order, so `sorted(...)` for deterministic pipelines. `shutil.copytree`, `shutil.move`
and `os.replace` (atomic rename) round out the set.

## Temporary files
<!-- topics: tempfiles -->

Use `tempfile` instead of inventing `/tmp/foo.txt`: names are unique, permissions are safe, and cleanup is
automatic.

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as d:          # directory removed on exit
    p = Path(d) / "x.txt"
    p.write_text("scratch")
    print(p.exists())                             # prints: True
print(p.exists())                                 # prints: False

with tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8") as f:
    f.write("tmp")
    f.flush()
    print(Path(f.name).read_text())               # prints: tmp
```

`TemporaryFile` has no visible name; `NamedTemporaryFile` does (on Windows you cannot reopen it while open unless
`delete=False`). Pytest's `tmp_path` fixture does the same for tests.

**Atomic write pattern**: write to a temp file in the *same directory*, then `os.replace` it over the target.
Readers see the old file or the new one, never a half-written one.

```python
import os, tempfile
from pathlib import Path

def atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent)   # same filesystem, so replace is atomic
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise

target = Path(tempfile.mkdtemp()) / "state.json"
atomic_write(target, '{"v": 1}')
atomic_write(target, '{"v": 2}')
print(target.read_text())                         # prints: {"v": 2}
```

> **AI use** Checkpointing embeddings or an eval-run state file this way means a crash mid-write never corrupts
> yesterday's checkpoint.

## Large file processing
<!-- topics: large_files -->

Never `read()` or `readlines()` a file you do not know the size of. Iterate, so memory stays flat:

```python
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), "big.txt")
with open(p, "w") as f:
    for i in range(1000):
        f.write(f"line {i}\n")

count = 0
with open(p) as f:
    for line in f:              # one line in memory at a time (buffered under the hood)
        if line.endswith("7\n"):
            count += 1
print(count)                    # prints: 100
```

Strategies:
* **Line iteration** for text/JSONL; wrap in a generator that parses and yields records.
* **Fixed-size chunks** for binary or one giant line: `while chunk := f.read(1 << 20): ...`.
* **Batches**: group N records (embedding APIs take lists) with `itertools.batched` (3.12+).
* **`mmap`** for random access into huge binary files without loading them.
* **pandas `read_csv(chunksize=...)`** or Parquet row groups for tables.
* Write results incrementally (append JSONL) so a crash does not lose everything, and make jobs resumable by
  recording the last processed offset or id.

```python
import tempfile, os
from itertools import batched

p = os.path.join(tempfile.mkdtemp(), "b.bin")
with open(p, "wb") as f:
    f.write(bytes(range(10)))

with open(p, "rb") as f:
    sizes = []
    while chunk := f.read(4):          # walrus: read until empty bytes
        sizes.append(len(chunk))
print(sizes)                           # prints: [4, 4, 2]
print(list(batched(range(5), 2)))      # prints: [(0, 1), (2, 3), (4,)]
```

> **Interview** "Process a 50 GB file on a 16 GB machine?" Stream it: iterate lines or chunks, keep running
> aggregates or emit results incrementally, avoid `readlines`/`json.load`/`read_text`, and make the job restartable.

## Quick recap
- `open` hands out a file descriptor plus a buffer; always close, which `with` guarantees on every exit path.
- `w` truncates on open, `a` appends, `x` refuses to overwrite, `r+` reads and writes without truncating.
- Iterate a file (`for line in f`) instead of `readlines()`; `write` never adds newlines.
- Text = bytes decoded with an encoding; always pass `encoding="utf-8"`; wrong decoding means errors or mojibake.
- Use `csv` (with `newline=""`) and `json`; CSV values are all strings; JSON turns tuples to lists and keys to strings.
- JSONL (one object per line) streams and appends: the default format for big LLM datasets.
- `pathlib.Path` with `/`, `read_text`, `mkdir(parents=True, exist_ok=True)`, `glob`/`rglob` replaces `os.path`.
- `tempfile` gives safe unique scratch space; write to temp then `os.replace` for atomic updates.
- Large files: stream lines or chunks, batch, write incrementally, keep jobs resumable.
