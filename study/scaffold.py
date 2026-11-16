"""Create the file skeleton for a (new) module so future tracks plug straight in."""

from __future__ import annotations

from pathlib import Path

from .curriculum import Module

EX_TEMPLATE = '''"""
Exercise {num:02d}.1 - <title>                       level: ★☆☆
topics: {first_topic}

Problem
-------
<what to build, inputs/outputs, edge cases>

Interviewers look for
---------------------
<the concept underneath / trade-offs>

Hints (only after 15 honest minutes)
------------------------------------
1. ...
"""


def solve(*args, **kwargs):
    raise NotImplementedError
'''

TEST_TEMPLATE = """import pytest

from study.loader import load

ex = load(__file__, "ex01_example")


@pytest.mark.topic("{first_topic}")
def test_example():
    assert ex.solve() is not None
"""

Q_TEMPLATE = '''# Interview question bank for {title}
# level: 1 = fundamentals, 2 = mid, 3 = senior (3-4 yrs)  kind: concept | predict | debug | design | compare

[[q]]
id = "{slug}-01"
topic = "{first_slug}"
level = 1
kind = "concept"
q = """<question>"""
a = """<key points an interviewer listens for>"""
followups = []
'''


def scaffold(m: Module) -> list[Path]:
    created: list[Path] = []
    first = m.topics[0]
    fmt = {"num": m.num, "first_topic": first.id, "first_slug": first.slug, "title": m.title, "slug": m.slug}
    files = {
        m.exercise_dir / "ex01_example.py": EX_TEMPLATE.format(**fmt),
        m.exercise_dir / "test_ex01_example.py": TEST_TEMPLATE.format(**fmt),
        m.solution_dir / "ex01_example.py": EX_TEMPLATE.format(**fmt).replace(
            "raise NotImplementedError", "return True"
        ),
        m.questions_file: Q_TEMPLATE.format(**fmt),
    }
    for path, text in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        created.append(path)
    return created
