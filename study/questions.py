"""Interview question bank loader.

One TOML per module at ``questions/<track>/<NN_module>.toml``::

    [[q]]
    id = "gen-01"                      # unique within the file
    topic = "send"                     # topic slug in this module (or full id)
    level = 2                          # 1 = fundamentals, 2 = mid, 3 = senior / 3-4 yrs exp
    kind = "concept"                   # concept | predict | debug | design | compare
    q = '''question text (may contain ```python code fences```)'''
    a = '''model answer / key points an interviewer listens for'''
    followups = ["...", "..."]         # optional: what they'll ask next
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field

from .curriculum import Curriculum, Module

KINDS = ("concept", "predict", "debug", "design", "compare")


@dataclass(frozen=True, slots=True)
class Question:
    qid: str  # "<module id>.<id>"
    topic_id: str
    level: int
    kind: str
    q: str
    a: str
    followups: tuple[str, ...] = field(default=())


def load_module_questions(cur: Curriculum, module: Module) -> list[Question]:
    path = module.questions_file
    if not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: list[Question] = []
    seen: set[str] = set()
    for raw in data.get("q", []):
        qid = f"{module.id}.{raw['id']}"
        if qid in seen:
            raise ValueError(f"{path.name}: duplicate question id {raw['id']}")
        seen.add(qid)
        topic_ref = raw["topic"]
        topic_id = topic_ref if topic_ref in cur.topics else f"{module.id}.{topic_ref}"
        if topic_id not in cur.topics:
            raise ValueError(f"{path.name}: question {raw['id']} has unknown topic {topic_ref!r}")
        kind = raw.get("kind", "concept")
        if kind not in KINDS:
            raise ValueError(f"{path.name}: question {raw['id']} has bad kind {kind!r}")
        out.append(
            Question(
                qid=qid,
                topic_id=topic_id,
                level=int(raw.get("level", 1)),
                kind=kind,
                q=raw["q"].strip(),
                a=raw["a"].strip(),
                followups=tuple(raw.get("followups", ())),
            )
        )
    return out


def load_all_questions(cur: Curriculum) -> list[Question]:
    return [q for m in cur.modules.values() for q in load_module_questions(cur, m)]
