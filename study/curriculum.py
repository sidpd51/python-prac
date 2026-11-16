"""Curriculum registry.

Every track (python, ml, llm, rag, agents, prod, interview, ...) is one TOML file in
``curriculum/``. Adding a new track = adding a new TOML file; nothing else changes.

Topic ids are hierarchical:  ``<track>.<module>.<topic>``  e.g. ``py.generators.send``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .paths import CURRICULUM_DIR, EXERCISES_DIR, QUESTIONS_DIR, SOLUTIONS_DIR

VALID_STATUS = ("covered", "learning", "planned")


@dataclass(frozen=True, slots=True)
class Topic:
    id: str
    slug: str
    title: str
    module_id: str


@dataclass(frozen=True, slots=True)
class Module:
    id: str
    slug: str
    num: int
    title: str
    track_id: str
    default_status: str
    topics: tuple[Topic, ...]

    @property
    def dirname(self) -> str:
        return f"{self.num:02d}_{self.slug}"

    @property
    def exercise_dir(self) -> Path:
        return EXERCISES_DIR / self.track_id / self.dirname

    @property
    def solution_dir(self) -> Path:
        return SOLUTIONS_DIR / self.track_id / self.dirname

    @property
    def questions_file(self) -> Path:
        return QUESTIONS_DIR / self.track_id / f"{self.dirname}.toml"


@dataclass(frozen=True, slots=True)
class Track:
    id: str
    title: str
    order: int
    description: str
    modules: tuple[Module, ...]


class CurriculumError(ValueError):
    pass


class Curriculum:
    def __init__(self, tracks: list[Track]) -> None:
        self.tracks = sorted(tracks, key=lambda t: t.order)
        self.modules: dict[str, Module] = {}
        self.topics: dict[str, Topic] = {}
        for track in self.tracks:
            for module in track.modules:
                if module.id in self.modules:
                    raise CurriculumError(f"duplicate module id {module.id}")
                self.modules[module.id] = module
                for topic in module.topics:
                    if topic.id in self.topics:
                        raise CurriculumError(f"duplicate topic id {topic.id}")
                    self.topics[topic.id] = topic

    # ---- lookups -------------------------------------------------------------
    @staticmethod
    def _resolve(ref: str, ids: list[str], kind: str) -> str:
        if ref in ids:
            return ref
        # Allow suffix match ("generators.send" -> "py.generators.send")
        # and module dir names ("08_generators" -> "py.generators").
        ref_norm = ref.split("_", 1)[1] if ref[:2].isdigit() and "_" in ref else ref
        matches = [i for i in ids if i.endswith("." + ref_norm) or i == ref_norm]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise CurriculumError(f"unknown {kind} '{ref}'")
        raise CurriculumError(f"ambiguous {kind} '{ref}': {', '.join(sorted(matches))}")

    def topic(self, ref: str) -> Topic:
        return self.topics[self._resolve(ref, list(self.topics), "topic")]

    def module(self, ref: str) -> Module:
        return self.modules[self._resolve(ref, list(self.modules), "module")]

    def track(self, ref: str) -> Track:
        for t in self.tracks:
            if t.id == ref:
                return t
        raise CurriculumError(f"unknown track '{ref}'")

    def module_of(self, topic_id: str) -> Module:
        return self.modules[self.topics[topic_id].module_id]

    def module_for_path(self, path: Path) -> Module | None:
        path = path.resolve()
        for m in self.modules.values():
            for base in (m.exercise_dir, m.solution_dir):
                if path == base or base in path.parents:
                    return m
        return None


def _load_track(path: Path) -> Track:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    t = data["track"]
    track_id = t["id"]
    modules: list[Module] = []
    for m in data.get("module", []):
        module_id = f"{track_id}.{m['slug']}"
        status = m.get("status", "planned")
        if status not in VALID_STATUS:
            raise CurriculumError(f"{path.name}: bad status {status!r} for {module_id}")
        topics = tuple(
            Topic(id=f"{module_id}.{slug}", slug=slug, title=title, module_id=module_id)
            for slug, title in m["topics"]
        )
        modules.append(
            Module(
                id=module_id,
                slug=m["slug"],
                num=int(m["num"]),
                title=m["title"],
                track_id=track_id,
                default_status=status,
                topics=topics,
            )
        )
    return Track(
        id=track_id,
        title=t["title"],
        order=int(t.get("order", 99)),
        description=t.get("description", ""),
        modules=tuple(sorted(modules, key=lambda x: x.num)),
    )


@cache
def load_curriculum() -> Curriculum:
    return Curriculum([_load_track(p) for p in sorted(CURRICULUM_DIR.glob("*.toml"))])
