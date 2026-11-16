"""Progress store + spaced repetition + mastery scoring.

Storage is a single human-readable JSON file (``progress/progress.json``) so you can
commit it to git and literally watch yourself improve over time.

Signals that feed a topic's mastery:
  * self-ratings   (``study rate`` / ``study baseline``)      source = "rate" | "baseline"
  * quiz grades    (``study quiz``)                            source = "quiz"
  * mistakes       (``study mistake``)                         source = "mistake" (grade 1)
  * exercise tests (automatic, via the pytest plugin)          source = "test"

Scheduling uses a simplified SM-2 algorithm replayed over the review history, so the
schedule is always derived from data and never gets out of sync.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean
from typing import Any

from .paths import PROGRESS_FILE

SCHEMA_VERSION = 1
GRADE_SOURCES = ("rate", "baseline", "quiz", "mistake")


def today() -> date:
    override = os.environ.get("STUDY_TODAY")  # handy for testing the scheduler
    return date.fromisoformat(override) if override else date.today()


@dataclass(slots=True)
class SRState:
    reps: int = 0
    interval: int = 0
    ease: float = 2.5
    due: date | None = None
    last: date | None = None


def replay_sm2(reviews: list[dict[str, Any]]) -> SRState:
    st = SRState()
    for r in sorted(reviews, key=lambda r: r["date"]):
        d = date.fromisoformat(r["date"])
        g = int(r["grade"])
        same_day = st.last == d
        if g < 3:
            st.reps, st.interval = 0, 1
        elif not same_day:
            st.reps += 1
            if st.reps == 1:
                st.interval = 1
            elif st.reps == 2:
                st.interval = 3
            else:
                st.interval = max(1, round(st.interval * st.ease))
        st.ease = max(1.3, st.ease + 0.1 - (5 - g) * (0.08 + (5 - g) * 0.02))
        st.last = d
        st.due = d + timedelta(days=st.interval)
    return st


@dataclass(slots=True)
class TopicStats:
    topic_id: str
    mastery: int | None  # 0..100, None = no evidence at all
    self_avg: float | None  # 0..5
    tests_passed: int
    tests_failed: int
    tests_todo: int
    n_reviews: int
    due: date | None
    overdue_days: int

    @property
    def tests_attempted(self) -> int:
        return self.tests_passed + self.tests_failed

    @property
    def bucket(self) -> str:
        if self.mastery is None:
            return "unknown"
        if self.mastery >= 80:
            return "strong"
        if self.mastery >= 60:
            return "ok"
        return "weak"


class Progress:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data or {
            "version": SCHEMA_VERSION,
            "topics": {},
            "tests": {},
            "questions": {},
            "modules": {},
        }
        for key in ("topics", "tests", "questions", "modules"):
            self.data.setdefault(key, {})
        self._tests_by_topic: dict[str, list[dict[str, Any]]] | None = None

    # ---- persistence ---------------------------------------------------------
    @classmethod
    def load(cls) -> Progress:
        if PROGRESS_FILE.exists():
            return cls(json.loads(PROGRESS_FILE.read_text(encoding="utf-8")))
        return cls()

    def save(self) -> None:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: never leave a half-written progress file behind.
        fd, tmp = tempfile.mkstemp(dir=PROGRESS_FILE.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, PROGRESS_FILE)

    # ---- writes --------------------------------------------------------------
    def _topic(self, topic_id: str) -> dict[str, Any]:
        return self.data["topics"].setdefault(topic_id, {"reviews": []})

    def record_review(
        self, topic_id: str, grade: int, source: str, note: str = "", day: date | None = None
    ) -> None:
        if not 0 <= grade <= 5:
            raise ValueError("grade must be 0..5")
        day = day or today()
        entry: dict[str, Any] = {"date": day.isoformat(), "grade": grade, "source": source}
        if note:
            entry["note"] = note
        reviews = self._topic(topic_id)["reviews"]
        if source == "test":
            # one test-derived review per topic per day (last run of the day wins)
            reviews[:] = [r for r in reviews if not (r["source"] == "test" and r["date"] == entry["date"])]
        reviews.append(entry)

    def record_test(self, nodeid: str, status: str, topics: list[str], day: date | None = None) -> None:
        day = day or today()
        rec = self.data["tests"].setdefault(nodeid, {"fails": 0, "first_pass": None})
        prev = rec.get("status")
        rec.update(status=status, date=day.isoformat(), topics=sorted(set(topics)))
        if status == "failed" and prev != "failed":
            rec["fails"] += 1
        if status == "passed" and rec["first_pass"] is None:
            rec["first_pass"] = day.isoformat()
        self._tests_by_topic = None

    def record_question(self, qid: str, grade: int, day: date | None = None) -> None:
        day = day or today()
        rec = self.data["questions"].setdefault(qid, {"asked": 0, "grades": []})
        rec["asked"] += 1
        rec["last"] = day.isoformat()
        rec["grades"] = (rec["grades"] + [grade])[-5:]

    def set_module_status(self, module_id: str, status: str) -> None:
        self.data["modules"].setdefault(module_id, {})["status"] = status

    # ---- reads ---------------------------------------------------------------
    def module_status(self, module_id: str, default: str) -> str:
        return self.data["modules"].get(module_id, {}).get("status", default)

    def tests_for_topic(self, topic_id: str) -> list[dict[str, Any]]:
        if self._tests_by_topic is None:
            idx: dict[str, list[dict[str, Any]]] = {}
            for nodeid, rec in self.data["tests"].items():
                for t in rec.get("topics", []):
                    idx.setdefault(t, []).append({"nodeid": nodeid, **rec})
            self._tests_by_topic = idx
        return self._tests_by_topic.get(topic_id, [])

    def reviews(self, topic_id: str) -> list[dict[str, Any]]:
        return self.data["topics"].get(topic_id, {}).get("reviews", [])

    def stats(self, topic_id: str, on: date | None = None) -> TopicStats:
        on = on or today()
        reviews = self.reviews(topic_id)
        grades = [
            r["grade"] for r in sorted(reviews, key=lambda r: r["date"]) if r["source"] in GRADE_SOURCES
        ]
        recent = grades[-3:]
        self_avg = mean(recent) if recent else None

        tests = self.tests_for_topic(topic_id)
        passed = sum(t["status"] == "passed" for t in tests)
        failed = sum(t["status"] == "failed" for t in tests)
        todo = sum(t["status"] == "todo" for t in tests)

        signals: list[float] = []
        if self_avg is not None:
            signals.append(self_avg / 5)
        if passed + failed:
            signals.append(passed / (passed + failed))

        sr = replay_sm2(reviews)
        overdue = max(0, (on - sr.due).days) if sr.due else 0

        mastery: int | None = None
        if signals:
            m = mean(signals)
            if overdue:
                m *= max(0.6, 1 - 0.015 * overdue)  # forgetting curve, roughly
            mastery = round(m * 100)

        return TopicStats(
            topic_id=topic_id,
            mastery=mastery,
            self_avg=self_avg,
            tests_passed=passed,
            tests_failed=failed,
            tests_todo=todo,
            n_reviews=len(reviews),
            due=sr.due,
            overdue_days=overdue,
        )
