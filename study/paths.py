"""Filesystem layout of the prep repo. Everything is resolved relative to the repo root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRICULUM_DIR = ROOT / "curriculum"
EXERCISES_DIR = ROOT / "exercises"
SOLUTIONS_DIR = ROOT / "solutions"
QUESTIONS_DIR = ROOT / "questions"
PROGRESS_FILE = ROOT / "progress" / "progress.json"
JOURNAL_DIR = ROOT / "journal"
QUIZ_LOG = JOURNAL_DIR / "quiz_log.jsonl"
MISTAKES_FILE = JOURNAL_DIR / "mistakes.md"
