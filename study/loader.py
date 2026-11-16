"""Load the module under test: your exercise file, or the reference solution.

Test files never import exercises directly; they do::

    from study.loader import load
    ex = load(__file__, "ex01_lazy_reader")

Normally this loads ``ex01_lazy_reader.py`` next to the test file (YOUR code).
With ``STUDY_SOLUTIONS=1`` (``study verify``) it loads the mirror file under
``solutions/`` instead — used to prove the tests themselves are correct.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from .paths import EXERCISES_DIR, SOLUTIONS_DIR


def using_solutions() -> bool:
    return os.environ.get("STUDY_SOLUTIONS") == "1"


def load(test_file: str, name: str) -> ModuleType:
    test_dir = Path(test_file).resolve().parent
    rel = test_dir.relative_to(EXERCISES_DIR)
    base = SOLUTIONS_DIR / rel if using_solutions() else test_dir
    path = base / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    tag = "sol" if using_solutions() else "ex"
    modname = "_study_" + tag + "_" + "_".join(rel.parts) + "_" + name
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod  # needed for dataclasses / pickling inside exercises
    sys.path.insert(0, str(base))  # lets an exercise import a sibling helper module
    try:
        spec.loader.exec_module(mod)
    except NotImplementedError:
        # e.g. a still-stubbed decorator applied at import time: every test is "todo"
        sys.modules.pop(modname, None)
        return _TodoModule(name)  # type: ignore[return-value]
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    finally:
        sys.path.remove(str(base))
    return mod


class _TodoModule:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str):  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            f"{self._name}.py raised NotImplementedError while importing "
            f"(a stubbed function is called at import time, e.g. a decorator)"
        )
