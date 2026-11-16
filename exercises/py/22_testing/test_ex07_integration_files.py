import json
import shutil

import pytest

from study.loader import load

ex = load(__file__, "ex07_integration_files")


def assert_suite_fails(run_suite, *args):
    """Like a red pytest run: ANY exception counts as "the suite caught the bug".
    NotImplementedError is re-raised so an unsolved stub shows up as TODO, not as a pass."""
    try:
        run_suite(*args)
    except NotImplementedError:
        raise
    except (Exception, pytest.fail.Exception):
        return
    raise AssertionError("your suite PASSED on a buggy implementation - add a check for it")


def _make(
    *,
    key=lambda t: " ".join(t.split()).casefold(),
    keep_last=False,
    ensure_ascii=False,
    leave_tmp=False,
    only_text=False,
    trailing_newline=True,
    count_blank=False,
):
    def impl(src, dst):
        stats = {"read": 0, "written": 0, "duplicates": 0, "invalid": 0}
        kept: dict[str, dict] = {}
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                stats["read"] += count_blank
                continue
            stats["read"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                stats["invalid"] += 1
                continue
            k = key(record["text"])
            if k in kept:
                stats["duplicates"] += 1
                if keep_last:
                    del kept[k]
                    kept[k] = record
                continue
            kept[k] = record
        out = [
            json.dumps({"text": r["text"]} if only_text else r, ensure_ascii=ensure_ascii)
            for r in kept.values()
        ]
        body = "\n".join(out) + ("\n" if trailing_newline else "")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        if leave_tmp:
            shutil.copyfile(tmp, dst)
        else:
            tmp.replace(dst)
        stats["written"] = len(out)
        return stats

    return impl


MUTANTS = {
    "case_sensitive_dedupe": _make(key=lambda t: " ".join(t.split())),
    "whitespace_not_collapsed": _make(key=lambda t: t.casefold()),
    "keeps_last_duplicate": _make(keep_last=True),
    "ascii_escaped_output": _make(ensure_ascii=True),
    "temp_file_left_behind": _make(leave_tmp=True),
    "drops_extra_fields": _make(only_text=True),
    "no_trailing_newline": _make(trailing_newline=False),
    "counts_blank_lines": _make(count_blank=True),
}


@pytest.mark.topic("py.testing.integration", "py.testing.fixtures")
def test_suite_passes_on_correct_impl(tmp_path):
    ex.check_dedupe(ex.dedupe_jsonl, tmp_path)


@pytest.mark.topic("py.testing.integration")
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_suite_catches_mutant(name, tmp_path):
    assert_suite_fails(ex.check_dedupe, MUTANTS[name], tmp_path)


@pytest.mark.topic("py.testing.integration", "py.testing.coverage")
def test_suite_works_in_any_directory(tmp_path):
    # the suite must only use the directory it is given (no hard-coded paths, no cwd)
    nested = tmp_path / "deeply" / "nested"
    nested.mkdir(parents=True)
    ex.check_dedupe(ex.dedupe_jsonl, nested)
    assert any(nested.iterdir()), "the suite should create its input files inside workdir"
