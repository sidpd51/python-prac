import pytest

from study.loader import load

ex = load(__file__, "ex02_csv_eval_results")

CSV = (
    "id,model,score,passed,latency_ms,tags,notes\n"
    "q1,gpt-4o,0.92,true,812,rag;faq,x\n"
    "q2,claude,0.40, NO ,,,\n"
    "q3,llama,1,Yes,5, a ; b ;,\n"
)


@pytest.mark.topic("py.files.csv", "py.files.read_write")
def test_read_and_coerce(tmp_path):
    p = tmp_path / "evals.csv"
    p.write_text(CSV, encoding="utf-8")
    rows = ex.read_eval_csv(p)
    assert rows == [
        ex.EvalRow("q1", "gpt-4o", 0.92, True, 812, ["rag", "faq"]),
        ex.EvalRow("q2", "claude", 0.40, False, None, []),
        ex.EvalRow("q3", "llama", 1.0, True, 5, ["a", "b"]),
    ]
    assert isinstance(rows[0].score, float) and isinstance(rows[0].latency_ms, int)


@pytest.mark.topic("py.files.csv")
def test_optional_columns_absent(tmp_path):
    p = tmp_path / "min.csv"
    p.write_text("model,id,passed,score\nm,q1,0,0.5\n", encoding="utf-8")
    assert ex.read_eval_csv(p) == [ex.EvalRow("q1", "m", 0.5, False, None, [])]


@pytest.mark.topic("py.files.csv")
def test_missing_required_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,model,passed\nq1,m,true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="score"):
        ex.read_eval_csv(p)


@pytest.mark.topic("py.files.csv")
@pytest.mark.parametrize(
    ("row", "line"),
    [
        ("q9,m,high,true,,", 3),
        ("q9,m,0.1,maybe,,", 3),
        ("q9,m,0.1,true,12ms,", 3),
    ],
)
def test_bad_values_report_line(tmp_path, row, line):
    p = tmp_path / "bad.csv"
    p.write_text(f"id,model,score,passed,latency_ms,tags\nq1,m,1,true,,\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"line {line}"):
        ex.read_eval_csv(p)


@pytest.mark.topic("py.files.csv")
def test_bad_float_is_chained(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,model,score,passed\nq1,m,abc,true\n", encoding="utf-8")
    with pytest.raises(ValueError) as info:
        ex.read_eval_csv(p)
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.topic("py.files.csv", "py.files.text_vs_binary")
def test_embedded_crlf_survives_reading(tmp_path):
    p = tmp_path / "crlf.csv"
    p.write_bytes(b'id,model,score,passed\r\n"q1","line1\r\nline2",0.5,true\r\n')
    rows = ex.read_eval_csv(p)
    assert rows[0].model == "line1\r\nline2", "open the csv with newline=''"


@pytest.mark.topic("py.files.csv", "py.files.with")
def test_round_trip_tricky_fields(tmp_path):
    rows = [
        ex.EvalRow("q,1", 'say "hi"', 0.1, True, 0, ["a", "b"]),
        ex.EvalRow("q2", "multi\nline\r\nanswer", 1 / 3, False, None, []),
        ex.EvalRow("q3", "ünïcødé ✓", 1e-9, True, 12345, ["x"]),
    ]
    p = tmp_path / "out.csv"
    assert ex.write_eval_csv(p, rows) == 3
    assert ex.read_eval_csv(p) == rows
    header = p.read_bytes().split(b"\r\n", 1)[0]
    assert header == b"id,model,score,passed,latency_ms,tags"


@pytest.mark.topic("py.files.csv")
def test_written_values(tmp_path):
    p = tmp_path / "out.csv"
    ex.write_eval_csv(p, [ex.EvalRow("q1", "m", 0.5, False, None, ["rag", "faq"])])
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "q1,m,0.5,false,,rag;faq"
