import pytest

from study.loader import load

ex = load(__file__, "ex01_bytecode_dis")

MAX_TOKENS = 4096


def add(a, b):
    return a + b


def total_len(chunks):
    total = 0
    for c in chunks:
        total += len(c)
    return total


def budget(prompt):
    return min(len(prompt.split()), MAX_TOKENS)


def stream_tokens(text):
    yield from text.split()


async def call_llm(prompt, temperature=0.0):
    return prompt


# Opcode names differ between versions (LOAD_FAST vs LOAD_FAST_LOAD_FAST vs
# LOAD_FAST_BORROW...), so tests only rely on long-stable names.


@pytest.mark.topic("py.internals.dis", "py.internals.bytecode")
def test_opnames():
    ops = ex.opnames(add)
    assert isinstance(ops, set)
    assert {"BINARY_OP", "RETURN_VALUE"} <= ops
    assert any(op.startswith("LOAD_FAST") for op in ops)
    assert "FOR_ITER" not in ops
    assert {"FOR_ITER", "GET_ITER", "LOAD_GLOBAL", "CALL"} <= ex.opnames(total_len)


@pytest.mark.topic("py.internals.dis", "py.internals.cpython")
def test_global_names():
    assert ex.global_names(total_len) == {"len"}
    assert ex.global_names(budget) == {"min", "len", "MAX_TOKENS"}  # 'split' is an attribute
    assert ex.global_names(add) == set()


@pytest.mark.topic("py.internals.bytecode", "py.internals.cpython")
def test_code_info():
    assert ex.code_info(total_len) == {
        "argcount": 1,
        "locals": ("chunks", "total", "c"),
        "is_generator": False,
        "is_coroutine": False,
    }
    gen = ex.code_info(stream_tokens)
    assert gen["is_generator"] and not gen["is_coroutine"]
    coro = ex.code_info(call_llm)
    assert coro["is_coroutine"] and not coro["is_generator"]
    assert coro["argcount"] == 2


@pytest.mark.topic("py.internals.bytecode")
def test_constant_folding():
    consts = ex.folded_constants('SECONDS = 24 * 60 * 60\nSEP = "-" * 3\nT = (1, 2) + (3,)\n')
    assert isinstance(consts, tuple)
    assert 86400 in consts
    assert "---" in consts
    assert (1, 2, 3) in consts


@pytest.mark.topic("py.internals.dis")
def test_disassembly_is_text():
    text = ex.disassembly(total_len)
    assert isinstance(text, str)
    assert "LOAD_GLOBAL" in text and "len" in text
    assert "FOR_ITER" in text
