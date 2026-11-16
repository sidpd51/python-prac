import pytest

from study.loader import load

ex = load(__file__, "ex03_mock_llm")


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
    check_empty=True,
    prompt_prefix="Summarize in one sentence:\n",
    pass_temperature=True,
    strip=True,
    extra_attempts=0,
    single_attempt=False,
    retry_on=None,
    swallow=False,
    hardcode_model=False,
):
    def impl(client, text, *, model="fast-1", retries=2):
        if check_empty and not text.strip():
            raise ValueError("empty text")
        prompt = prompt_prefix + text
        use_model = "fast-1" if hardcode_model else model
        catch = retry_on or ex.TransientError
        attempts = 1 if single_attempt else retries + 1 + extra_attempts
        for attempt in range(attempts):
            try:
                if pass_temperature:
                    reply = client.complete(prompt, model=use_model, temperature=0.0)
                else:
                    reply = client.complete(prompt, model=use_model)
                return reply.strip() if strip else reply
            except catch:
                if attempt == attempts - 1:
                    if swallow:
                        return ""
                    raise
        raise AssertionError("unreachable")

    return impl


MUTANTS = {
    "calls_client_on_empty_text": _make(check_empty=False),
    "wrong_prompt": _make(prompt_prefix="Summarize:\n"),
    "default_temperature": _make(pass_temperature=False),
    "no_strip": _make(strip=False),
    "no_retry": _make(single_attempt=True),
    "one_retry_too_many": _make(extra_attempts=1),
    "retries_every_exception": _make(retry_on=Exception),
    "swallows_final_error": _make(swallow=True),
    "ignores_model_argument": _make(hardcode_model=True),
}


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything reaches the real client."""
    calls = []
    real = ex.LLMClient.complete

    def guard(self, *args, **kwargs):
        calls.append(args)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(ex.LLMClient, "complete", guard)
    return calls


@pytest.mark.topic("py.testing.mocking", "py.testing.unittest_mock")
def test_suite_passes_on_correct_impl(no_network):
    ex.check_summarize(ex.summarize)
    assert no_network == [], "your suite called the real LLMClient.complete"


@pytest.mark.topic("py.testing.mocking", "py.testing.unittest_mock", "py.testing.unit")
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_suite_catches_mutant(name, no_network):
    assert_suite_fails(ex.check_summarize, MUTANTS[name])


@pytest.mark.topic("py.testing.unittest_mock", "py.testing.mocking")
def test_patched_llm_replaces_client_where_looked_up(no_network):
    with ex.patched_llm("  Paris \n") as llm:
        assert ex.ask("Capital of France?") == "Paris"
        llm.complete.assert_called_once_with("Q: Capital of France?\nA:", model="smart-1")
    assert no_network == []


@pytest.mark.topic("py.testing.unittest_mock")
def test_patched_llm_restores_real_class():
    real = ex.LLMClient
    with ex.patched_llm("x"):
        assert ex.LLMClient is not real
    assert ex.LLMClient is real


@pytest.mark.topic("py.testing.unittest_mock")
def test_patched_llm_is_autospecced():
    with ex.patched_llm("x") as llm:
        with pytest.raises(AttributeError):
            llm.complet  # noqa: B018 - typo'd attribute must not exist on an autospec mock
        with pytest.raises(TypeError):
            llm.complete("prompt", modle="typo")  # wrong signature must fail
