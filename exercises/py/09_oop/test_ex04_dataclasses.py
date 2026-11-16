import dataclasses

import pytest

from study.loader import load

ex = load(__file__, "ex04_dataclasses")


@pytest.mark.topic("py.oop.dataclasses", "py.oop.init")
def test_model_config_defaults_and_validation():
    cfg = ex.ModelConfig("gpt-x")
    assert (cfg.name, cfg.temperature, cfg.max_tokens, cfg.stop) == ("gpt-x", 0.7, 1024, ())
    for kwargs in ({"name": ""}, {"name": "m", "temperature": 2.1}, {"name": "m", "max_tokens": 0}):
        with pytest.raises(ValueError):
            ex.ModelConfig(**kwargs)


@pytest.mark.topic("py.oop.dataclasses")
def test_model_config_frozen_hashable_normalised():
    cfg = ex.ModelConfig("m", stop=["###", "END"])
    assert cfg.stop == ("###", "END")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.temperature = 0.0
    same = ex.ModelConfig("m", stop=("###", "END"))
    assert cfg == same
    assert {cfg: "cached"}[same] == "cached"


@pytest.mark.topic("py.oop.slots", "py.oop.dataclasses")
def test_model_config_uses_slots():
    cfg = ex.ModelConfig("m")
    assert not hasattr(cfg, "__dict__")
    assert "temperature" in ex.ModelConfig.__slots__


@pytest.mark.topic("py.oop.class_methods")
def test_from_dict():
    cfg = ex.ModelConfig.from_dict({"name": "m", "max_tokens": 10})
    assert cfg == ex.ModelConfig("m", max_tokens=10)
    with pytest.raises(TypeError, match="max_token, temprature"):
        ex.ModelConfig.from_dict({"name": "m", "temprature": 0.1, "max_token": 5})


@pytest.mark.topic("py.oop.class_methods")
def test_from_env():
    env = {"LLM_NAME": "gpt-x", "LLM_TEMPERATURE": "0.2", "LLM_MAX_TOKENS": "256", "LLM_STOP": "###,END"}
    assert ex.ModelConfig.from_env(env) == ex.ModelConfig("gpt-x", 0.2, 256, ("###", "END"))
    assert ex.ModelConfig.from_env({"APP_NAME": "m", "APP_STOP": ""}, prefix="APP_") == ex.ModelConfig("m")
    with pytest.raises(KeyError):
        ex.ModelConfig.from_env({"LLM_TEMPERATURE": "0.1"})
    with pytest.raises(ValueError):
        ex.ModelConfig.from_env({"LLM_NAME": "m", "LLM_TEMPERATURE": "9"})


@pytest.mark.topic("py.oop.dataclasses")
def test_with_overrides_returns_new_validated_copy():
    base = ex.ModelConfig("m")
    hot = base.with_overrides(temperature=1.5, stop=["\n"])
    assert hot == ex.ModelConfig("m", 1.5, 1024, ("\n",))
    assert base.temperature == 0.7
    with pytest.raises(ValueError):
        base.with_overrides(max_tokens=-1)


@pytest.mark.topic("py.oop.dataclasses")
def test_eval_result_ordering_and_repr():
    a = ex.EvalResult("a", 0.9, latency_ms=300)
    b = ex.EvalResult("b", 0.9, latency_ms=120)
    c = ex.EvalResult("c", 0.5)
    d = ex.EvalResult("d", 0.95, latency_ms=900)
    assert [r.name for r in sorted([a, b, c, d])] == ["d", "b", "a", "c"]
    assert b < a
    assert repr(c) == "EvalResult(name='c', score=0.5, latency_ms=0.0)"
    with pytest.raises(TypeError):
        ex.EvalResult("x", 1.0, 0.0, (0,))  # sort_key is not an __init__ parameter


@pytest.mark.topic("py.oop.dataclasses")
def test_eval_run_default_factory_and_methods():
    cfg = ex.ModelConfig("m")
    r1, r2 = ex.EvalRun(cfg), ex.EvalRun(cfg)
    r1.add(ex.EvalResult("q1", 0.5))
    r1.tags["suite"] = "rag"
    assert r2.results == [] and r2.tags == {}, "each run needs its own containers"
    r1.add(ex.EvalResult("q2", 1.0))
    r1.add(ex.EvalResult("q3", 0.0))
    assert [r.name for r in r1.leaderboard(2)] == ["q2", "q1"]
    assert len(r1.leaderboard()) == 3
    assert r1.mean_score == pytest.approx(0.5)
    with pytest.raises(ValueError):
        _ = r2.mean_score
