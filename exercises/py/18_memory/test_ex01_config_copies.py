import copy

import pytest

from study.loader import load

ex = load(__file__, "ex01_config_copies")

BASE = {
    "model": "text-embedding-3-small",
    "training": {"lr": 3e-4, "seed": 0, "schedule": {"warmup": 100}},
    "tags": ["baseline"],
}


@pytest.mark.topic("py.memory.references", "py.memory.object_model")
def test_identity_groups_is_not_equality():
    a, b = [1], [1]
    assert a == b
    assert ex.identity_groups([a, b, a]) == [[0, 2], [1]]


@pytest.mark.topic("py.memory.references")
def test_identity_groups_unhashable_and_order():
    d = {"k": 1}
    s = "shared"
    items = [d, s, {"k": 1}, d, s, d]
    assert ex.identity_groups(items) == [[0, 3, 5], [1, 4], [2]]
    assert ex.identity_groups([]) == []


@pytest.mark.topic("py.memory.copy")
def test_run_configs_are_independent():
    base = copy.deepcopy(BASE)
    snapshot = copy.deepcopy(base)
    configs = ex.make_run_configs(base, [1, 2, 3])
    assert [c["training"]["seed"] for c in configs] == [1, 2, 3]
    assert base == snapshot, "base config was mutated"

    configs[0]["training"]["schedule"]["warmup"] = 999
    configs[0]["tags"].append("mutated")
    assert configs[1]["training"]["schedule"]["warmup"] == 100
    assert configs[1]["tags"] == ["baseline"]
    assert base == snapshot


@pytest.mark.topic("py.memory.copy")
def test_merge_config_recursive():
    override = {"training": {"lr": 1e-3, "schedule": {"decay": "cosine"}}, "tags": ["exp7"]}
    merged = ex.merge_config(BASE, override)
    assert merged == {
        "model": "text-embedding-3-small",
        "training": {"lr": 1e-3, "seed": 0, "schedule": {"warmup": 100, "decay": "cosine"}},
        "tags": ["exp7"],
    }


@pytest.mark.topic("py.memory.copy", "py.memory.references")
def test_merge_config_shares_nothing_mutable():
    base = copy.deepcopy(BASE)
    override = {"training": {"schedule": {"decay": "cosine"}}, "extra": {"notes": ["a"]}}
    base_snap, over_snap = copy.deepcopy(base), copy.deepcopy(override)

    merged = ex.merge_config(base, override)
    merged["training"]["schedule"]["warmup"] = -1
    merged["tags"].append("x")
    merged["extra"]["notes"].append("b")

    assert base == base_snap
    assert override == over_snap
    assert merged["tags"] is not base["tags"]
    assert merged["extra"] is not override["extra"]


@pytest.mark.topic("py.memory.copy")
def test_merge_config_non_dict_overrides_dict():
    assert ex.merge_config({"a": {"b": 1}}, {"a": 5}) == {"a": 5}
    assert ex.merge_config({"a": 5}, {"a": {"b": 1}}) == {"a": {"b": 1}}
    assert ex.merge_config({}, {}) == {}
