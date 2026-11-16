import json
import os

import pytest

from study.loader import load

ex = load(__file__, "ex05_exitstack_resources")

HAS_PROC = os.path.isdir("/proc/self/fd")


def open_fds():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.topic("py.context_managers.exitstack", "py.context_managers.resources")
def test_open_shards_opens_and_closes_all(tmp_path):
    with ex.open_shards(tmp_path, 3) as files:
        assert len(files) == 3
        assert all(not f.closed for f in files)
        files[1].write("hello\n")
    assert all(f.closed for f in files)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "shard-00000.jsonl",
        "shard-00001.jsonl",
        "shard-00002.jsonl",
    ]
    assert (tmp_path / "shard-00001.jsonl").read_text(encoding="utf-8") == "hello\n"


@pytest.mark.topic("py.context_managers.cleanup")
def test_open_shards_closes_on_exception_in_block(tmp_path):
    with pytest.raises(UnicodeError), ex.open_shards(tmp_path, 2, prefix="out") as files:
        raise UnicodeError("tokenizer crashed")
    assert all(f.closed for f in files)
    assert (tmp_path / "out-00001.jsonl").exists()


@pytest.mark.topic("py.context_managers.exitstack", "py.context_managers.cleanup")
@pytest.mark.skipif(not HAS_PROC, reason="needs /proc/self/fd")
def test_open_shards_partial_failure_closes_already_opened(tmp_path):
    (tmp_path / "shard-00002.jsonl").mkdir()  # opening this one for writing fails
    before = open_fds()
    with pytest.raises(IsADirectoryError), ex.open_shards(tmp_path, 5):
        pass  # pragma: no cover
    assert open_fds() == before, "files 0 and 1 leaked"


@pytest.mark.topic("py.context_managers.exitstack")
def test_open_shards_validation(tmp_path):
    with pytest.raises(ValueError), ex.open_shards(tmp_path, 0):
        pass  # pragma: no cover


@pytest.mark.topic("py.context_managers.custom", "py.context_managers.exitstack")
def test_sharded_writer_round_robin(tmp_path):
    with ex.ShardedWriter(tmp_path, 3) as w:
        for i in range(7):
            w.write({"id": i, "text": f"chunk {i} ✓"})
        assert w.counts == [3, 2, 2]
    shard0 = (tmp_path / "shard-00000.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in shard0] == [0, 3, 6]
    shard2 = (tmp_path / "shard-00002.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in shard2] == [2, 5]


@pytest.mark.topic("py.context_managers.cleanup", "py.context_managers.resources")
@pytest.mark.skipif(not HAS_PROC, reason="needs /proc/self/fd")
def test_sharded_writer_closes_and_propagates(tmp_path):
    before = open_fds()
    with pytest.raises(KeyError), ex.ShardedWriter(tmp_path, 4) as w:
        w.write({"id": 0})
        assert open_fds() == before + 4
        raise KeyError("bad record")
    assert open_fds() == before
    assert (tmp_path / "shard-00000.jsonl").read_text(encoding="utf-8") == '{"id": 0}\n'


@pytest.mark.topic("py.context_managers.cleanup")
@pytest.mark.skipif(not HAS_PROC, reason="needs /proc/self/fd")
def test_sharded_writer_enter_failure_leaks_nothing(tmp_path):
    (tmp_path / "shard-00003.jsonl").mkdir()
    before = open_fds()
    with pytest.raises(IsADirectoryError), ex.ShardedWriter(tmp_path, 4):
        pass  # pragma: no cover
    assert open_fds() == before


class FakeClient:
    def __init__(self, name, log):
        self.name, self.log, self.closed = name, log, False

    def close(self):
        self.closed = True
        self.log.append(f"close {self.name}")


def factory(name, log, fail=False):
    def make():
        if fail:
            raise ConnectionError(f"{name} unreachable")
        log.append(f"open {name}")
        return FakeClient(name, log)

    return make


@pytest.mark.topic("py.context_managers.exitstack", "py.context_managers.resources")
def test_acquire_all_success_hands_over_open_clients():
    log = []
    clients = ex.acquire_all([factory(n, log) for n in ("embedder", "llm", "vectordb")])
    assert [c.name for c in clients] == ["embedder", "llm", "vectordb"]
    assert not any(c.closed for c in clients), "caller owns them now - use pop_all()"
    assert log == ["open embedder", "open llm", "open vectordb"]


@pytest.mark.topic("py.context_managers.exitstack", "py.context_managers.cleanup")
def test_acquire_all_partial_failure_closes_in_reverse():
    log = []
    factories = [factory("a", log), factory("b", log), factory("c", log, fail=True), factory("d", log)]
    with pytest.raises(ConnectionError, match="c unreachable"):
        ex.acquire_all(factories)
    assert log == ["open a", "open b", "close b", "close a"]
    assert ex.acquire_all([]) == []
