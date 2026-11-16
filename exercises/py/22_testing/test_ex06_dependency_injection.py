from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from study.loader import load

ex = load(__file__, "ex06_dependency_injection")

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


def rec(hours_ago: float, tokens: int) -> "ex.UsageRecord":
    return ex.UsageRecord(ts=NOW - timedelta(hours=hours_ago), tokens=tokens)


@pytest.mark.topic("py.testing.unit", "py.testing.clean_code")
def test_cost_in_window_pure():
    records = [rec(1, 500_000), rec(23, 500_000), rec(25, 9_000_000)]
    assert ex.cost_in_window(records, NOW) == pytest.approx(2.0)
    assert ex.cost_in_window([], NOW) == 0.0


@pytest.mark.topic("py.testing.unit")
@pytest.mark.parametrize(
    ("hours_ago", "counted"),
    [(0, True), (24, False), (23.999, True), (-0.001, False)],
    ids=["exactly-now", "exactly-window-start", "just-inside", "future"],
)
def test_cost_in_window_edges(hours_ago, counted):
    cost = ex.cost_in_window([rec(hours_ago, 1000)], NOW, price_per_1k=1.0)
    assert cost == (1.0 if counted else 0.0)


@pytest.mark.topic("py.testing.unit")
def test_cost_in_window_custom_window_and_price():
    records = [rec(0.5, 2000), rec(2, 2000)]
    assert ex.cost_in_window(records, NOW, window=timedelta(hours=1), price_per_1k=0.5) == 1.0


@pytest.mark.topic("py.testing.mocking", "py.testing.clean_code")
def test_fake_clock():
    clock = ex.FakeClock(NOW)
    assert clock() == NOW
    clock.advance(hours=2, minutes=30)
    assert clock() == NOW + timedelta(hours=2, minutes=30)


@pytest.mark.topic("py.testing.mocking")
def test_recording_notifier():
    notify = ex.RecordingNotifier()
    assert notify.messages == []
    notify("a")
    notify("b")
    assert notify.messages == ["a", "b"]


@pytest.mark.topic("py.testing.clean_code", "py.testing.mocking")
def test_alerter_notifies_over_budget_with_fakes():
    notify = ex.RecordingNotifier()
    alerter = ex.UsageAlerter(
        load_records=lambda: [rec(1, 1_000_000), rec(2, 500_000)],
        clock=ex.FakeClock(NOW),
        notify=notify,
        budget_usd=2.5,
    )
    assert alerter.run() == pytest.approx(3.0)
    assert notify.messages == ["LLM spend $3.00 over budget $2.50"]


@pytest.mark.topic("py.testing.clean_code")
def test_alerter_silent_at_or_under_budget():
    notify = ex.RecordingNotifier()
    alerter = ex.UsageAlerter(
        load_records=lambda: [rec(1, 1_000_000)],
        clock=ex.FakeClock(NOW),
        notify=notify,
        budget_usd=2.0,  # cost == budget exactly -> no alert
    )
    assert alerter.run() == pytest.approx(2.0)
    assert notify.messages == []


@pytest.mark.topic("py.testing.clean_code", "py.testing.unit")
def test_alerter_window_moves_with_injected_clock():
    clock = ex.FakeClock(NOW)
    notify = ex.RecordingNotifier()
    alerter = ex.UsageAlerter(
        load_records=lambda: [rec(1, 1_000_000)],
        clock=clock,
        notify=notify,
        budget_usd=1.0,
        price_per_1k=0.002,
    )
    assert alerter.run() == pytest.approx(2.0)
    clock.advance(hours=24)  # the record is now 25h old
    assert alerter.run() == 0.0
    assert len(notify.messages) == 1


@pytest.mark.topic("py.testing.mocking", "py.testing.unittest_mock")
def test_alerter_calls_dependencies_once_per_run():
    load_records = Mock(return_value=[])
    clock = Mock(return_value=NOW)
    notify = Mock()
    alerter = ex.UsageAlerter(load_records=load_records, clock=clock, notify=notify, budget_usd=1.0)
    alerter.run()
    alerter.run()
    assert load_records.call_count == 2
    assert clock.call_count == 2
    notify.assert_not_called()


@pytest.mark.topic("py.testing.clean_code")
def test_alerter_accepts_a_generator_of_records():
    # load_records may stream records lazily (e.g. reading a big JSONL file)
    def stream():
        yield rec(1, 1_000_000)
        yield rec(2, 1_000_000)

    notify = ex.RecordingNotifier()
    alerter = ex.UsageAlerter(load_records=stream, clock=lambda: NOW, notify=notify, budget_usd=10)
    assert alerter.run() == pytest.approx(4.0)
