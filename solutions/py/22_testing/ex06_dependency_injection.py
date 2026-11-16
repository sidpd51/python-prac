"""
Exercise 22.6 - Refactor for testability: dependency injection + fakes   level: ★★★
topics: testing.clean_code, testing.mocking, testing.unit

Problem
-------
This function is impossible to unit-test: it reads the wall clock, a hard-coded file,
an env var and posts to Slack - all inline.

    def send_usage_alert():
        now = datetime.now(timezone.utc)
        records = [json.loads(line) for line in open("/var/log/llm_usage.jsonl")]
        recent = [r for r in records
                  if now - datetime.fromisoformat(r["ts"]) < timedelta(hours=24)]
        cost = sum(r["tokens"] for r in recent) / 1000 * 0.002
        if cost > float(os.environ["BUDGET_USD"]):
            requests.post(SLACK_URL, json={"text": f"LLM spend ${cost:.2f} over budget"})

Refactor it into a pure core + an imperative shell with injected dependencies:

1. cost_in_window(records, now, *, window=timedelta(hours=24), price_per_1k=0.002) -> float
   PURE function. records: iterable of UsageRecord (given). A record counts when
   now - window < record.ts <= now (records from the future are ignored too).
   cost = total_tokens / 1000 * price_per_1k

2. class UsageAlerter
       __init__(self, *, load_records, clock, notify, budget_usd, window=timedelta(hours=24),
                price_per_1k=0.002)
         load_records: () -> Iterable[UsageRecord]     clock: () -> datetime
         notify: (str) -> None
       run(self) -> float
         computes the cost for the window ending at clock(), calls
         notify(f"LLM spend ${cost:.2f} over budget ${budget:.2f}") only when
         cost > budget_usd (strictly), and returns the cost either way.
         Calls load_records and clock exactly once per run.

3. class FakeClock - a fake (working, simplified implementation), not a mock:
       FakeClock(start: datetime); calling the instance returns the current time;
       advance(**kwargs) moves it forward by timedelta(**kwargs).

4. class RecordingNotifier - a fake notifier: calling it appends the message to
   `self.messages` (a list, starts empty).

Interviewers look for
---------------------
* "Functional core, imperative shell": business logic in pure functions = trivial tests,
  no mocks. I/O at the edges, injected.
* Inject time. datetime.now() inside logic makes "24h window" tests flaky at midnight/DST.
* Fakes vs mocks: a fake has real (simplified) behaviour and you assert on STATE
  (notifier.messages); a mock records calls and you assert on INTERACTIONS.
  Prefer fakes for your own boundaries - they don't break when you refactor internals.
* Strict vs non-strict comparisons at the window edges are where the bugs hide.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class UsageRecord:
    ts: datetime
    tokens: int
    model: str = "fast-1"


def cost_in_window(
    records: Iterable[UsageRecord],
    now: datetime,
    *,
    window: timedelta = timedelta(hours=24),
    price_per_1k: float = 0.002,
) -> float:
    start = now - window
    tokens = sum(r.tokens for r in records if start < r.ts <= now)
    return tokens / 1000 * price_per_1k


class UsageAlerter:
    def __init__(
        self,
        *,
        load_records: Callable[[], Iterable[UsageRecord]],
        clock: Callable[[], datetime],
        notify: Callable[[str], None],
        budget_usd: float,
        window: timedelta = timedelta(hours=24),
        price_per_1k: float = 0.002,
    ) -> None:
        self._load_records = load_records
        self._clock = clock
        self._notify = notify
        self._budget = budget_usd
        self._window = window
        self._price = price_per_1k

    def run(self) -> float:
        cost = cost_in_window(
            self._load_records(), self._clock(), window=self._window, price_per_1k=self._price
        )
        if cost > self._budget:
            self._notify(f"LLM spend ${cost:.2f} over budget ${self._budget:.2f}")
        return cost


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)
