"""QuestDB round-trip. Skipped automatically when no server is reachable."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from strategies151.config import Config
from strategies151.data.panel import Panel
from strategies151.data.questdb import QuestDBClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    cfg = Config.load().questdb
    cfg = replace(cfg, table="stooq.daily_pytest")
    client = QuestDBClient(cfg)
    if not client.ping():
        pytest.skip(f"no QuestDB at {cfg.http_url}")
    client.drop_table()
    client.create_table()
    yield client
    client.drop_table()


def test_dotted_table_names_round_trip(client):
    bars = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
            "open": [10.0, 11.0, 20.0],
            "high": [10.5, 11.5, 20.5],
            "low": [9.5, 10.5, 19.5],
            "close": [10.2, 11.2, 20.2],
            "volume": [1000.0, 1100.0, 2000.0],
        }
    )
    assert client.insert_bars(bars) == 3
    out = client.read_bars(["AAA", "BBB"])
    assert len(out) == 3
    assert set(out["ticker"]) == {"AAA", "BBB"}
    assert out["close"].max() == pytest.approx(20.2)


def test_reload_is_idempotent(client):
    bars = pd.DataFrame(
        {
            "ticker": ["CCC"],
            "date": pd.to_datetime(["2024-02-01"]),
            "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0],
        }
    )
    client.insert_bars(bars)
    client.insert_bars(bars)
    assert len(client.read_bars(["CCC"])) == 1


def test_date_filters_are_applied(client):
    out = client.read_bars(["AAA"], start="2024-01-03")
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-03")


def test_read_bars_builds_a_usable_panel(client):
    frame = client.read_bars(["AAA", "BBB"])
    built = Panel.from_long(frame, ["AAA", "BBB"])
    # 2024-01-03 has no BBB bar, so the rectangular panel keeps only 2024-01-02.
    assert len(built) == 1


def test_insert_rejects_missing_columns(client):
    with pytest.raises(ValueError, match="missing columns"):
        client.insert_bars(pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2024-01-01")]}))


def test_coverage_reports_ranges(client):
    coverage = client.coverage()
    assert set(coverage.columns) == {"ticker", "bars", "first_bar", "last_bar"}
    assert len(coverage) >= 2
