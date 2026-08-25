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
    # A scratch table this project owns: writable, and named with whatever
    # column vocabulary the config asks for.
    cfg = replace(cfg, table="stooq.daily_pytest", read_only=False)
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


def test_insert_negotiates_a_timestamp_format(client):
    """The client must not depend on one QuestDB dialect.

    `SSSUUU` is a recent pattern token; an older build fails to parse it, does
    not recognise the column as a timestamp, and rejects every batch with
    `not a timestamp 'date'`. The plain date is tried first and remembered.
    """
    bars = pd.DataFrame(
        {"ticker": ["FMT"], "date": pd.to_datetime(["2024-03-01"]),
         "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]}
    )
    client._timestamp_format = None
    client.insert_bars(bars)
    assert client._timestamp_format == ("%Y-%m-%d", "yyyy-MM-dd")


def test_negotiated_format_round_trips_the_instant(client):
    dates = pd.to_datetime(["2012-01-03", "2026-08-21"])
    bars = pd.DataFrame(
        {"ticker": ["RT", "RT"], "date": dates, "open": [1.0, 2.0], "high": [2.0, 3.0],
         "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10.0, 20.0]}
    )
    client.insert_bars(bars)
    assert list(client.read_bars(["RT"])["date"]) == list(dates)


def test_verify_schema_rejects_a_wrongly_typed_table(client):
    from strategies151.data.questdb import QuestDBError

    table = "stooq.wrongtype_pytest"
    ticker_col, date_col = client.cfg.ticker_column, client.cfg.date_column
    client.exec(f"DROP TABLE IF EXISTS '{table}'")
    client.exec(
        f"CREATE TABLE '{table}' "
        f"({ticker_col} SYMBOL, {date_col} STRING, close DOUBLE)"
    )
    from dataclasses import replace as dc_replace

    other = QuestDBClient(dc_replace(client.cfg, table=table))
    try:
        with pytest.raises(QuestDBError, match="not TIMESTAMP"):
            other.verify_schema()
    finally:
        client.exec(f"DROP TABLE IF EXISTS '{table}'")


def test_verify_schema_accepts_the_real_table(client):
    client.verify_schema()


def test_build_version_is_readable(client):
    assert "QuestDB" in client.build_version()


def test_verify_schema_points_at_the_right_column_names(client):
    """A foreign table is a config problem, not a reason to drop it."""
    from strategies151.data.questdb import QuestDBError

    table = "stooq.foreign_pytest"
    client.exec(f"DROP TABLE IF EXISTS '{table}'")
    client.exec(
        f"CREATE TABLE '{table}' (symbol SYMBOL, ts TIMESTAMP, open DOUBLE, "
        "high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE) TIMESTAMP(ts)"
    )
    other = QuestDBClient(replace(client.cfg, table=table, date_column="nope"))
    try:
        with pytest.raises(QuestDBError, match="ticker_column: symbol, date_column: ts"):
            other.verify_schema()
    finally:
        client.exec(f"DROP TABLE IF EXISTS '{table}'")


def test_read_only_tables_are_never_written_or_dropped(client):
    """The shared bar table must survive a stray `s151 load`."""
    from strategies151.data.questdb import QuestDBError

    guarded = QuestDBClient(replace(client.cfg, read_only=True))
    bars = pd.DataFrame(
        {
            "ticker": ["AAA"], "date": pd.to_datetime(["2024-01-02"]),
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1.0],
        }
    )
    with pytest.raises(QuestDBError, match="read-only"):
        guarded.insert_bars(bars)
    with pytest.raises(QuestDBError, match="read-only"):
        guarded.drop_table()
    guarded.create_table()  # a no-op, not an error: the table already exists
    assert client.read_bars(["AAA"]) is not None
