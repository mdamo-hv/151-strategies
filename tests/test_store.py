"""The two bar stores must be interchangeable."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from strategies151.config import Config
from strategies151.data.store import (
    OHLCV_COLUMNS,
    DuckDBStore,
    StoreError,
    open_store,
)


@pytest.fixture
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=120)
    frames = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 50 * (i + 1) * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(dates))))
        frames.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "open": close * 0.99, "high": close * 1.02,
            "low": close * 0.98, "close": close, "volume": rng.lognormal(15, 0.3, len(dates)),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def store(tmp_path) -> DuckDBStore:
    return DuckDBStore(tmp_path / "bars.duckdb", table="stooq.daily")


# ------------------------------------------------------------------ DuckDB --
def test_creates_the_file_and_table_on_demand(store, bars):
    assert store.insert_bars(bars) == len(bars)
    assert store.path.exists()
    assert store.ping()


def test_round_trips_every_column(store, bars):
    store.insert_bars(bars)
    out = store.read_bars(["AAA", "BBB"])
    assert list(out.columns) == OHLCV_COLUMNS
    assert len(out) == len(bars)
    merged = bars.merge(out, on=["ticker", "date"], suffixes=("_in", "_out"))
    for column in ("open", "high", "low", "close", "volume"):
        assert (merged[f"{column}_in"] - merged[f"{column}_out"]).abs().max() < 1e-9


def test_reload_is_idempotent(store, bars):
    store.insert_bars(bars)
    store.insert_bars(bars)
    assert len(store.read_bars(["AAA", "BBB"])) == len(bars)


def test_reload_overwrites_revised_bars(store, bars):
    store.insert_bars(bars)
    revised = bars.copy()
    revised["close"] = revised["close"] * 2
    store.insert_bars(revised)
    out = store.read_bars(["AAA"]).sort_values("date")
    expected = revised[revised.ticker == "AAA"].sort_values("date")
    assert out["close"].iloc[0] == pytest.approx(expected["close"].iloc[0])


def test_duplicate_keys_within_one_batch_take_the_last(store, bars):
    """DuckDB refuses to update the same row twice in one statement."""
    doubled = pd.concat([bars, bars.assign(close=bars["close"] + 1)], ignore_index=True)
    store.insert_bars(doubled)
    assert len(store.read_bars(["AAA", "BBB"])) == len(bars)


def test_date_filters_are_applied(store, bars):
    store.insert_bars(bars)
    cutoff = bars["date"].iloc[50]
    assert store.read_bars(["AAA"], start=str(cutoff.date()))["date"].min() == cutoff
    assert store.read_bars(["AAA"], end=str(cutoff.date()))["date"].max() == cutoff


def test_coverage_reports_ranges(store, bars):
    store.insert_bars(bars)
    coverage = store.coverage().set_index("ticker")
    assert list(coverage.index) == ["AAA", "BBB"]
    assert coverage.loc["AAA", "bars"] == (bars.ticker == "AAA").sum()


def test_reading_an_empty_store_returns_the_schema(store):
    out = store.read_bars(["AAA"])
    assert out.empty and list(out.columns) == OHLCV_COLUMNS


def test_insert_rejects_missing_columns(store):
    with pytest.raises(ValueError, match="missing columns"):
        store.insert_bars(pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2024-01-01")]}))


def test_catalog_name_clash_is_handled(tmp_path, bars):
    """A file named stooq.duckdb makes catalog and schema collide in DuckDB."""
    store = DuckDBStore(tmp_path / "stooq.duckdb", table="stooq.daily")
    store.insert_bars(bars)
    assert len(store.read_bars(["AAA"])) == (bars.ticker == "AAA").sum()


def test_verify_schema_rejects_a_wrongly_typed_table(tmp_path, bars):
    store = DuckDBStore(tmp_path / "bars.duckdb", table="stooq.daily")
    con = store.connect()
    con.execute('CREATE SCHEMA IF NOT EXISTS "stooq"')
    con.execute(f'CREATE TABLE {store.qualified} (ticker VARCHAR, date VARCHAR, close DOUBLE)')
    with pytest.raises(StoreError, match="not TIMESTAMP"):
        store.verify_schema()


def test_build_version_names_the_backend(store):
    assert "DuckDB" in store.build_version()


# ----------------------------------------------------------------- factory --
def test_factory_selects_the_backend():
    cfg = Config.load()
    assert open_store(cfg, "duckdb").description.startswith("duckdb")
    assert open_store(cfg, "questdb").description.startswith("questdb")


def test_factory_honours_the_config_default():
    cfg = Config.load()
    cfg = replace(cfg, storage=replace(cfg.storage, backend="duckdb"))
    assert open_store(cfg).description.startswith("duckdb")


def test_factory_rejects_an_unknown_backend():
    with pytest.raises(StoreError, match="unknown storage backend"):
        open_store(Config.load(), "sqlite")


# ------------------------------------------------------------------ parity --
@pytest.mark.integration
def test_both_backends_return_identical_frames(tmp_path, bars):
    """Same rows in, same frame out - so `--store` cannot change a result."""
    from strategies151.data.questdb import QuestDBClient

    cfg = Config.load()
    questdb = QuestDBClient(replace(cfg.questdb, table="stooq.parity_pytest"))
    if not questdb.ping():
        pytest.skip("no QuestDB reachable")
    duck = DuckDBStore(tmp_path / "parity.duckdb", table="stooq.parity_pytest")
    try:
        for store in (questdb, duck):
            store.drop_table()
            store.create_table()
            store.insert_bars(bars)
        left = questdb.read_bars(["AAA", "BBB"]).reset_index(drop=True)
        right = duck.read_bars(["AAA", "BBB"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, atol=1e-9)
    finally:
        questdb.drop_table()


@pytest.mark.integration
def test_a_panel_built_from_either_backend_matches(tmp_path, bars):
    from strategies151.data.panel import Panel

    duck = DuckDBStore(tmp_path / "panel.duckdb", table="stooq.daily")
    duck.insert_bars(bars)
    panel = Panel.from_long(duck.read_bars(["AAA", "BBB"]), ["AAA", "BBB"])
    assert panel.tickers == ["AAA", "BBB"]
    assert len(panel) == bars["date"].nunique()
    assert np.isfinite(panel.returns.iloc[1:].to_numpy()).all()
