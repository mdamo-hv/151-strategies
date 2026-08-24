"""Loader parsing, exercised offline against canned payloads."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies151.data.loaders import StooqSource, YahooSource
from strategies151.data.questdb import OHLCV_COLUMNS

STOOQ_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,10.0,10.8,9.9,10.5,1000\n"
    "2024-01-03,10.5,11.0,10.2,10.9,1200\n"
)


def test_stooq_csv_maps_onto_the_table_schema():
    frame = StooqSource._parse("nvda", STOOQ_CSV)
    assert list(frame.columns) == OHLCV_COLUMNS
    assert frame["ticker"].unique().tolist() == ["NVDA"]
    assert frame["date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert frame["close"].iloc[1] == pytest.approx(10.9)


def _yahoo_payload(closes, adjcloses):
    stamps = [
        int(pd.Timestamp(f"2024-01-0{i + 2} 14:30", tz="UTC").timestamp())
        for i in range(len(closes))
    ]
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "timestamp": stamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [c * 0.99 for c in closes],
                                    "high": [c * 1.02 for c in closes],
                                    "low": [c * 0.98 for c in closes],
                                    "close": closes,
                                    "volume": [1000.0] * len(closes),
                                }
                            ],
                            "adjclose": [{"adjclose": adjcloses}],
                        },
                    }
                ]
            }
        }
    )


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.status_code = 200


def test_yahoo_bars_are_rescaled_onto_the_adjusted_close(monkeypatch):
    closes = [100.0, 102.0]
    adj = [50.0, 51.0]  # a 2:1 split halves the adjusted series
    source = YahooSource()
    monkeypatch.setattr(source, "_get", lambda url, params: _FakeResponse(_yahoo_payload(closes, adj)))

    frame = source.fetch("NVDA")
    assert list(frame.columns) == OHLCV_COLUMNS
    assert frame["close"].tolist() == pytest.approx(adj)
    # Intraday ratios must survive the rescaling, so IBS and pivots stay valid.
    ratio = (frame["high"] - frame["low"]) / frame["close"]
    assert ratio.iloc[0] == pytest.approx((1.02 - 0.98) / 1.0, rel=1e-6)


def test_yahoo_returns_are_total_return_consistent(monkeypatch):
    closes = [100.0, 102.0]
    adj = [50.0, 51.0]
    source = YahooSource()
    monkeypatch.setattr(source, "_get", lambda url, params: _FakeResponse(_yahoo_payload(closes, adj)))
    frame = source.fetch("NVDA")
    assert frame["close"].pct_change().iloc[1] == pytest.approx(0.02)


def test_yahoo_dates_are_normalised_to_trading_days(monkeypatch):
    source = YahooSource()
    monkeypatch.setattr(
        source, "_get", lambda url, params: _FakeResponse(_yahoo_payload([1.0, 2.0], [1.0, 2.0]))
    )
    frame = source.fetch("NVDA")
    assert (frame["date"] == frame["date"].dt.normalize()).all()
    assert frame["date"].dt.tz is None


def test_circuit_breaker_stops_retrying_a_dead_source(monkeypatch):
    """A source that keeps failing must be dropped, not retried 500 times.

    Loading a 500-name universe against a host that is blocking us otherwise
    pays a full timeout per ticker - the difference between a ten-minute load
    and a four-hour one.
    """
    from strategies151.data import loaders

    loaders.reset_circuit_breakers()
    calls = {"stooq": 0, "yahoo": 0}

    class DeadStooq:
        def fetch(self, ticker, start=None, end=None):
            calls["stooq"] += 1
            raise loaders.DataSourceError("access denied")

    class LiveYahoo:
        def fetch(self, ticker, start=None, end=None):
            calls["yahoo"] += 1
            return StooqSource._parse(ticker, STOOQ_CSV)

    monkeypatch.setitem(loaders.SOURCES, "stooq", DeadStooq)
    monkeypatch.setitem(loaders.SOURCES, "yahoo", LiveYahoo)

    for i in range(20):
        loaders.fetch_bars(f"T{i}", source="auto")

    assert calls["stooq"] == loaders.CIRCUIT_BREAKER_THRESHOLD
    assert calls["yahoo"] == 20
    loaders.reset_circuit_breakers()


def test_circuit_breaker_resets_after_a_success(monkeypatch):
    from strategies151.data import loaders

    loaders.reset_circuit_breakers()
    state = {"fail": True}

    class Flaky:
        def fetch(self, ticker, start=None, end=None):
            if state["fail"]:
                raise loaders.DataSourceError("temporary")
            return StooqSource._parse(ticker, STOOQ_CSV)

    monkeypatch.setitem(loaders.SOURCES, "stooq", Flaky)
    for _ in range(3):
        with pytest.raises(loaders.DataSourceError):
            loaders.fetch_bars("X", source="stooq")
    state["fail"] = False
    loaders.fetch_bars("X", source="stooq")
    assert loaders._consecutive_failures["stooq"] == 0
    loaders.reset_circuit_breakers()
