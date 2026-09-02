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


# --------------------------------------------------------------------- yfinance --
def _yf_frame(symbols, n=6, start="2024-01-02"):
    """A yfinance-shaped download: MultiIndex (Ticker, Price) columns."""
    index = pd.date_range(start, periods=n, freq="B", name="Date")
    blocks = {}
    for i, symbol in enumerate(symbols):
        base = 100.0 * (i + 1)
        blocks[(symbol, "Open")] = base + np.arange(n) * 0.9
        blocks[(symbol, "High")] = base + np.arange(n) * 1.1
        blocks[(symbol, "Low")] = base + np.arange(n) * 0.7
        blocks[(symbol, "Close")] = base + np.arange(n)
        blocks[(symbol, "Volume")] = np.full(n, 1_000.0 * (i + 1))
    frame = pd.DataFrame(blocks, index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Ticker", "Price"])
    return frame


def test_yfinance_batch_maps_onto_the_table_schema(monkeypatch):
    from strategies151.data.loaders import YFinanceSource

    monkeypatch.setattr(
        YFinanceSource, "_download",
        staticmethod(lambda symbols, start, end: _yf_frame(symbols)),
    )
    out = YFinanceSource().fetch_many(["NVDA", "JPM"])
    assert set(out) == {"NVDA", "JPM"}
    assert list(out["NVDA"].columns) == OHLCV_COLUMNS
    assert out["NVDA"]["ticker"].unique().tolist() == ["NVDA"]
    assert out["NVDA"]["date"].dt.tz is None


def test_yfinance_translates_share_class_symbols(monkeypatch):
    """`BRK.B` is `BRK-B` upstream but must come back under the ticker asked for."""
    from strategies151.data.loaders import YFinanceSource

    seen = {}

    def fake(symbols, start, end):
        seen["symbols"] = list(symbols)
        return _yf_frame(symbols)

    monkeypatch.setattr(YFinanceSource, "_download", staticmethod(fake))
    out = YFinanceSource().fetch_many(["BRK.B"])
    assert seen["symbols"] == ["BRK-B"]
    assert list(out) == ["BRK.B"]


def test_yfinance_end_date_is_inclusive(monkeypatch):
    """yfinance excludes `end`; the other sources include it."""
    from strategies151.data.loaders import YFinanceSource

    captured = {}

    def fake(symbols, start, end):
        captured["end"] = end
        return _yf_frame(symbols)

    monkeypatch.setattr(YFinanceSource, "_download", staticmethod(fake))
    YFinanceSource().fetch_many(["NVDA"], start="2024-01-01", end="2024-01-31")
    assert captured["end"] == "2024-02-01"


def test_yfinance_skips_tickers_the_batch_did_not_return(monkeypatch):
    from strategies151.data.loaders import YFinanceSource

    monkeypatch.setattr(
        YFinanceSource, "_download",
        staticmethod(lambda symbols, start, end: _yf_frame(["NVDA"])),
    )
    out = YFinanceSource().fetch_many(["NVDA", "DELISTED"])
    assert set(out) == {"NVDA"}


def test_yfinance_single_fetch_raises_when_empty(monkeypatch):
    from strategies151.data import loaders

    monkeypatch.setattr(
        loaders.YFinanceSource, "_download",
        staticmethod(lambda symbols, start, end: pd.DataFrame()),
    )
    with pytest.raises(loaders.DataSourceError, match="no rows"):
        loaders.YFinanceSource().fetch("NOPE")


def test_batch_fetch_falls_back_when_the_batch_raises(monkeypatch):
    """A failed bulk request must degrade to the per-ticker path, not abort."""
    from strategies151.data import loaders

    monkeypatch.setattr(
        loaders.YFinanceSource, "fetch_many",
        lambda self, tickers, start=None, end=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert loaders._batch_fetch(["A", "B"], "yfinance", None, None, 100) == {}


def test_batch_fetch_is_disabled_for_sources_without_it():
    from strategies151.data import loaders

    assert loaders._batch_fetch(["A"], "yahoo", None, None, 100) == {}
    assert loaders._batch_fetch(["A"], "yfinance", None, None, 1) == {}


def test_auto_order_skips_unavailable_sources(monkeypatch):
    from strategies151.data import loaders

    loaders.reset_circuit_breakers()
    monkeypatch.setattr(loaders.YFinanceSource, "available", staticmethod(lambda: False))
    tried = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def fetch(self, ticker, start=None, end=None):
            tried.append(self.name)
            return StooqSource._parse(ticker, STOOQ_CSV)

    monkeypatch.setitem(loaders.SOURCES, "stooq", lambda: Recorder("stooq"))
    loaders.fetch_bars("X", source="auto")
    assert tried == ["stooq"]          # yfinance never consulted
    loaders.reset_circuit_breakers()


@pytest.mark.integration
def test_yfinance_and_the_direct_yahoo_client_agree():
    """Switching source must not change any result.

    Both paths return split/dividend adjusted bars, so the close-to-close return
    series - which is all the strategies consume - has to match.
    """
    from strategies151.data.loaders import YahooSource, YFinanceSource

    if not YFinanceSource.available():
        pytest.skip("yfinance not installed")
    try:
        a = YFinanceSource().fetch("MSFT", start="2020-01-01").set_index("date")
        b = YahooSource().fetch("MSFT", start="2020-01-01").set_index("date")
    except Exception as exc:  # noqa: BLE001 - network flakiness is not a failure
        pytest.skip(f"upstream unavailable: {exc}")

    shared = a.index.intersection(b.index)
    assert len(shared) > 500
    ra = a.loc[shared, "close"].pct_change().dropna()
    rb = b.loc[shared, "close"].pct_change().dropna()
    common = ra.index.intersection(rb.index)
    assert (ra.loc[common] - rb.loc[common]).abs().max() < 1e-4
