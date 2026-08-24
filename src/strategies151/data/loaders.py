"""Fetch daily bars and land them in the QuestDB ``stooq.daily`` table.

Stooq is the source of record for this project.  ``stooq.com``'s CSV endpoint
sits behind a proof-of-work JS challenge and then blocks most datacenter IP
ranges outright, so :class:`YahooSource` provides a drop-in fallback that emits
the exact same split/dividend-adjusted OHLCV schema.  ``source: auto`` tries
Stooq first and falls back only when Stooq refuses to serve.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

from strategies151.data.questdb import OHLCV_COLUMNS, QuestDBClient
from strategies151.data.universe import to_stooq_symbol, to_yahoo_symbol

log = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_YAHOO_UAS = (
    "Mozilla/5.0",
    "python-requests/2.31",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    _UA,
)


class DataSourceError(RuntimeError):
    pass


@dataclass
class Bars:
    ticker: str
    frame: pd.DataFrame
    source: str


class StooqSource:
    """stooq.com daily CSV (``/q/d/l/?s=<sym>.us&i=d``).

    The endpoint answers the first request with a SHA-256 proof-of-work
    challenge; solving it and POSTing the nonce to ``/__verify`` sets the
    session cookie needed for the CSV.
    """

    name = "stooq"
    base = "https://stooq.com"

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _UA, "Referer": f"{self.base}/"})
        # urllib3 retries a failed connection three times underneath the
        # per-request timeout, so a host that black-holes us costs about four
        # times the timeout rather than the timeout. Disable it and let the
        # circuit breaker in `fetch_bars` decide whether to keep trying.
        adapter = requests.adapters.HTTPAdapter(max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._verified = False

    def _solve_challenge(self, body: str) -> bool:
        match_c = re.search(r'const c="([^"]+)"', body)
        match_d = re.search(r"d=(\d+)", body)
        if not (match_c and match_d):
            return False
        challenge, difficulty = match_c.group(1), int(match_d.group(1))
        target = "0" * difficulty
        nonce = 0
        while True:
            digest = hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest()
            if digest.startswith(target):
                break
            nonce += 1
        self.session.post(
            f"{self.base}/__verify",
            data=urllib.parse.urlencode({"c": challenge, "n": nonce}),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        return True

    def fetch(self, ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        params = {"s": f"{to_stooq_symbol(ticker)}.us", "i": "d"}
        if start:
            params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
        if end:
            params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
        url = f"{self.base}/q/d/l/"
        for attempt in range(2):
            resp = self.session.get(url, params=params, timeout=self.timeout)
            text = resp.text
            if "crypto.subtle" in text and not self._verified:
                self._verified = self._solve_challenge(text)
                continue
            if text.lstrip().lower().startswith("date,"):
                return self._parse(ticker, text)
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise DataSourceError(
                f"stooq refused {ticker}: {text.strip()[:120] or resp.status_code}"
            )
        raise DataSourceError(f"stooq refused {ticker} after retries")

    @staticmethod
    def _parse(ticker: str, csv_text: str) -> pd.DataFrame:
        frame = pd.read_csv(io.StringIO(csv_text))
        frame.columns = [c.strip().lower() for c in frame.columns]
        frame = frame.rename(columns={"date": "date"})
        frame["ticker"] = ticker.upper()
        frame["date"] = pd.to_datetime(frame["date"])
        if "volume" not in frame:
            frame["volume"] = float("nan")
        return frame.loc[:, OHLCV_COLUMNS]


class YahooSource:
    """Yahoo Finance chart API, normalised to the stooq schema.

    Yahoo returns raw OHLC plus an ``adjclose`` series.  Stooq publishes fully
    adjusted bars, so we scale the raw OHLC by ``adjclose / close`` to match:
    that keeps intraday ratios (high/low/close, IBS, pivot points) intact while
    making close-to-close returns total-return consistent.
    """

    name = "yahoo"
    base = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, timeout: float = 30.0, retries: int = 4):
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        # Yahoo's edge answers 429 to requests' default header set; a minimal,
        # curl-like header block gets served normally.
        self.session.headers.clear()
        self.session.headers.update({"User-Agent": _YAHOO_UAS[0], "Accept": "*/*"})

    def _get(self, url: str, params: dict) -> requests.Response:
        last: requests.Response | None = None
        for attempt in range(self.retries):
            # Yahoo's edge throttles some user-agent strings outright, so a 429
            # is retried under the next agent rather than just backed off.
            self.session.headers["User-Agent"] = _YAHOO_UAS[attempt % len(_YAHOO_UAS)]
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code < 400:
                return resp
            last = resp
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(1.0 * (attempt + 1))
                continue
            break
        # `requests.Response.__bool__` is False for error statuses, so test for None.
        status = last.status_code if last is not None else "no response"
        raise DataSourceError(f"yahoo returned {status} for {url}")

    def fetch(self, ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        period1 = int(pd.Timestamp(start or "1990-01-01").timestamp())
        period2 = int(
            pd.Timestamp(end).timestamp()
            if end
            else datetime.now(tz=timezone.utc).timestamp()
        )
        resp = self._get(
            f"{self.base}/{to_yahoo_symbol(ticker)}",
            {"period1": period1, "period2": period2, "interval": "1d", "events": "div,split"},
        )
        payload = json.loads(resp.text)
        result = (payload.get("chart") or {}).get("result")
        if not result:
            raise DataSourceError(f"yahoo returned no series for {ticker}")
        node = result[0]
        quote = node["indicators"]["quote"][0]
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(node["timestamp"], unit="s", utc=True)
                .tz_convert("America/New_York")
                .normalize()
                .tz_localize(None),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            }
        )
        adj = (node["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        if adj is not None:
            factor = pd.Series(adj, index=frame.index) / frame["close"]
            factor = factor.replace([float("inf"), -float("inf")], pd.NA).ffill().bfill()
            for col in ("open", "high", "low", "close"):
                frame[col] = frame[col] * factor
        frame["ticker"] = ticker.upper()
        frame = frame.dropna(subset=["close"])
        return frame.loc[:, OHLCV_COLUMNS]


SOURCES = {"stooq": StooqSource, "yahoo": YahooSource}


#: Consecutive failures after which a source is skipped for the rest of the run.
CIRCUIT_BREAKER_THRESHOLD = 5
_consecutive_failures: dict[str, int] = {}


def reset_circuit_breakers() -> None:
    """Forget past failures - used by tests and by long-lived processes."""
    _consecutive_failures.clear()


def _source_is_open(name: str) -> bool:
    return _consecutive_failures.get(name, 0) < CIRCUIT_BREAKER_THRESHOLD


def fetch_bars(
    ticker: str,
    source: str = "auto",
    start: str | None = None,
    end: str | None = None,
) -> Bars:
    """Fetch one ticker, trying each configured source in order.

    Under ``auto`` a source that has failed :data:`CIRCUIT_BREAKER_THRESHOLD`
    times consecutively is skipped for the rest of the process. Without it, a
    500-name load against a source that is blocking us pays a full timeout on
    every ticker - which is the difference between a ten-minute load and a
    four-hour one.
    """
    order = ["stooq", "yahoo"] if source == "auto" else [source]
    if source == "auto":
        live = [name for name in order if _source_is_open(name)]
        order = live or order[-1:]
    errors: list[str] = []
    for name in order:
        try:
            frame = SOURCES[name]().fetch(ticker, start=start, end=end)
            if frame.empty:
                raise DataSourceError(f"{name} returned an empty frame for {ticker}")
            _consecutive_failures[name] = 0
            log.info("fetched %s bars for %s from %s", len(frame), ticker, name)
            return Bars(ticker=ticker.upper(), frame=frame, source=name)
        except Exception as exc:  # noqa: BLE001 - try the next source
            _consecutive_failures[name] = _consecutive_failures.get(name, 0) + 1
            errors.append(f"{name}: {exc}")
            log.warning("source %s failed for %s (%s)", name, ticker, exc)
            if _consecutive_failures[name] == CIRCUIT_BREAKER_THRESHOLD:
                log.warning(
                    "source %s failed %d times in a row; skipping it for the rest "
                    "of this run", name, CIRCUIT_BREAKER_THRESHOLD,
                )
    raise DataSourceError(f"all sources failed for {ticker} -> {errors}")


def load_universe(
    tickers: list[str],
    client: QuestDBClient,
    source: str = "auto",
    start: str | None = None,
    end: str | None = None,
    pause: float = 0.0,
    skip_existing: bool = False,
    on_error: str = "warn",
) -> pd.DataFrame:
    """Fetch every ticker and write it to ``stooq.daily``. Returns a load report.

    ``on_error='warn'`` keeps going when a symbol cannot be fetched, which is
    what a 500-name load needs: a handful of tickers are always delisted,
    renamed, or simply absent upstream, and one of them should not abort the
    other 499.
    """
    client.create_table()
    client.verify_schema()
    log.info("QuestDB: %s", client.build_version())
    existing: set[str] = set()
    if skip_existing:
        coverage = client.coverage()
        if not coverage.empty:
            existing = {str(t) for t in coverage["ticker"]}
    rows = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        symbol = ticker.strip().upper()
        if symbol in existing:
            rows.append({"ticker": symbol, "source": "cached", "rows": 0,
                         "first_bar": None, "last_bar": None, "error": ""})
            continue
        try:
            bars = fetch_bars(symbol, source=source, start=start, end=end)
            inserted = client.insert_bars(bars.frame)
            rows.append({"ticker": bars.ticker, "source": bars.source, "rows": inserted,
                         "first_bar": bars.frame["date"].min(),
                         "last_bar": bars.frame["date"].max(), "error": ""})
        except Exception as exc:  # noqa: BLE001
            if on_error == "raise":
                raise
            log.warning("[%d/%d] %s failed: %s", i, total, symbol, exc)
            rows.append({"ticker": symbol, "source": "", "rows": 0,
                         "first_bar": None, "last_bar": None, "error": str(exc)[:160]})
        if i % 25 == 0:
            log.info("[%d/%d] loaded", i, total)
        if pause:
            time.sleep(pause)
    return pd.DataFrame(rows)
