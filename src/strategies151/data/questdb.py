"""Thin QuestDB client: DDL + bulk insert over HTTP, reads over the PG wire."""

from __future__ import annotations

import io
import json
import logging
from typing import Iterable, Sequence

import pandas as pd
import requests

from strategies151.config import QuestDBConfig

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


class QuestDBError(RuntimeError):
    pass


class QuestDBClient:
    """Access layer for the ``stooq.daily`` bar table.

    Writes go through the REST ``/imp`` CSV endpoint (fast, and it is the only
    QuestDB ingest path that handles the dotted table name without a dedicated
    ILP schema). Reads go through the Postgres wire protocol so pandas can
    stream results.
    """

    def __init__(self, cfg: QuestDBConfig | None = None, timeout: float = 60.0):
        self.cfg = cfg or QuestDBConfig()
        self.timeout = timeout

    # ------------------------------------------------------------------ DDL --
    def exec(self, query: str) -> dict:
        resp = requests.get(
            f"{self.cfg.http_url}/exec",
            params={"query": query},
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code >= 400 or "error" in payload:
            raise QuestDBError(f"{payload.get('error', resp.text)} :: {query}")
        return payload

    def ping(self) -> bool:
        try:
            self.exec("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - a ping must never raise
            return False

    def create_table(self) -> None:
        """Create the bar table if missing.

        ``DEDUP UPSERT KEYS(date, ticker)`` makes re-loading a date range
        idempotent, which matters because loaders are re-run to extend history.
        """
        self.exec(
            f"CREATE TABLE IF NOT EXISTS {self.cfg.quoted_table} ("
            "  ticker SYMBOL CAPACITY 4096 CACHE,"
            "  date TIMESTAMP,"
            "  open DOUBLE,"
            "  high DOUBLE,"
            "  low DOUBLE,"
            "  close DOUBLE,"
            "  volume DOUBLE"
            ") TIMESTAMP(date) PARTITION BY YEAR WAL"
            " DEDUP UPSERT KEYS(date, ticker)"
        )

    def drop_table(self) -> None:
        self.exec(f"DROP TABLE IF EXISTS {self.cfg.quoted_table}")

    # --------------------------------------------------------------- writes --
    def insert_bars(self, bars: pd.DataFrame) -> int:
        """Bulk-insert OHLCV rows. ``bars`` must carry :data:`OHLCV_COLUMNS`."""
        missing = set(OHLCV_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        frame = bars.loc[:, OHLCV_COLUMNS].copy()
        frame = frame.dropna(subset=["ticker", "date", "close"])
        if frame.empty:
            return 0
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        schema = [
            {"name": "ticker", "type": "SYMBOL"},
            {"name": "date", "type": "TIMESTAMP", "pattern": "yyyy-MM-ddTHH:mm:ss.SSSUUUZ"},
            {"name": "open", "type": "DOUBLE"},
            {"name": "high", "type": "DOUBLE"},
            {"name": "low", "type": "DOUBLE"},
            {"name": "close", "type": "DOUBLE"},
            {"name": "volume", "type": "DOUBLE"},
        ]
        resp = requests.post(
            f"{self.cfg.http_url}/imp",
            params={"name": self.cfg.table, "timestamp": "date", "fmt": "json"},
            files={
                "schema": (None, json.dumps(schema)),
                "data": ("bars.csv", buf.getvalue(), "text/csv"),
            },
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise QuestDBError(f"/imp failed: {resp.status_code} {resp.text[:500]}")
        body = resp.json()
        status = body.get("status")
        if status != "OK":
            raise QuestDBError(f"/imp rejected the batch: {body}")
        return len(frame)

    # ---------------------------------------------------------------- reads --
    def query(self, sql: str, params: Sequence | None = None) -> pd.DataFrame:
        import psycopg

        conninfo = (
            f"host={self.cfg.host} port={self.cfg.pg_port} dbname={self.cfg.database} "
            f"user={self.cfg.user} password={self.cfg.password}"
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d.name for d in cur.description or []]
                rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)

    def read_bars(
        self,
        tickers: Iterable[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        tickers = [t.upper() for t in tickers]
        where = ["ticker IN (" + ", ".join(f"'{t}'" for t in tickers) + ")"]
        if start:
            where.append(f"date >= '{start}'")
        if end:
            where.append(f"date <= '{end}'")
        sql = (
            f"SELECT ticker, date, open, high, low, close, volume "
            f"FROM {self.cfg.quoted_table} WHERE {' AND '.join(where)} "
            f"ORDER BY date, ticker"
        )
        frame = self.query(sql)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def coverage(self) -> pd.DataFrame:
        """Per-ticker row count and date range - used by the CLI status command."""
        sql = (
            f"SELECT ticker, count() AS bars, min(date) AS first_bar, max(date) AS last_bar "
            f"FROM {self.cfg.quoted_table} ORDER BY ticker"
        )
        return self.query(sql)
