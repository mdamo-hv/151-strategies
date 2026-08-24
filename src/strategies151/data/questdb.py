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
        self._timestamp_format: tuple[str, str] | None = None

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

    def verify_schema(self) -> None:
        """Fail loudly if an existing table cannot receive bars.

        ``/imp`` uses an existing table's column types and ignores the schema it
        is handed, so a table created earlier by another tool - with ``date`` as
        a STRING, say - rejects every batch with ``not a timestamp 'date'`` no
        matter what the client sends. Saying so beats 503 identical warnings.
        """
        try:
            payload = self.exec(
                f'SELECT "column", type FROM table_columns(\'{self.cfg.table}\')'
            )
        except Exception:  # noqa: BLE001 - an absent table is created on demand
            return
        types = {row[0]: row[1] for row in payload.get("dataset", [])}
        if not types:
            return
        if types.get("date") != "TIMESTAMP":
            raise QuestDBError(
                f"table {self.cfg.table} already exists with date typed "
                f"{types.get('date', 'missing')}, not TIMESTAMP. QuestDB /imp "
                "uses the existing table's types, so every insert will be "
                f"rejected. Drop it and reload:\n"
                f"  curl -G http://{self.cfg.host}:{self.cfg.http_port}/exec "
                f"--data-urlencode \"query=DROP TABLE '{self.cfg.table}'\""
            )
        missing = {"ticker", "open", "high", "low", "close", "volume"} - set(types)
        if missing:
            raise QuestDBError(
                f"table {self.cfg.table} is missing columns {sorted(missing)}"
            )

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
        timestamps = pd.to_datetime(frame["date"], utc=True)
        errors: list[str] = []
        for date_format, pattern in self._timestamp_formats():
            frame["date"] = timestamps.dt.strftime(date_format)
            buf = io.StringIO()
            frame.to_csv(buf, index=False)
            schema = [
                {"name": "ticker", "type": "SYMBOL"},
                {"name": "date", "type": "TIMESTAMP", "pattern": pattern},
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
            if body.get("status") == "OK":
                self._timestamp_format = (date_format, pattern)
                return len(frame)
            errors.append(f"{pattern}: {body.get('status')}")
            # Any other rejection is about the data, not the dialect.
            if "timestamp" not in str(body.get("status", "")).lower():
                raise QuestDBError(f"/imp rejected the batch: {body}")
        raise QuestDBError(
            "/imp rejected every timestamp format this client knows "
            f"(QuestDB {self.build_version()}): {errors}"
        )

    #: Tried in order until one is accepted, then remembered. Daily bars have no
    #: sub-second component, so the plain date is both the simplest and the most
    #: portable; the sub-second patterns are kept as fallbacks because a table
    #: created by another tool may expect them. The microsecond token `UUU` is a
    #: recent addition to QuestDB - on an older build it fails to parse, the
    #: column is not recognised as a timestamp, and /imp answers
    #: `not a timestamp 'date'`.
    TIMESTAMP_FORMATS = (
        ("%Y-%m-%d", "yyyy-MM-dd"),
        ("%Y-%m-%dT%H:%M:%S.%fZ", "yyyy-MM-ddTHH:mm:ss.SSSUUUZ"),
        ("%Y-%m-%dT%H:%M:%SZ", "yyyy-MM-ddTHH:mm:ssZ"),
    )

    def _timestamp_formats(self) -> list[tuple[str, str]]:
        """Formats to try, best-known-good first."""
        if self._timestamp_format is not None:
            return [self._timestamp_format]
        return list(self.TIMESTAMP_FORMATS)

    def build_version(self) -> str:
        """QuestDB build string, or ``unknown`` when it cannot be read."""
        try:
            payload = self.exec("SELECT build()")
            return str(payload["dataset"][0][0])
        except Exception:  # noqa: BLE001 - diagnostics must not raise
            return "unknown"

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
