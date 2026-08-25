"""Thin QuestDB client: DDL + bulk insert over HTTP, reads over the PG wire."""

from __future__ import annotations

import io
import json
import logging
from typing import Iterable, Sequence

import pandas as pd
import requests

from strategies151.config import QuestDBConfig
from strategies151.data.universe import to_db_symbol

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]

#: Column names other loaders use for the instrument identifier.
TICKER_ALIASES = ("ticker", "symbol", "instrument", "sym")


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
        if self.cfg.read_only:
            log.info("%s is read-only; leaving its schema alone", self.cfg.table)
            return
        ticker, date = self.cfg.ticker_column, self.cfg.date_column
        self.exec(
            f"CREATE TABLE IF NOT EXISTS {self.cfg.quoted_table} ("
            f"  {ticker} SYMBOL CAPACITY 4096 CACHE,"
            f"  {date} TIMESTAMP,"
            "  open DOUBLE,"
            "  high DOUBLE,"
            "  low DOUBLE,"
            "  close DOUBLE,"
            "  volume DOUBLE"
            f") TIMESTAMP({date}) PARTITION BY YEAR WAL"
            f" DEDUP UPSERT KEYS({date}, {ticker})"
        )

    def drop_table(self) -> None:
        self._refuse_if_read_only("drop")
        self.exec(f"DROP TABLE IF EXISTS {self.cfg.quoted_table}")

<<<<<<< Updated upstream
=======
    def verify_schema(self) -> None:
        """Fail loudly if an existing table cannot receive - or serve - bars.

        ``/imp`` uses an existing table's column types and ignores the schema it
        is handed, so a table created earlier by another tool - with the date as
        a STRING, say - rejects every batch with ``not a timestamp`` no matter
        what the client sends. A table that simply names its columns differently
        is not broken at all: it only needs ``questdb.ticker_column`` /
        ``questdb.date_column`` pointed at the names it uses.
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
        date_col = self.cfg.date_column
        if date_col not in types:
            raise QuestDBError(
                f"table {self.cfg.table} has no column {date_col!r} "
                f"(it has {sorted(types)}). Point questdb.date_column and "
                "questdb.ticker_column in configs/default.yaml at the names this "
                f"table actually uses - {self._suggest_columns(types)} - rather "
                "than dropping a table another loader owns."
            )
        if types[date_col] != "TIMESTAMP":
            raise QuestDBError(
                f"table {self.cfg.table} already exists with {date_col} typed "
                f"{types[date_col]}, not TIMESTAMP. QuestDB /imp uses the "
                "existing table's types, so every insert will be rejected. "
                "Drop it and reload:\n"
                f"  curl -G http://{self.cfg.host}:{self.cfg.http_port}/exec "
                f"--data-urlencode \"query=DROP TABLE '{self.cfg.table}'\""
            )
        required = {self.cfg.ticker_column, "open", "high", "low", "close", "volume"}
        missing = required - set(types)
        if missing:
            raise QuestDBError(
                f"table {self.cfg.table} is missing columns {sorted(missing)}. "
                f"Candidate names in the table: {self._suggest_columns(types)}"
            )

    def _refuse_if_read_only(self, verb: str) -> None:
        if self.cfg.read_only:
            raise QuestDBError(
                f"refusing to {verb} {self.cfg.table}: it is marked read-only in "
                "the config because another loader owns it. Set "
                "questdb.read_only: false, or point questdb.table at a table "
                "this project owns."
            )

    @staticmethod
    def _suggest_columns(types: dict[str, str]) -> str:
        """Best guess at the ticker/date columns of a foreign table, for errors."""
        ticker = next(
            (c for c, t in types.items() if t == "SYMBOL" and c in TICKER_ALIASES), "?"
        )
        date = next((c for c, t in types.items() if t == "TIMESTAMP"), "?")
        return f"ticker_column: {ticker}, date_column: {date}"

>>>>>>> Stashed changes
    # --------------------------------------------------------------- writes --
    def insert_bars(self, bars: pd.DataFrame) -> int:
        """Bulk-insert OHLCV rows. ``bars`` must carry :data:`OHLCV_COLUMNS`."""
        self._refuse_if_read_only("write to")
        missing = set(OHLCV_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        frame = bars.loc[:, OHLCV_COLUMNS].copy()
        frame = frame.dropna(subset=["ticker", "date", "close"])
        if frame.empty:
            return 0
<<<<<<< Updated upstream
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
=======
        timestamps = pd.to_datetime(frame["date"], utc=True)
        ticker_col, date_col = self.cfg.ticker_column, self.cfg.date_column
        frame["ticker"] = frame["ticker"].map(to_db_symbol)
        frame = frame.rename(columns={"ticker": ticker_col, "date": date_col})
        errors: list[str] = []
        for date_format, pattern in self._timestamp_formats():
            frame[date_col] = timestamps.dt.strftime(date_format)
            buf = io.StringIO()
            frame.to_csv(buf, index=False)
            schema = [
                {"name": ticker_col, "type": "SYMBOL"},
                {"name": date_col, "type": "TIMESTAMP", "pattern": pattern},
                {"name": "open", "type": "DOUBLE"},
                {"name": "high", "type": "DOUBLE"},
                {"name": "low", "type": "DOUBLE"},
                {"name": "close", "type": "DOUBLE"},
                {"name": "volume", "type": "DOUBLE"},
            ]
            resp = requests.post(
                f"{self.cfg.http_url}/imp",
                params={"name": self.cfg.table, "timestamp": date_col,
                        "fmt": "json"},
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
>>>>>>> Stashed changes
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
        tickers = [to_db_symbol(t) for t in tickers]
        ticker_col, date_col = self.cfg.ticker_column, self.cfg.date_column
        where = [ticker_col + " IN (" + ", ".join(f"'{t}'" for t in tickers) + ")"]
        if start:
            where.append(f"{date_col} >= '{start}'")
        if end:
            where.append(f"{date_col} <= '{end}'")
        sql = (
            f"SELECT {ticker_col} AS ticker, {date_col} AS date, "
            f"open, high, low, close, volume "
            f"FROM {self.cfg.quoted_table} WHERE {' AND '.join(where)} "
            f"ORDER BY {date_col}, {ticker_col}"
        )
        frame = self.query(sql)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def coverage(self, tickers: Iterable[str] | None = None) -> pd.DataFrame:
        """Per-ticker row count and date range - used by the CLI status command.

        ``tickers`` narrows the scan, which matters on a shared table holding
        thousands of names that this study never looks at.
        """
        ticker_col, date_col = self.cfg.ticker_column, self.cfg.date_column
        where = ""
        if tickers is not None:
            names = ", ".join(f"'{to_db_symbol(t)}'" for t in tickers)
            if not names:
                return pd.DataFrame(columns=["ticker", "bars", "first_bar", "last_bar"])
            where = f"WHERE {ticker_col} IN ({names}) "
        sql = (
            f"SELECT {ticker_col} AS ticker, count() AS bars, "
            f"min({date_col}) AS first_bar, max({date_col}) AS last_bar "
            f"FROM {self.cfg.quoted_table} {where}ORDER BY ticker"  # the alias: an
            # aggregate's ORDER BY must name a column in the select list.
        )
        return self.query(sql)
