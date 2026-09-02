"""Bar storage: QuestDB in a container, or a local DuckDB file.

Both back ends hold the same table - ``stooq.daily``, one row per ticker per
trading day - and expose the same handful of operations, so every other module
takes a ``BarStore`` and never learns which one it got.

Which to use:

``questdb``
    A server in Docker. Purpose-built for time series, and what the committed
    results were produced against. Needs the container running.
``duckdb``
    A single file on disk, no server and no Docker. Simpler to set up and to
    copy around; the natural choice for a laptop or CI.

QuestDB has no schemas, so ``stooq.daily`` is one quoted table name there; in
DuckDB ``stooq`` is a real schema and ``daily`` a table inside it. The
distinction is confined to the two implementations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import pandas as pd

log = logging.getLogger(__name__)

#: Column order every source normalises to and every back end stores.
OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")


class StoreError(RuntimeError):
    pass


@runtime_checkable
class BarStore(Protocol):
    """What the rest of the project needs from a bar store."""

    def ping(self) -> bool: ...
    def create_table(self) -> None: ...
    def verify_schema(self) -> None: ...
    def drop_table(self) -> None: ...
    def insert_bars(self, bars: pd.DataFrame) -> int: ...
    def read_bars(
        self, tickers: Iterable[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame: ...
    def coverage(self) -> pd.DataFrame: ...
    def build_version(self) -> str: ...
    @property
    def description(self) -> str: ...


def _normalise_read(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every back end's reader identical dtypes."""
    if frame.empty:
        return frame
    frame = frame.copy()
    dates = pd.to_datetime(frame["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    frame["date"] = dates.dt.normalize()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


class DuckDBStore:
    """``stooq.daily`` in a local DuckDB file.

    The primary key on ``(ticker, date)`` plus ``ON CONFLICT ... DO UPDATE``
    gives the same idempotent reload QuestDB's ``DEDUP UPSERT KEYS`` provides,
    so re-running a load over an overlapping range overwrites instead of
    duplicating.

    DuckDB allows a single writer at a time. That is not a constraint here -
    only ``s151 load`` writes, and the parallel per-ticker study reads its panel
    once in the parent process - but two concurrent loads into one file will not
    work.
    """

    def __init__(self, path: str | Path, table: str = "stooq.daily", read_only: bool = False):
        self.path = Path(path).expanduser()
        self.table = table
        self.read_only = read_only
        schema, _, name = table.rpartition(".")
        self.schema = schema or "main"
        self.name = name
        self._connection = None
        self._catalog_name: str | None = None

    # ------------------------------------------------------------ plumbing --
    @property
    def qualified(self) -> str:
        """Catalog-qualified name.

        DuckDB names the catalog after the file, so a database at
        ``stooq.duckdb`` holding a ``stooq`` schema makes ``"stooq"."daily"``
        ambiguous - catalog or schema? Naming all three parts removes the
        question for any filename the user picks.
        """
        return f'"{self._catalog()}"."{self.schema}"."{self.name}"'

    def _catalog(self) -> str:
        if self._catalog_name is None:
            self._catalog_name = self.connect().execute(
                "SELECT current_database()"
            ).fetchone()[0]
        return self._catalog_name

    @property
    def description(self) -> str:
        return f"duckdb {self.path}"

    def connect(self):
        import duckdb

        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = duckdb.connect(str(self.path), read_only=self.read_only)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._catalog_name = None

    def ping(self) -> bool:
        try:
            self.connect().execute("SELECT 1").fetchone()
            return True
        except Exception:  # noqa: BLE001 - a ping must never raise
            return False

    def build_version(self) -> str:
        try:
            import duckdb

            return f"DuckDB {duckdb.__version__} ({self.path})"
        except Exception:  # noqa: BLE001 - diagnostics must not raise
            return "unknown"

    # ----------------------------------------------------------------- DDL --
    def create_table(self) -> None:
        con = self.connect()
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._catalog()}"."{self.schema}"')
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {self.qualified} ("
            "  ticker VARCHAR NOT NULL,"
            "  date   TIMESTAMP NOT NULL,"
            "  open   DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,"
            "  PRIMARY KEY (ticker, date)"
            ")"
        )

    def drop_table(self) -> None:
        self.connect().execute(f"DROP TABLE IF EXISTS {self.qualified}")

    def verify_schema(self) -> None:
        con = self.connect()
        rows = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ?",
            [self.schema, self.name],
        ).fetchall()
        if not rows:
            return
        types = {name: kind.upper() for name, kind in rows}
        if not types.get("date", "").startswith("TIMESTAMP"):
            raise StoreError(
                f"{self.table} in {self.path} has date typed "
                f"{types.get('date', 'missing')}, not TIMESTAMP. Drop it and reload:\n"
                f"  python -c \"import duckdb; duckdb.connect('{self.path}')"
                f".execute('DROP TABLE {self.qualified}')\""
            )
        missing = set(OHLCV_COLUMNS) - set(types)
        if missing:
            raise StoreError(f"{self.table} is missing columns {sorted(missing)}")

    # -------------------------------------------------------------- writes --
    def insert_bars(self, bars: pd.DataFrame) -> int:
        missing = set(OHLCV_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        frame = bars.loc[:, OHLCV_COLUMNS].dropna(subset=["ticker", "date", "close"]).copy()
        if frame.empty:
            return 0
        frame["date"] = pd.to_datetime(frame["date"])
        if getattr(frame["date"].dt, "tz", None) is not None:
            frame["date"] = frame["date"].dt.tz_localize(None)
        # A batch that repeats a key would trip "cannot update the same row
        # twice"; the last row wins, matching the upsert's own semantics.
        frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last")

        self.create_table()
        con = self.connect()
        con.register("_incoming_bars", frame)
        try:
            con.execute(
                f"INSERT INTO {self.qualified} "
                f"SELECT {', '.join(OHLCV_COLUMNS)} FROM _incoming_bars "
                "ON CONFLICT (ticker, date) DO UPDATE SET "
                "open = excluded.open, high = excluded.high, low = excluded.low, "
                "close = excluded.close, volume = excluded.volume"
            )
        finally:
            con.unregister("_incoming_bars")
        return len(frame)

    # --------------------------------------------------------------- reads --
    def read_bars(
        self,
        tickers: Iterable[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        wanted = [t.strip().upper() for t in tickers]
        if not wanted:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        self.create_table()
        clauses, params = ["ticker IN (" + ", ".join("?" * len(wanted)) + ")"], list(wanted)
        if start:
            clauses.append("date >= ?")
            params.append(pd.Timestamp(start).to_pydatetime())
        if end:
            clauses.append("date <= ?")
            params.append(pd.Timestamp(end).to_pydatetime())
        sql = (
            f"SELECT {', '.join(OHLCV_COLUMNS)} FROM {self.qualified} "
            f"WHERE {' AND '.join(clauses)} ORDER BY date, ticker"
        )
        return _normalise_read(self.connect().execute(sql, params).df())

    def coverage(self) -> pd.DataFrame:
        self.create_table()
        frame = self.connect().execute(
            f"SELECT ticker, count(*) AS bars, min(date) AS first_bar, "
            f"max(date) AS last_bar FROM {self.qualified} GROUP BY ticker ORDER BY ticker"
        ).df()
        if frame.empty:
            return pd.DataFrame(columns=["ticker", "bars", "first_bar", "last_bar"])
        for column in ("first_bar", "last_bar"):
            frame[column] = pd.to_datetime(frame[column])
        return frame


def open_store(cfg, backend: str | None = None) -> BarStore:
    """Build the configured back end.

    ``backend`` overrides ``cfg.storage.backend`` so a single ``--store`` flag
    can retarget any command without touching the config file.
    """
    from strategies151.data.questdb import QuestDBClient

    chosen = (backend or cfg.storage.backend).lower()
    if chosen == "questdb":
        return QuestDBClient(cfg.questdb)
    if chosen == "duckdb":
        path = Path(cfg.storage.duckdb_path)
        if not path.is_absolute():
            from strategies151.config import REPO_ROOT

            path = REPO_ROOT / path
        return DuckDBStore(path, table=cfg.questdb.table)
    raise StoreError(f"unknown storage backend '{chosen}'; use questdb or duckdb")
