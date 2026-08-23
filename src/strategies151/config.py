"""Typed view over ``configs/default.yaml``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class QuestDBConfig:
    host: str = "localhost"
    http_port: int = 9000
    pg_port: int = 8812
    user: str = "admin"
    password: str = "quest"
    database: str = "qdb"
    table: str = "stooq.daily"

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    @property
    def quoted_table(self) -> str:
        """QuestDB table names containing a dot must be quoted in SQL."""
        return f"'{self.table}'"


@dataclass(frozen=True)
class UniverseConfig:
    tickers: Sequence[str] = ("NVDA", "TSLA", "MSFT", "AMZN", "WMT", "JPM")
    start: str | None = "2015-01-01"
    end: str | None = None


@dataclass(frozen=True)
class DataConfig:
    source: str = "auto"
    history_years: int = 12


@dataclass(frozen=True)
class BacktestConfig:
    train_days: int = 252
    test_days: int = 21
    step_days: int = 21
    min_train_days: int = 200
    cost_bps: float = 5.0
    investment_level: float = 1.0
    annualization: int = 252
    delay: int = 1


@dataclass(frozen=True)
class SelectionConfig:
    objective: str = "sharpe"
    fallback_to_default: bool = True


@dataclass(frozen=True)
class Config:
    questdb: QuestDBConfig = field(default_factory=QuestDBConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output_dir: Path = REPO_ROOT / "results"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        cfg = cls(
            questdb=QuestDBConfig(**(raw.get("questdb") or {})),
            universe=UniverseConfig(**(raw.get("universe") or {})),
            data=DataConfig(**(raw.get("data") or {})),
            backtest=BacktestConfig(**(raw.get("backtest") or {})),
            selection=SelectionConfig(**(raw.get("selection") or {})),
            output_dir=Path((raw.get("output") or {}).get("dir", "results")),
        )
        if not cfg.output_dir.is_absolute():
            cfg = replace(cfg, output_dir=REPO_ROOT / cfg.output_dir)
        return cfg.with_env_overrides()

    def with_env_overrides(self) -> "Config":
        """Environment wins over the file, so CI/containers can retarget QuestDB."""
        qdb = self.questdb
        env = os.environ
        qdb = QuestDBConfig(
            host=env.get("QUESTDB_HOST", qdb.host),
            http_port=int(env.get("QUESTDB_HTTP_PORT", qdb.http_port)),
            pg_port=int(env.get("QUESTDB_PG_PORT", qdb.pg_port)),
            user=env.get("QUESTDB_USER", qdb.user),
            password=env.get("QUESTDB_PASSWORD", qdb.password),
            database=env.get("QUESTDB_DATABASE", qdb.database),
            table=env.get("QUESTDB_TABLE", qdb.table),
        )
        return replace(self, questdb=qdb)
