"""Wide OHLCV panel: the single data structure every strategy consumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from strategies151.config import Config
from strategies151.data.questdb import QuestDBClient

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Panel:
    """Aligned per-field frames, indexed by date with one column per ticker.

    Every field shares the same index and column order, so strategies can do
    plain elementwise arithmetic across fields without re-aligning.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def __len__(self) -> int:
        return len(self.close.index)

    # ---------------------------------------------------------- derivations --
    @property
    def returns(self) -> pd.DataFrame:
        """Simple close-to-close returns (the P&L series the engine trades)."""
        return self.close.pct_change()

    @property
    def log_returns(self) -> pd.DataFrame:
        """Log close-to-close returns, Eq. (285)-(286) / (292) of the paper."""
        return np.log(self.close / self.close.shift(1))

    @property
    def intraday_returns(self) -> pd.DataFrame:
        """``ln(close/open)`` - the delay-1 momentum input of Appendix A."""
        return np.log(self.close / self.open)

    @property
    def overnight_returns(self) -> pd.DataFrame:
        """``ln(open_t / close_{t-1})`` - the delay-0 mean-reversion input."""
        return np.log(self.open / self.close.shift(1))

    @property
    def dollar_volume(self) -> pd.DataFrame:
        """ADDV input: ``volume * close`` (Appendix A's ``vol * close``)."""
        return self.volume * self.close

    def slice(self, start: int, stop: int) -> "Panel":
        """Positional slice shared across all fields."""
        return Panel(**{f: getattr(self, f).iloc[start:stop] for f in FIELDS})

    def loc(self, index: pd.Index) -> "Panel":
        return Panel(**{f: getattr(self, f).loc[index] for f in FIELDS})

    def tail(self, n: int) -> "Panel":
        return self.slice(max(len(self) - n, 0), len(self))

    @classmethod
    def from_long(cls, frame: pd.DataFrame, tickers: Sequence[str] | None = None) -> "Panel":
        if frame.empty:
            raise ValueError("cannot build a Panel from an empty frame")
        wide = {}
        for field in FIELDS:
            pivot = frame.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
            wide[field] = pivot.sort_index()
        columns = list(tickers) if tickers else sorted(wide["close"].columns)
        columns = [c for c in columns if c in wide["close"].columns]
        index = wide["close"].index
        panel = cls(**{f: wide[f].reindex(index=index, columns=columns) for f in FIELDS})
        return panel.dropna_rows()

    def dropna_rows(self) -> "Panel":
        """Keep only dates on which every ticker has a close.

        Cross-sectional strategies (mean-reversion, momentum ranking, portfolio
        optimisation) need a rectangular universe; ragged history at the front
        of a young ticker would otherwise silently change the universe size.
        """
        valid = self.close.notna().all(axis=1)
        index = self.close.index[valid]
        return self.loc(index)


def load_panel(cfg: Config, client: QuestDBClient | None = None) -> Panel:
    client = client or QuestDBClient(cfg.questdb)
    frame = client.read_bars(cfg.universe.tickers, cfg.universe.start, cfg.universe.end)
    if frame.empty:
        raise RuntimeError(
            f"no rows in {cfg.questdb.table} for {list(cfg.universe.tickers)}; "
            "run `s151 load` first"
        )
    return Panel.from_long(frame, cfg.universe.tickers)
