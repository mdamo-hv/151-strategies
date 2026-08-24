"""Wide OHLCV panel: the single data structure every strategy consumes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from strategies151.config import Config
from strategies151.data.questdb import QuestDBClient

log = logging.getLogger(__name__)

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


def select_full_history(
    frame: pd.DataFrame,
    min_coverage: float = 0.99,
) -> tuple[list[str], pd.DataFrame]:
    """Keep the tickers that span the window; report the ones that do not.

    The strategies need a rectangular universe, and :meth:`Panel.dropna_rows`
    enforces that by discarding any date on which a single name is missing.  On
    a six-name universe that is harmless.  On five hundred it is fatal: one 2021
    listing would truncate a decade of history for everyone.  Selecting on
    coverage first keeps the long history and drops the late arrivals instead.

    Note the cost: this excludes companies that listed mid-window, which is a
    second survivorship filter on top of using today's index membership.
    """
    counts = frame.groupby("ticker")["date"].count()
    dates = frame["date"].nunique()
    threshold = min_coverage * dates
    keep = sorted(counts[counts >= threshold].index)
    dropped = (
        counts[counts < threshold]
        .rename("bars")
        .reset_index()
        .assign(
            coverage=lambda f: (f["bars"] / dates).round(3),
            first_bar=lambda f: f["ticker"].map(frame.groupby("ticker")["date"].min()),
        )
        .sort_values("coverage")
    )
    return keep, dropped


def load_panel(
    cfg: Config,
    client: QuestDBClient | None = None,
    tickers: Sequence[str] | None = None,
    min_coverage: float | None = None,
) -> Panel:
    client = client or QuestDBClient(cfg.questdb)
    wanted = list(tickers) if tickers is not None else list(cfg.universe.tickers)
    frame = client.read_bars(wanted, cfg.universe.start, cfg.universe.end)
    if frame.empty:
        raise RuntimeError(
            f"no rows in {cfg.questdb.table} for {len(wanted)} requested tickers; "
            "run `s151 load` first"
        )
    coverage = cfg.universe.min_coverage if min_coverage is None else min_coverage
    if coverage and coverage > 0 and len(wanted) > 1:
        keep, dropped = select_full_history(frame, coverage)
        if not dropped.empty:
            log.info(
                "dropping %d of %d tickers below %.0f%% coverage of the window "
                "(shortest: %s)",
                len(dropped), frame["ticker"].nunique(), coverage * 100,
                ", ".join(dropped["ticker"].head(5)),
            )
        if not keep:
            raise RuntimeError(
                f"no ticker covers at least {coverage:.0%} of the requested window"
            )
        frame = frame[frame["ticker"].isin(keep)]
        wanted = [t for t in wanted if t in set(keep)]
    return Panel.from_long(frame, wanted)
