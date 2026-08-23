from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies151.data.panel import Panel

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def synthetic_frame(n_days: int = 1400, seed: int = 7) -> pd.DataFrame:
    """Long-format OHLCV with a mild common factor plus idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    market = rng.normal(0.0004, 0.010, n_days)
    rows = []
    for i, ticker in enumerate(TICKERS):
        beta = 0.6 + 0.2 * i
        shocks = beta * market + rng.normal(0.0002 * (i - 2), 0.012, n_days)
        close = 50.0 * (1 + i) * np.exp(np.cumsum(shocks))
        intraday = np.abs(rng.normal(0.0, 0.008, n_days))
        rows.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": close * (1 - rng.normal(0.0, 0.004, n_days)),
                    "high": close * (1 + intraday),
                    "low": close * (1 - intraday),
                    "close": close,
                    "volume": rng.lognormal(15.0, 0.4, n_days),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def long_frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="session")
def panel(long_frame: pd.DataFrame) -> Panel:
    return Panel.from_long(long_frame, TICKERS)
