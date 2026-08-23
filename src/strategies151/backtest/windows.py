"""Sliding train/test window schedule: 1 year in, 1 month out, step forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, in positional index space over the panel."""

    index: int
    train_start: int
    train_end: int  # exclusive
    test_start: int  # == train_end
    test_end: int  # exclusive

    @property
    def train_length(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_length(self) -> int:
        return self.test_end - self.test_start

    def label(self, dates: Sequence[pd.Timestamp]) -> dict:
        return {
            "fold": self.index,
            "train_start": dates[self.train_start],
            "train_end": dates[self.train_end - 1],
            "test_start": dates[self.test_start],
            "test_end": dates[self.test_end - 1],
            "train_days": self.train_length,
            "test_days": self.test_length,
        }


def make_folds(
    n_rows: int,
    train_days: int = 252,
    test_days: int = 21,
    step_days: int | None = None,
    min_train_days: int = 200,
    warmup: int = 0,
) -> list[Fold]:
    """Build the walk-forward schedule.

    The first test window starts at ``warmup + train_days`` so that every
    training window is itself computed on fully warmed-up signals - otherwise
    the in-sample parameter choice would be made on partly-NaN indicators.
    Windows never overlap in the test dimension, so the concatenated test
    windows form one continuous out-of-sample track record.
    """
    step = step_days or test_days
    folds: list[Fold] = []
    test_start = warmup + train_days
    idx = 0
    while test_start + test_days <= n_rows:
        train_start = max(warmup, test_start - train_days)
        if test_start - train_start >= min_train_days:
            folds.append(
                Fold(
                    index=idx,
                    train_start=train_start,
                    train_end=test_start,
                    test_start=test_start,
                    test_end=test_start + test_days,
                )
            )
            idx += 1
        test_start += step
    return folds


def folds_frame(folds: Sequence[Fold], dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame([f.label(dates) for f in folds])
