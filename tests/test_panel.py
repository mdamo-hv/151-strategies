from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies151.data.panel import FIELDS, Panel


def test_fields_share_index_and_columns(panel: Panel):
    for field in FIELDS:
        frame = getattr(panel, field)
        assert frame.index.equals(panel.close.index)
        assert list(frame.columns) == panel.tickers


def test_returns_match_close_ratio(panel: Panel):
    expected = panel.close.iloc[5, 0] / panel.close.iloc[4, 0] - 1.0
    assert panel.returns.iloc[5, 0] == pytest.approx(expected)
    assert np.isnan(panel.returns.iloc[0, 0])


def test_log_returns_are_consistent(panel: Panel):
    recovered = np.exp(panel.log_returns) - 1.0
    pd.testing.assert_frame_equal(
        recovered.iloc[1:], panel.returns.iloc[1:], check_exact=False, atol=1e-12
    )


def test_dropna_rows_removes_partial_dates(long_frame: pd.DataFrame):
    frame = long_frame.copy()
    first_date = frame["date"].min()
    frame = frame[~((frame["ticker"] == "AAA") & (frame["date"] == first_date))]
    built = Panel.from_long(frame)
    assert first_date not in built.close.index
    assert built.close.notna().all().all()


def test_slice_is_positional_and_consistent(panel: Panel):
    piece = panel.slice(10, 40)
    assert len(piece) == 30
    assert piece.close.index[0] == panel.close.index[10]
    for field in FIELDS:
        assert getattr(piece, field).index.equals(piece.close.index)


def test_from_long_rejects_empty():
    with pytest.raises(ValueError):
        Panel.from_long(pd.DataFrame(columns=["ticker", "date", "close"]))
