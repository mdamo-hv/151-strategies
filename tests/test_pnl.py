from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies151.backtest.engine import portfolio_pnl
from strategies151.backtest import metrics as m


@pytest.fixture
def simple():
    index = pd.bdate_range("2020-01-01", periods=4)
    returns = pd.DataFrame({"A": [0.10, 0.10, 0.10, 0.10], "B": [0.0, 0.0, 0.0, 0.0]}, index=index)
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]}, index=index)
    return weights, returns


def test_signal_is_traded_with_a_one_day_delay(simple):
    weights, returns = simple
    pnl = portfolio_pnl(weights, returns, cost_bps=0.0, delay=1)
    # Day 0 has no prior weight, so no P&L; days 1-2 hold A and earn 10%.
    assert pnl["gross_return"].iloc[0] == pytest.approx(0.0)
    assert pnl["gross_return"].iloc[1] == pytest.approx(0.10)
    assert pnl["gross_return"].iloc[2] == pytest.approx(0.10)
    # Day 3 holds B, which returns nothing.
    assert pnl["gross_return"].iloc[3] == pytest.approx(0.0)


def test_delay_zero_would_use_same_day_returns(simple):
    weights, returns = simple
    pnl = portfolio_pnl(weights, returns, cost_bps=0.0, delay=0)
    assert pnl["gross_return"].iloc[0] == pytest.approx(0.10)


def test_costs_are_charged_on_traded_notional(simple):
    weights, returns = simple
    pnl = portfolio_pnl(weights, returns, cost_bps=10.0, delay=1)
    # Rotating A -> B trades 2.0 units of notional at 10bps.
    assert pnl["turnover"].iloc[3] == pytest.approx(2.0)
    assert pnl["cost"].iloc[3] == pytest.approx(2.0 * 10e-4)
    assert (pnl["net_return"] == pnl["gross_return"] - pnl["cost"]).all()


def test_holding_a_constant_book_costs_nothing_after_entry(simple):
    weights, returns = simple
    pnl = portfolio_pnl(weights, returns, cost_bps=10.0, delay=1)
    assert pnl["turnover"].iloc[2] == pytest.approx(0.0)


def test_previous_weights_seed_the_first_bar(simple):
    weights, returns = simple
    prev = pd.Series({"A": 1.0, "B": 0.0})
    pnl = portfolio_pnl(weights, returns, cost_bps=0.0, delay=1, prev_weights=prev)
    assert pnl["gross_return"].iloc[0] == pytest.approx(0.10)
    assert pnl["turnover"].iloc[0] == pytest.approx(0.0)


def test_sharpe_matches_the_papers_definition():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0.001, 0.01, 5000))
    expected = series.mean() / series.std(ddof=1) * np.sqrt(252)
    assert m.sharpe(series) == pytest.approx(expected)


def test_max_drawdown_is_negative_and_bounded():
    series = pd.Series([0.1, -0.5, 0.2, -0.1])
    mdd = m.max_drawdown(series)
    assert -1.0 < mdd < 0.0


def test_summarize_reports_the_cost_decomposition():
    index = pd.bdate_range("2020-01-01", periods=50)
    gross = pd.Series(0.001, index=index)
    net = gross - 0.0002
    turnover = pd.Series(0.4, index=index)
    stats = m.summarize(net, gross_returns=gross, turnover=turnover)
    assert stats["cost_drag_ann"] == pytest.approx(0.0002 * 252)
    assert stats["ann_turnover"] == pytest.approx(0.4 * 252)
