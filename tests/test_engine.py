from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from strategies151.backtest.engine import buy_and_hold, common_folds, walk_forward
from strategies151.backtest.report import leaderboard, markdown_summary
from strategies151.config import Config
from strategies151.data.panel import Panel
from strategies151.strategies.registry import build


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = Config()
    return replace(base, backtest=replace(base.backtest, train_days=252, test_days=21))


@pytest.fixture(scope="module")
def result(panel: Panel, cfg: Config):
    return walk_forward(build("3.11.single_moving_average"), panel, cfg)


def test_out_of_sample_days_equal_folds_times_test_length(result, cfg):
    expected = int(result.stats["folds"]) * cfg.backtest.test_days - cfg.backtest.delay
    assert len(result.daily) == expected


def test_every_fold_records_its_chosen_parameters(result):
    assert not result.folds.empty
    assert result.folds["params"].map(lambda p: isinstance(p, dict)).all()
    assert (result.folds["grid_size"] > 1).all()


def test_training_windows_precede_their_test_windows(result):
    assert (result.folds["train_end"] < result.folds["test_start"]).all()


def test_test_windows_do_not_overlap(result):
    starts = result.folds["test_start"].tolist()
    ends = result.folds["test_end"].tolist()
    for previous_end, next_start in zip(ends, starts[1:]):
        assert previous_end < next_start


def test_daily_track_record_is_finite(result):
    assert np.isfinite(result.daily["net_return"].to_numpy()).all()
    assert (result.daily["gross_exposure"] <= 1.0 + 1e-9).all()


def test_net_is_gross_minus_cost(result):
    diff = result.daily["gross_return"] - result.daily["cost"] - result.daily["net_return"]
    assert diff.abs().max() < 1e-12


def test_zero_cost_dominates_positive_cost(panel: Panel, cfg: Config):
    free = replace(cfg, backtest=replace(cfg.backtest, cost_bps=0.0))
    charged = replace(cfg, backtest=replace(cfg.backtest, cost_bps=50.0))
    strategy_free = walk_forward(build("3.15.channel"), panel, free)
    strategy_charged = walk_forward(build("3.15.channel"), panel, charged)
    assert strategy_free.daily["net_return"].sum() > strategy_charged.daily["net_return"].sum()


def test_stateful_strategy_runs_end_to_end(panel: Panel, cfg: Config):
    result = walk_forward(build("3.8.pairs_trading"), panel, cfg)
    assert len(result.daily) > 0
    assert "fitted_pair" in result.folds.columns


def test_common_folds_are_shared_by_every_strategy(panel: Panel, cfg: Config):
    strategies = [build("3.11.single_moving_average"), build("3.4.low_volatility")]
    folds = common_folds(strategies, panel, cfg)
    results = [walk_forward(s, panel, cfg, folds=folds) for s in strategies]
    assert results[0].daily.index.equals(results[1].daily.index)


def test_insufficient_history_is_reported_clearly(panel: Panel, cfg: Config):
    short = panel.slice(0, 300)
    with pytest.raises(ValueError, match="not enough history"):
        walk_forward(build("3.1.price_momentum"), short, cfg)


def test_benchmark_is_long_only_and_fully_invested(panel: Panel, cfg: Config):
    bench = buy_and_hold(panel, cfg)
    assert bench.daily["gross_exposure"].iloc[-1] == pytest.approx(1.0)
    assert bench.daily["net_exposure"].iloc[-1] == pytest.approx(1.0)


def test_leaderboard_is_sorted_by_sharpe(panel: Panel, cfg: Config):
    results = [
        walk_forward(build(k), panel, cfg)
        for k in ("3.11.single_moving_average", "3.4.low_volatility")
    ]
    board = leaderboard(results)
    sharpes = board["sharpe"].dropna().tolist()
    assert sharpes == sorted(sharpes, reverse=True)
    assert "# 151 Strategies" in markdown_summary(board)
