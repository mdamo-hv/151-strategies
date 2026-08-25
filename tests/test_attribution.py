"""Per-ticker P&L attribution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from strategies151.backtest.engine import asset_pnl, buy_and_hold, portfolio_pnl, walk_forward
from strategies151.backtest.report import (
    format_ticker_performance,
    format_universe_attribution,
    ticker_attribution,
    ticker_performance,
    universe_attribution,
)
from strategies151.config import Config
from strategies151.data.panel import Panel
from strategies151.strategies.registry import build


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = Config()
    return replace(base, backtest=replace(base.backtest, train_days=252, test_days=21))


@pytest.fixture(scope="module")
def result(panel: Panel, cfg: Config):
    return walk_forward(build("3.1.price_momentum"), panel, cfg)


# --------------------------------------------------------------- decomposition --
def test_asset_pnl_sums_to_the_portfolio_series(panel: Panel):
    weights = build("3.11.single_moving_average").weights(panel)
    returns = panel.returns
    parts = asset_pnl(weights, returns, cost_bps=5.0, delay=1)
    portfolio = portfolio_pnl(weights, returns, cost_bps=5.0, delay=1)
    for part, column in (("gross", "gross_return"), ("cost", "cost"), ("net", "net_return"),
                         ("turnover", "turnover")):
        pd.testing.assert_series_equal(
            parts[part].sum(axis=1), portfolio[column], check_names=False, atol=1e-15
        )


def test_held_weights_are_the_lagged_targets(panel: Panel):
    weights = build("3.11.single_moving_average").weights(panel)
    parts = asset_pnl(weights, panel.returns, delay=1)
    pd.testing.assert_frame_equal(parts["held"].iloc[1:], weights.iloc[:-1].set_axis(
        parts["held"].index[1:]
    ), atol=1e-15)


# ----------------------------------------------------------------- attribution --
def test_contributions_sum_to_the_strategy_return(result):
    attribution = result.ticker_attribution()
    total = attribution["contribution_ann_%"].sum() / 100
    assert total == pytest.approx(result.stats["ann_return"], rel=1e-9, abs=1e-12)


def test_every_traded_ticker_appears_once(result, panel: Panel):
    attribution = result.ticker_attribution()
    assert list(attribution["ticker"].sort_values()) == sorted(panel.tickers)


def test_attribution_is_sorted_by_contribution(result):
    values = result.ticker_attribution()["contribution_ann_%"].tolist()
    assert values == sorted(values, reverse=True)


def test_shares_of_pnl_add_to_one_hundred(result):
    shares = result.ticker_attribution()["share_of_pnl_%"]
    assert shares.sum() == pytest.approx(100.0, abs=1e-6)


def test_gross_contribution_minus_cost_is_net(result):
    attribution = result.ticker_attribution()
    diff = (
        attribution["gross_contribution_ann_%"]
        - attribution["cost_ann_%"]
        - attribution["contribution_ann_%"]
    )
    assert diff.abs().max() < 1e-9


def test_long_and_short_days_are_bounded_by_days_held(result):
    attribution = result.ticker_attribution()
    total = attribution["long_days_%"] + attribution["short_days_%"]
    assert (total <= attribution["days_held_%"] + 1e-9).all()


def test_long_only_strategy_never_attributes_short_exposure(panel: Panel, cfg: Config):
    attribution = walk_forward(build("4.1.momentum_rotation"), panel, cfg).ticker_attribution()
    assert (attribution["short_days_%"] == 0).all()
    assert (attribution["avg_net_weight"] >= -1e-12).all()


def test_benchmark_attributes_evenly_across_the_universe(panel: Panel, cfg: Config):
    attribution = buy_and_hold(panel, cfg).ticker_attribution()
    weights = attribution["avg_gross_weight"]
    assert weights.std() < 1e-12
    assert weights.iloc[0] == pytest.approx(1.0 / len(panel.tickers))


# -------------------------------------------------------------- rolled-up views --
def test_universe_attribution_covers_every_ticker(panel: Panel, cfg: Config):
    results = [
        walk_forward(build(k), panel, cfg)
        for k in ("3.11.single_moving_average", "3.4.low_volatility")
    ]
    universe = universe_attribution(ticker_attribution(results))
    assert sorted(universe["ticker"]) == sorted(panel.tickers)
    assert (universe["strategies"] == 2).all()


def test_universe_attribution_excludes_the_benchmark(panel: Panel, cfg: Config):
    results = [walk_forward(build("3.4.low_volatility"), panel, cfg), buy_and_hold(panel, cfg)]
    universe = universe_attribution(ticker_attribution(results))
    assert (universe["strategies"] == 1).all()


def test_universe_attribution_omits_buy_and_hold_without_performance(panel: Panel, cfg: Config):
    universe = universe_attribution(ticker_attribution([walk_forward(build("3.4.low_volatility"), panel, cfg)]))
    assert "buy_hold_ann_pct" not in universe.columns


def test_universe_attribution_adds_the_buy_and_hold_benchmark(panel: Panel, cfg: Config):
    results = [walk_forward(build("3.4.low_volatility"), panel, cfg)]
    performance = ticker_performance(panel, index=results[0].daily.index)
    universe = universe_attribution(ticker_attribution(results), performance)

    expected = performance.set_index("ticker")["ann_return"] * 100
    actual = universe.set_index("ticker")["buy_hold_ann_pct"]
    pd.testing.assert_series_equal(
        actual, expected.reindex(actual.index), check_names=False
    )
    assert np.allclose(
        universe["edge_vs_buy_hold_pct"],
        universe["per_unit_contribution_ann_pct"] - universe["buy_hold_ann_pct"],
    )


def test_buy_and_holds_only_edge_over_itself_is_its_cost(panel: Panel, cfg: Config):
    """The per-unit rescaling must recover buy and hold for a strategy that is buy and hold.

    Equal-weight buy and hold holds every name at a constant weight, so dividing
    its contribution by that weight gives the name's own return back - less the
    rebalancing cost, because contributions are net of costs while the
    ``buy_hold_ann_pct`` benchmark is a costless paper return. Pinning the gap to
    exactly the cost drag is what proves the rescaling itself is unbiased.
    """
    result = buy_and_hold(panel, cfg)
    attribution = ticker_attribution([result])
    attribution["style"] = "not-a-benchmark"  # keep it in the roll-up
    performance = ticker_performance(panel, index=result.daily.index)
    universe = universe_attribution(attribution, performance)

    cost_per_unit = (
        attribution.set_index("ticker")["cost_ann_%"]
        / attribution.set_index("ticker")["avg_gross_weight"]
    )
    expected = -cost_per_unit.reindex(universe["ticker"]).to_numpy()
    assert np.allclose(universe["edge_vs_buy_hold_pct"], expected, atol=1e-8)


def test_format_universe_attribution_keeps_weights_visible(panel: Panel, cfg: Config):
    """Weights on a wide universe are ~1/N, so two decimals would round them to zero."""
    results = [walk_forward(build("3.4.low_volatility"), panel, cfg)]
    universe = universe_attribution(ticker_attribution(results))
    display = format_universe_attribution(universe)
    assert (display["avg_gross_weight"] > 0).any()


def test_ticker_performance_reports_the_headline_four(panel: Panel):
    performance = ticker_performance(panel)
    assert sorted(performance["ticker"]) == sorted(panel.tickers)
    for column in ("cagr", "sharpe", "max_drawdown", "calmar"):
        assert column in performance.columns
    assert np.isfinite(performance["sharpe"].to_numpy()).all()


def test_ticker_performance_matches_a_single_name_buy_and_hold(panel: Panel):
    from strategies151.backtest import metrics as m

    performance = ticker_performance(panel).set_index("ticker")
    ticker = panel.tickers[0]
    expected = m.sharpe(panel.returns[ticker].dropna())
    assert performance.loc[ticker, "sharpe"] == pytest.approx(expected)


def test_ticker_performance_respects_the_index_filter(panel: Panel):
    window = panel.close.index[-200:]
    performance = ticker_performance(panel, index=window)
    assert performance["days"].max() <= 200


def test_format_ticker_performance_labels_units(panel: Panel):
    formatted = format_ticker_performance(ticker_performance(panel))
    assert list(formatted.columns)[:5] == [
        "ticker", "ann_return_%", "sharpe", "max_drawdown_%", "calmar",
    ]
