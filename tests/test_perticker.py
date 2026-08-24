"""Per-ticker study: applicability screening, ranking and chart artifacts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from strategies151.backtest.charts import summary_chart, ticker_chart
from strategies151.backtest.perticker import (
    ACTIVE,
    BUY_AND_HOLD,
    NO_POSITION,
    classify,
    modal_params,
    run_ticker_study,
    studies_frame,
    takes_a_position,
    winners_frame,
)
from strategies151.config import Config
from strategies151.data.panel import Panel
from strategies151.strategies.registry import build, get


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = Config()
    return replace(base, backtest=replace(base.backtest, train_days=252, test_days=21))


@pytest.fixture(scope="module")
def single(panel: Panel) -> Panel:
    ticker = panel.tickers[0]
    return Panel(**{f: getattr(panel, f)[[ticker]] for f in
                    ("open", "high", "low", "close", "volume")})


@pytest.fixture(scope="module")
def study(panel: Panel, cfg: Config):
    return run_ticker_study(
        panel.tickers[0], panel, cfg,
        strategy_keys=[
            "3.11.single_moving_average",   # single-name, active
            "3.15.channel",                 # single-name, active
            "3.9.mean_reversion_single_cluster",  # cross-sectional, degenerate
            "4.1.momentum_rotation",        # long-only ranking -> buy & hold
        ],
    )


# ------------------------------------------------------------------ screening --
def test_cross_sectional_strategy_is_screened_out(single: Panel, cfg: Config):
    assert not takes_a_position(get("3.9.mean_reversion_single_cluster"), single, cfg)


def test_single_name_strategy_survives_screening(single: Panel, cfg: Config):
    assert takes_a_position(get("3.11.single_moving_average"), single, cfg)


def test_screening_keeps_a_grid_with_any_workable_setting(single: Panel, cfg: Config):
    """10.4 is flat when demeaning is on but not when it is off - keep it."""
    assert takes_a_position(get("10.4.trend_following"), single, cfg)


def test_screening_passes_on_a_multi_name_universe(panel: Panel, cfg: Config):
    assert takes_a_position(get("3.9.mean_reversion_single_cluster"), panel, cfg)


# -------------------------------------------------------------- classification --
def test_classify_flags_a_flat_result(study):
    flat = study.table[study.table["key"] == "3.9.mean_reversion_single_cluster"]
    assert flat["applicability"].iloc[0] == NO_POSITION


def test_classify_flags_buy_and_hold_equivalence(study):
    row = study.table[study.table["key"] == "4.1.momentum_rotation"]
    assert row["applicability"].iloc[0] == BUY_AND_HOLD


def test_active_strategies_are_the_ones_ranked(study):
    assert (study.ranked["applicability"] == ACTIVE).all()
    assert study.applicable == len(study.ranked)


def test_ranking_is_by_descending_sharpe(study):
    sharpes = study.ranked["sharpe"].tolist()
    assert sharpes == sorted(sharpes, reverse=True)


def test_best_is_the_top_active_strategy(study):
    assert study.best is not None
    assert study.best.key == study.ranked["key"].iloc[0]


def test_degenerate_strategies_are_reported_not_dropped(study):
    assert study.tested == 4
    assert set(study.table["key"]) == {
        "3.11.single_moving_average", "3.15.channel",
        "3.9.mean_reversion_single_cluster", "4.1.momentum_rotation",
    }


# ------------------------------------------------------------------ parameters --
def test_modal_params_reports_the_per_axis_mode():
    folds = pd.DataFrame({"params": [{"length": 200}, {"length": 200}, {"length": 50}]})
    assert modal_params(folds) == {"length": "200 (67% of folds)"}


def test_modal_params_omits_the_share_when_unanimous():
    folds = pd.DataFrame({"params": [{"length": 200}, {"length": 200}]})
    assert modal_params(folds) == {"length": "200"}


def test_modal_params_handles_json_encoded_folds():
    folds = pd.DataFrame({"params": ['{"length": 200}', '{"length": 200}']})
    assert modal_params(folds) == {"length": "200"}


def test_best_params_are_populated(study):
    assert study.best_params
    assert all(isinstance(k, str) for k in study.best_params)


# ---------------------------------------------------------------------- study --
def test_position_series_covers_the_out_of_sample_window(study):
    position = study.best_position()
    assert len(position) == len(study.oos_index)
    assert np.isfinite(position.to_numpy()).all()


def test_benchmark_shares_the_strategy_window(study):
    assert study.benchmark.daily.index.equals(study.best.daily.index)


def test_winners_frame_has_one_row_per_ticker(study):
    winners = winners_frame([study])
    assert len(winners) == 1
    assert winners["ticker"].iloc[0] == study.ticker
    assert winners["sharpe_vs_buy_hold"].notna().all()


def test_studies_frame_serialises_parameters(study):
    frame = studies_frame([study])
    assert len(frame) == study.tested
    assert frame["params"].map(lambda p: isinstance(p, str)).all()


# --------------------------------------------------------------------- charts --
def test_ticker_chart_is_written(study, tmp_path):
    path = ticker_chart(study, tmp_path / f"{study.ticker}.png")
    assert path.exists() and path.stat().st_size > 20_000


def test_summary_chart_is_written(study, tmp_path):
    path = summary_chart(winners_frame([study]), tmp_path / "summary.png")
    assert path.exists() and path.stat().st_size > 10_000


def test_summary_chart_renders_from_a_saved_table(study, tmp_path):
    """The chart must be reproducible from best_per_ticker.csv alone."""
    csv = tmp_path / "best_per_ticker.csv"
    winners_frame([study]).to_csv(csv, index=False)
    path = summary_chart(pd.read_csv(csv), tmp_path / "from_csv.png")
    assert path.exists() and path.stat().st_size > 10_000


def test_summary_chart_skips_tickers_without_a_winner(tmp_path):
    empty = pd.DataFrame([{"ticker": "AAA", "best_strategy": None, "sharpe": None}])
    assert summary_chart(empty, tmp_path / "none.png") == tmp_path / "none.png"


def test_chart_survives_a_ticker_with_no_applicable_strategy(panel: Panel, cfg: Config, tmp_path):
    study = run_ticker_study(
        panel.tickers[0], panel, cfg,
        strategy_keys=["3.9.mean_reversion_single_cluster"],
    )
    assert study.best is None
    assert ticker_chart(study, tmp_path / "empty.png").exists()


def test_parallel_and_sequential_studies_agree(panel: Panel, cfg: Config, tmp_path):
    """Workers must not change any number - only the wall-clock.

    Each ticker is an independent one-name study, so parallelising the loop is
    safe; this pins that down rather than assuming it.
    """
    from strategies151.cli import _per_ticker_worker, _single_ticker_panel

    keys = ["3.11.single_moving_average", "3.15.channel"]
    payloads = [
        (t, _single_ticker_panel(panel, t), cfg, keys, 200, 10.0)
        for t in panel.tickers[:2]
    ]
    direct = [_per_ticker_worker(p) for p in payloads]

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(_per_ticker_worker, payloads, chunksize=1))

    for a, b in zip(direct, parallel):
        assert a.ticker == b.ticker
        assert a.best.key == b.best.key
        pd.testing.assert_frame_equal(
            a.table.drop(columns="params"), b.table.drop(columns="params")
        )
        assert a.significance["spa_p"] == pytest.approx(b.significance["spa_p"])


def test_single_ticker_panel_keeps_one_column(panel: Panel):
    from strategies151.cli import _single_ticker_panel

    single = _single_ticker_panel(panel, panel.tickers[0])
    assert single.tickers == [panel.tickers[0]]
    assert len(single) == len(panel)
