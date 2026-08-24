"""``s151`` command line: load bars, inspect the catalog, run the study."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from strategies151 import catalog
from strategies151.backtest.charts import significance_chart, summary_chart, ticker_chart
from strategies151.backtest.engine import buy_and_hold, common_folds, walk_forward
from strategies151.backtest.perticker import (
    ACTIVE,
    run_ticker_study,
    significance_frame,
    studies_frame,
    winners_frame,
)
from strategies151.backtest import significance as sig
from strategies151.backtest.report import (
    leaderboard,
    markdown_summary,
    plot_equity_curves,
    ticker_attribution,
    ticker_performance,
    write_results,
)
from strategies151.config import REPO_ROOT, Config
from strategies151.data.loaders import load_universe
from strategies151.data.panel import load_panel
from strategies151.data.questdb import QuestDBClient
from strategies151.data.universe import sp500_constituents
from strategies151.strategies.registry import implemented_keys, resolve

log = logging.getLogger("strategies151")


def _print(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("(no rows)")
        return
    print(frame.to_string(index=False))



def _sp500_tickers() -> list[str]:
    """The membership saved by `s151 load --sp500`."""
    path = REPO_ROOT / "configs" / "sp500.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `s151 load --sp500` first")
    return [str(t) for t in pd.read_csv(path)["ticker"]]

def _stratified_sample(tickers: list[str], size: int) -> list[str]:
    """A sector-spread sample of the universe.

    Running the per-ticker study on all 500 names would take about a day - the
    cost is per name and does not fall with universe size. Sampling across GICS
    sectors gives a representative spread of verdicts instead of an arbitrary
    alphabetical prefix.
    """
    path = REPO_ROOT / "configs" / "sp500.csv"
    wanted = list(dict.fromkeys(tickers))
    if not path.exists():
        return wanted[:size]
    meta = pd.read_csv(path)
    meta = meta[meta["ticker"].isin(wanted)]
    if meta.empty:
        return wanted[:size]
    chosen: list[str] = []
    groups = {sector: list(frame["ticker"]) for sector, frame in meta.groupby("sector")}
    round_index = 0
    while len(chosen) < size and any(len(v) > round_index for v in groups.values()):
        for sector in sorted(groups):
            if len(chosen) >= size:
                break
            names = groups[sector]
            if len(names) > round_index:
                chosen.append(names[round_index])
        round_index += 1
    return chosen[:size]

# ------------------------------------------------------------------ commands --
def cmd_load(args, cfg: Config) -> int:
    client = QuestDBClient(cfg.questdb)
    if not client.ping():
        print(f"cannot reach QuestDB at {cfg.questdb.http_url}", file=sys.stderr)
        return 2
    if args.sp500:
        constituents = sp500_constituents()
        tickers = [c["ticker"] for c in constituents]
        pd.DataFrame(constituents).to_csv(REPO_ROOT / "configs" / "sp500.csv", index=False)
        log.info("S&P 500 membership: %d tickers (saved to configs/sp500.csv)", len(tickers))
    else:
        tickers = args.tickers or list(cfg.universe.tickers)
    start = args.start or (
        pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.data.history_years)
    ).strftime("%Y-%m-%d")
    report = load_universe(
        tickers, client, source=args.source or cfg.data.source, start=start,
        pause=args.pause, skip_existing=args.skip_existing,
    )
    failed = report[report["error"] != ""] if "error" in report else pd.DataFrame()
    _print(report.head(20) if len(report) > 20 else report)
    if len(report) > 20:
        print(f"... {len(report)} tickers; {len(report) - len(failed)} loaded, "
              f"{len(failed)} failed")
    if not failed.empty:
        print("\nfailed:")
        _print(failed.loc[:, ["ticker", "error"]])
    return 0


def cmd_status(args, cfg: Config) -> int:
    client = QuestDBClient(cfg.questdb)
    if not client.ping():
        print(f"cannot reach QuestDB at {cfg.questdb.http_url}", file=sys.stderr)
        return 2
    print(f"table: {cfg.questdb.table}")
    _print(client.coverage())
    return 0


def cmd_catalog(args, cfg: Config) -> int:
    frame = catalog.as_frame()
    if args.status:
        frame = frame[frame["status"] == args.status]
    if args.chapter:
        frame = frame[frame["chapter"].str.contains(args.chapter, case=False, na=False)]
    if args.summary:
        _print(catalog.summary().reset_index())
        return 0
    columns = ["section", "chapter", "title", "status", "strategies"]
    if args.verbose:
        columns += ["requires", "note"]
    _print(frame.loc[:, columns])
    print(f"\n{len(frame)} of {len(catalog.CATALOG)} catalog entries shown")
    return 0


def cmd_strategies(args, cfg: Config) -> int:
    rows = []
    for strategy in resolve():
        rows.append(
            {
                "section": strategy.section,
                "key": strategy.key,
                "title": strategy.title,
                "style": strategy.style,
                "grid": len(list(type(strategy).grid())),
                "warmup": strategy.warmup,
            }
        )
    _print(pd.DataFrame(rows).sort_values("section"))
    return 0


def cmd_backtest(args, cfg: Config) -> int:
    if args.train_days or args.test_days or args.step_days or args.cost_bps is not None:
        bt = cfg.backtest
        cfg = replace(
            cfg,
            backtest=replace(
                bt,
                train_days=args.train_days or bt.train_days,
                test_days=args.test_days or bt.test_days,
                step_days=args.step_days or args.test_days or bt.step_days,
                cost_bps=bt.cost_bps if args.cost_bps is None else args.cost_bps,
            ),
        )
    tickers = _sp500_tickers() if args.sp500 else args.tickers
    if tickers:
        cfg = replace(cfg, universe=replace(cfg.universe, tickers=tickers))
    if args.start:
        cfg = replace(cfg, universe=replace(cfg.universe, start=args.start))

    panel = load_panel(cfg)
    log.info(
        "panel: %s bars x %s tickers, %s to %s",
        len(panel), len(panel.tickers), panel.dates[0].date(), panel.dates[-1].date(),
    )

    keys = args.strategies or implemented_keys()
    strategies = resolve(keys)
    folds = common_folds(strategies, panel, cfg) if args.align_folds else None
    if args.align_folds and not folds:
        print("not enough history for a single aligned fold", file=sys.stderr)
        return 2

    results = []
    for strategy in strategies:
        try:
            result = walk_forward(strategy, panel, cfg, folds=folds)
        except Exception as exc:  # noqa: BLE001 - one bad strategy must not sink the run
            log.error("%s failed: %s", strategy.key, exc)
            continue
        results.append(result)
        log.info(
            "%-42s ann.return %7.2f%%  sharpe %6.2f  max.dd %7.2f%%  calmar %6.2f  (%.1fs)",
            result.key,
            result.stats.get("cagr", float("nan")) * 100,
            result.stats.get("sharpe", float("nan")),
            result.stats.get("max_drawdown", float("nan")) * 100,
            result.stats.get("calmar", float("nan")),
            result.elapsed_s,
        )
    if not results:
        print("no strategy produced a result", file=sys.stderr)
        return 1

    oos_index = results[0].daily.index
    for res in results[1:]:
        oos_index = oos_index.union(res.daily.index)
    results.append(buy_and_hold(panel, cfg, index=oos_index))

    output_dir = Path(args.output) if args.output else cfg.output_dir
    context = {
        "tickers": list(panel.tickers),
        "table": cfg.questdb.table,
        "bars": len(panel),
        "first_bar": str(panel.dates[0].date()),
        "last_bar": str(panel.dates[-1].date()),
        "train_days": cfg.backtest.train_days,
        "test_days": cfg.backtest.test_days,
        "step_days": cfg.backtest.step_days,
        "cost_bps": cfg.backtest.cost_bps,
        "delay": cfg.backtest.delay,
        "objective": cfg.selection.objective,
        "folds": int(results[0].stats.get("folds", 0)),
        "oos_start": str(oos_index[0].date()),
        "oos_end": str(oos_index[-1].date()),
        "aligned_folds": bool(args.align_folds),
    }
    paths = write_results(
        results,
        output_dir,
        context=context,
        panel=panel,
        per_ticker_daily=args.per_ticker_daily,
    )

    # Is the top of the leaderboard real, or the best of 28 tries?
    strategies_only = [r for r in results if r.style != "benchmark"]
    benchmark = next((r for r in results if r.style == "benchmark"), None)
    joint = {}
    if benchmark is not None and len(strategies_only) > 1:
        per_strategy, joint = sig.assess_study(
            pd.DataFrame({r.key: r.net_returns for r in strategies_only}),
            benchmark.net_returns,
            labels={r.key: f"{r.section} {r.title}" for r in strategies_only},
            annualization=cfg.backtest.annualization,
            draws=args.bootstrap_draws,
        )
        per_strategy.to_csv(output_dir / "significance_by_strategy.csv", index=False)
        pd.DataFrame([joint]).to_csv(output_dir / "significance.csv", index=False)
        significance_chart(
            per_strategy.assign(spa_p=float("nan")),
            output_dir / "significance.png",
            label_column="title",
            p_column="p_vs_benchmark",
            t_column="t_stat_vs_benchmark",
            title="Does any strategy beat the equal-weighted benchmark?",
            show_verdict=False,
            max_rows=20,
            subtitle=(
                f"Each strategy's annualised return in excess of the benchmark, with a "
                f"95% Newey-West interval.\nBest of {joint['n_candidates']} candidates: "
                f"Hansen's SPA p = {joint['spa_p']:.3f}, Reality Check p = "
                f"{joint['reality_check_p']:.3f} - the chance of a maximum this large "
                "when no strategy has an edge."
            ),
        )
        log.info(
            "joint test: best = %s, excess %+.2f%%/yr, SPA p = %.3f, RC p = %.3f",
            joint["best_title"], joint["best_excess_ann_return"] * 100,
            joint["spa_p"], joint["reality_check_p"],
        )
    if not args.no_plot:
        plot_equity_curves(results, output_dir / "equity_curves.png")
    board = leaderboard(results)
    attribution = ticker_attribution(results)
    print(
        markdown_summary(
            board,
            context,
            performance=ticker_performance(panel, index=oos_index),
            attribution=attribution,
        )
    )
    print(f"artifacts written to {output_dir}")
    return 0


def cmd_per_ticker(args, cfg: Config) -> int:
    """Find the best strategy for each ticker individually and chart it."""
    tickers = _sp500_tickers() if args.sp500 else args.tickers
    if args.sample:
        tickers = _stratified_sample(tickers or list(cfg.universe.tickers), args.sample)
    if tickers:
        cfg = replace(cfg, universe=replace(cfg.universe, tickers=tickers))
    if args.train_days or args.test_days or args.cost_bps is not None:
        bt = cfg.backtest
        cfg = replace(
            cfg,
            backtest=replace(
                bt,
                train_days=args.train_days or bt.train_days,
                test_days=args.test_days or bt.test_days,
                step_days=args.test_days or bt.step_days,
                cost_bps=bt.cost_bps if args.cost_bps is None else args.cost_bps,
            ),
        )

    panel = load_panel(cfg)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d%H%M")
    output_dir = Path(args.output or (REPO_ROOT / "data")) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("writing per-ticker study to %s", output_dir)

    studies = []
    for ticker in panel.tickers:
        study = run_ticker_study(
            ticker, panel, cfg, strategy_keys=args.strategies,
            bootstrap_draws=args.bootstrap_draws, bootstrap_block=args.bootstrap_block,
        )
        studies.append(study)
        if study.best is None:
            log.info("%-6s no applicable strategy on a one-name universe", ticker)
        else:
            log.info(
                "%-6s best %-38s sharpe %5.2f  ann.return %7.2f%%  max.dd %7.2f%%  calmar %5.2f",
                ticker,
                f"{study.best.section} {study.best.title}",
                study.best_stats.get("sharpe", float("nan")),
                study.best_stats.get("cagr", float("nan")) * 100,
                study.best_stats.get("max_drawdown", float("nan")) * 100,
                study.best_stats.get("calmar", float("nan")),
            )
        if study.significance:
            log.info(
                "       vs buy&hold %+6.2f%%/yr  p=%.3f | selection-corrected SPA p=%.3f | %s",
                study.significance["excess_ann_return"] * 100,
                study.significance["p_vs_buy_hold"],
                study.significance["spa_p"],
                study.verdict,
            )
        chart = ticker_chart(study, output_dir / f"{ticker}.png")
        study.table.assign(params=study.table["params"].apply(json.dumps, default=str)).to_csv(
            output_dir / f"{ticker}_strategies.csv", index=False
        )
        if not study.candidate_returns.empty:
            # Persisted so `s151 significance` can re-run the tests - with more
            # bootstrap draws, or a different block length - without a backtest.
            series = study.candidate_returns.copy()
            series["__benchmark__"] = study.benchmark.net_returns
            series.to_csv(output_dir / f"{ticker}_daily_returns.csv")
        log.info("  chart: %s", chart)

    winners = winners_frame(studies)
    winners.to_csv(output_dir / "best_per_ticker.csv", index=False)
    studies_frame(studies).to_csv(output_dir / "all_results.csv", index=False)
    summary_chart(
        winners,
        output_dir / "summary.png",
        subtitle=(
            f"Each ticker tested on its own against all {studies[0].tested} strategies. "
            f"{cfg.backtest.train_days}d train \u2192 {cfg.backtest.test_days}d test, walked "
            f"forward {studies[0].folds} times, {cfg.backtest.cost_bps:.0f} bps costs.\n"
            "Bars are the winning strategy's Sharpe; the marker is that ticker's own buy & hold."
        ),
    )
    tests = significance_frame(studies)
    if not tests.empty:
        tests.to_csv(output_dir / "significance.csv", index=False)
        significance_chart(
            tests, output_dir / "significance.png",
            subtitle=(
                "Winning strategy minus that ticker's own buy & hold. The p-value is "
                "Hansen's SPA:\nthe chance of a maximum this large arising from "
                f"{studies[0].applicable} candidates when none has an edge."
            ),
        )
    (output_dir / "summary.md").write_text(
        _per_ticker_markdown(studies, winners, cfg, panel, tests)
    )
    (output_dir / "index.html").write_text(_per_ticker_html(studies, stamp, not tests.empty))

    _print(
        winners.assign(
            ann_return_pct=(winners["cagr"] * 100).round(2),
            max_drawdown_pct=(winners["max_drawdown"] * 100).round(2),
            sharpe=winners["sharpe"].round(2),
            calmar=winners["calmar"].round(2),
        ).loc[:, ["ticker", "section", "best_strategy", "ann_return_pct", "sharpe",
                  "max_drawdown_pct", "calmar", "applicable_strategies"]]
    )
    print(f"\nartifacts written to {output_dir}")
    return 0


def _per_ticker_markdown(studies, winners: pd.DataFrame, cfg: Config, panel,
                         tests: pd.DataFrame | None = None) -> str:
    lines = [
        "# Best strategy per ticker",
        "",
        f"* Universe tested one name at a time: `{', '.join(panel.tickers)}`",
        f"* Bars: `{cfg.questdb.table}`, {len(panel)} rows, "
        f"{panel.dates[0].date()} to {panel.dates[-1].date()}",
        f"* Windows: {cfg.backtest.train_days} training days -> "
        f"{cfg.backtest.test_days} test days, walked forward",
        f"* Transaction cost: {cfg.backtest.cost_bps} bps, delay {cfg.backtest.delay}",
        "",
        "Roughly half the library is cross-sectional and cannot express a view on a "
        "single name: demeaning one stock's return against itself gives zero, and a "
        "top third that is also the bottom third nets out. Those are detected and "
        "excluded from the ranking with the reason recorded in "
        "`<TICKER>_strategies.csv`.",
        "",
    ]
    display = winners.copy()
    for column in ("cagr", "max_drawdown", "buy_hold_cagr", "buy_hold_max_drawdown"):
        if column in display:
            display[column] = (display[column] * 100).round(2)
    for column in ("sharpe", "calmar", "buy_hold_sharpe", "sharpe_vs_buy_hold", "ann_turnover"):
        if column in display:
            display[column] = display[column].round(2)
    keep = ["ticker", "section", "best_strategy", "cagr", "sharpe", "max_drawdown",
            "calmar", "buy_hold_sharpe", "sharpe_vs_buy_hold", "applicable_strategies"]
    keep = [c for c in keep if c in display]
    display = display.loc[:, keep].rename(
        columns={"cagr": "ann_return_%", "max_drawdown": "max_drawdown_%"}
    )
    lines += [display.to_markdown(index=False), ""]

    if tests is not None and not tests.empty:
        lines += _significance_markdown(tests).splitlines()

    for study in studies:
        lines += [f"## {study.ticker}", ""]
        if study.best is None:
            lines += ["No applicable strategy.", ""]
            continue
        lines += [
            f"**{study.best.section} {study.best.title}** — "
            f"{study.applicable} of {study.tested} strategies applicable.",
            "",
            "Parameters selected in-sample:",
            "",
            "```",
            "\n".join(f"{k} = {v}" for k, v in study.best_params.items()),
            "```",
            "",
            f"![{study.ticker}]({study.ticker}.png)",
            "",
        ]
    return "\n".join(lines)


def _per_ticker_html(studies, stamp: str, has_significance: bool = False) -> str:
    cards = "\n".join(
        f'  <section><h2>{s.ticker}'
        + (f' <small>{s.best.section} {s.best.title}</small>' if s.best else ' <small>no applicable strategy</small>')
        + f'</h2><img src="{s.ticker}.png" alt="{s.ticker} chart"></section>'
        for s in studies
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Best strategy per ticker - {stamp}</title>
<style>
  body {{ background: #f9f9f7; color: #0b0b0b; margin: 0; padding: 32px;
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  p.sub {{ color: #52514e; margin: 0 0 28px; }}
  h2 {{ font-size: 16px; margin: 0 0 10px; }}
  h2 small {{ color: #52514e; font-weight: 400; margin-left: 8px; }}
  section {{ background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
             border-radius: 8px; padding: 18px; margin-bottom: 24px; }}
  img {{ width: 100%; height: auto; display: block; }}
</style></head><body>
<h1>Best strategy per ticker</h1>
<p class="sub">Walk-forward study generated {stamp}. Full tables in
<code>best_per_ticker.csv</code> and <code>all_results.csv</code>.</p>
<section><h2>Summary</h2><img src="summary.png" alt="summary chart"></section>
{'<section><h2>Statistical significance</h2><img src="significance.png" alt="significance chart"></section>' if has_significance else ''}
{cards}
</body></html>
"""

SIGNIFICANCE_HEADING = "## Is it statistically relevant, or luck?"


def _significance_markdown(tests: pd.DataFrame) -> str:
    """The significance section, shared by `per-ticker` and `significance`."""
    view = tests.copy()
    view["excess_ann_return_%"] = (view["excess_ann_return"] * 100).round(2)
    for column in ("t_stat_vs_zero", "t_stat_vs_buy_hold"):
        view[column] = view[column].round(2)
    for column in ("p_vs_zero", "p_vs_buy_hold", "reality_check_p", "spa_p",
                   "deflated_sharpe_prob"):
        view[column] = view[column].round(3)
    keep = ["ticker", "best_strategy", "excess_ann_return_%", "t_stat_vs_buy_hold",
            "p_vs_buy_hold", "reality_check_p", "spa_p", "deflated_sharpe_prob", "verdict"]
    return "\n".join([
        SIGNIFICANCE_HEADING,
        "",
        "Three different questions, in increasing order of how much they ask:",
        "",
        "* `p_vs_buy_hold` - a one-sided Newey-West t-test that the strategy's daily "
        "return exceeds that ticker's own buy & hold. Autocorrelation-robust, but it "
        "takes the winner as given. A value near 1 does not mean "
        "\"no difference\" - it means the difference runs the other way.",
        "* `reality_check_p` / `spa_p` - White's Reality Check and Hansen's SPA, which "
        "bootstrap all applicable candidates jointly (stationary bootstrap, so serial "
        "dependence survives resampling) and ask how often chance alone produces a "
        "maximum this large. **This is the test that accounts for the winner having "
        "been selected as the best of many.**",
        "* `deflated_sharpe_prob` - the probability the winner's Sharpe survives "
        "deflation for the number of trials, their dispersion, and the return "
        "distribution's skew and kurtosis.",
        "",
        view.loc[:, keep].to_markdown(index=False),
        "",
        "![significance](significance.png)",
        "",
    ])


def _splice_significance(path: Path, block: str) -> None:
    """Replace the significance section of an existing summary.md in place."""
    if not path.exists():
        return
    text = path.read_text()
    if SIGNIFICANCE_HEADING not in text:
        path.write_text(text.rstrip() + "\n\n" + block)
        return
    start = text.index(SIGNIFICANCE_HEADING)
    rest = text.find("\n## ", start + len(SIGNIFICANCE_HEADING))
    end = len(text) if rest == -1 else rest + 1
    path.write_text(text[:start] + block.rstrip() + "\n\n" + text[end:])


def cmd_significance(args, cfg: Config) -> int:
    """Re-run the significance tests from a saved per-ticker folder."""
    folder = Path(args.folder)
    files = sorted(folder.glob("*_daily_returns.csv"))
    if not files:
        print(f"no *_daily_returns.csv in {folder}; run `s151 per-ticker` first", file=sys.stderr)
        return 2
    winners = pd.read_csv(folder / "best_per_ticker.csv").set_index("ticker")

    rows = []
    for path in files:
        ticker = path.name.removesuffix("_daily_returns.csv")
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        benchmark = frame.pop("__benchmark__")
        if ticker not in winners.index or frame.empty:
            continue
        best_key = winners.loc[ticker, "key"]
        if best_key not in frame.columns:
            continue
        rows.append(
            sig.assess(
                ticker=ticker,
                best_name=best_key,
                best_returns=frame[best_key],
                benchmark_returns=benchmark,
                candidate_returns=frame,
                annualization=cfg.backtest.annualization,
                draws=args.draws,
                block=args.block,
                seed=args.seed,
            )
        )
    if not rows:
        print("no ticker had both a winner and a saved return series", file=sys.stderr)
        return 1

    tests = pd.DataFrame(rows)
    tests["verdict"] = [sig.verdict(r) for r in rows]
    tests.to_csv(folder / "significance.csv", index=False)
    significance_chart(
        tests, folder / "significance.png",
        subtitle=(
            "Winning strategy minus that ticker's own buy & hold. The p-value is "
            f"Hansen's SPA:\nthe chance of a maximum this large arising from "
            f"{int(tests['n_candidates'].iloc[0])} candidates when none has an edge."
        ),
    )
    _splice_significance(folder / "summary.md", _significance_markdown(tests))
    display = tests.copy()
    display["excess_%"] = (display["excess_ann_return"] * 100).round(2)
    _print(display.loc[:, ["ticker", "best_strategy", "excess_%", "t_stat_vs_buy_hold",
                           "p_vs_buy_hold", "reality_check_p", "spa_p",
                           "deflated_sharpe_prob", "verdict"]].round(3))
    print(f"\nwritten to {folder}")
    return 0

# --------------------------------------------------------------------- parser --
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s151",
        description="151 Trading Strategies research track over QuestDB stooq.daily bars.",
    )
    parser.add_argument("--config", help="path to a YAML config (default: configs/default.yaml)")
    parser.add_argument("-v", "--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="download daily bars into QuestDB")
    p_load.add_argument("--tickers", nargs="+")
    p_load.add_argument("--source", choices=["auto", "stooq", "yahoo"])
    p_load.add_argument("--start")
    p_load.add_argument("--sp500", action="store_true",
                        help="load the current S&P 500 membership instead of --tickers")
    p_load.add_argument("--pause", type=float, default=0.0,
                        help="seconds to wait between tickers, to stay under rate limits")
    p_load.add_argument("--skip-existing", action="store_true",
                        help="leave tickers already present in the table untouched")
    p_load.set_defaults(func=cmd_load)

    p_status = sub.add_parser("status", help="show per-ticker coverage in the bar table")
    p_status.set_defaults(func=cmd_status)

    p_catalog = sub.add_parser("catalog", help="list the paper's strategies and their status")
    p_catalog.add_argument("--status", choices=["implemented", "substituted", "not_implemented"])
    p_catalog.add_argument("--chapter")
    p_catalog.add_argument("--summary", action="store_true")
    p_catalog.add_argument("--verbose", action="store_true")
    p_catalog.set_defaults(func=cmd_catalog)

    p_strategies = sub.add_parser("strategies", help="list runnable strategy implementations")
    p_strategies.set_defaults(func=cmd_strategies)

    p_bt = sub.add_parser("backtest", help="run the sliding-window out-of-sample study")
    p_bt.add_argument("--strategies", nargs="+", help="strategy keys (default: all)")
    p_bt.add_argument("--tickers", nargs="+")
    p_bt.add_argument("--sp500", action="store_true",
                      help="use the S&P 500 membership saved by `s151 load --sp500`")
    p_bt.add_argument("--start")
    p_bt.add_argument("--train-days", type=int)
    p_bt.add_argument("--test-days", type=int)
    p_bt.add_argument("--step-days", type=int)
    p_bt.add_argument("--cost-bps", type=float)
    p_bt.add_argument("--output")
    p_bt.add_argument("--no-plot", action="store_true")
    p_bt.add_argument("--bootstrap-draws", type=int, default=5000,
                      help="resamples for the joint significance test")
    p_bt.add_argument(
        "--per-ticker-daily",
        action="store_true",
        help="also write each strategy's daily per-ticker P&L contributions "
             "to results/by_ticker/ (roughly 300 KB per strategy)",
    )
    p_bt.add_argument(
        "--no-align-folds",
        dest="align_folds",
        action="store_false",
        help="let each strategy use as many folds as its warmup allows "
             "(default: share one fold schedule so results are comparable)",
    )
    p_bt.set_defaults(func=cmd_backtest, align_folds=True)

    p_pt = sub.add_parser(
        "per-ticker",
        help="find the best strategy for each ticker on its own and chart it",
    )
    p_pt.add_argument("--tickers", nargs="+")
    p_pt.add_argument("--sp500", action="store_true",
                      help="use the S&P 500 membership saved by `s151 load --sp500`")
    p_pt.add_argument("--sample", type=int,
                      help="test only this many names, spread across GICS sectors")
    p_pt.add_argument("--strategies", nargs="+", help="restrict the search (default: all)")
    p_pt.add_argument("--train-days", type=int)
    p_pt.add_argument("--test-days", type=int)
    p_pt.add_argument("--cost-bps", type=float)
    p_pt.add_argument("--output", help="root directory (default: data/)")
    p_pt.add_argument("--stamp", help="folder name (default: YYYYMMDDHHMM)")
    p_pt.add_argument("--bootstrap-draws", type=int, default=5000)
    p_pt.add_argument("--bootstrap-block", type=float, default=10.0)
    p_pt.set_defaults(func=cmd_per_ticker)

    p_sig = sub.add_parser(
        "significance",
        help="re-run the significance tests on a saved per-ticker folder",
    )
    p_sig.add_argument("folder", help="a data/YYYYMMDDHHMM directory")
    p_sig.add_argument("--draws", type=int, default=5000, help="bootstrap resamples")
    p_sig.add_argument("--block", type=float, default=10.0,
                       help="expected stationary-bootstrap block length, in days")
    p_sig.add_argument("--seed", type=int, default=20260823)
    p_sig.set_defaults(func=cmd_significance)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config.load(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
