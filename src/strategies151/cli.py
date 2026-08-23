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
from strategies151.backtest.charts import summary_chart, ticker_chart
from strategies151.backtest.engine import buy_and_hold, common_folds, walk_forward
from strategies151.backtest.perticker import (
    ACTIVE,
    run_ticker_study,
    studies_frame,
    winners_frame,
)
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
from strategies151.strategies.registry import implemented_keys, resolve

log = logging.getLogger("strategies151")


def _print(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("(no rows)")
        return
    print(frame.to_string(index=False))


# ------------------------------------------------------------------ commands --
def cmd_load(args, cfg: Config) -> int:
    client = QuestDBClient(cfg.questdb)
    if not client.ping():
        print(f"cannot reach QuestDB at {cfg.questdb.http_url}", file=sys.stderr)
        return 2
    tickers = args.tickers or list(cfg.universe.tickers)
    start = args.start or (
        pd.Timestamp.today().normalize() - pd.DateOffset(years=cfg.data.history_years)
    ).strftime("%Y-%m-%d")
    report = load_universe(tickers, client, source=args.source or cfg.data.source, start=start)
    _print(report)
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
    if args.tickers:
        cfg = replace(cfg, universe=replace(cfg.universe, tickers=args.tickers))
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
    if args.tickers:
        cfg = replace(cfg, universe=replace(cfg.universe, tickers=args.tickers))
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
        study = run_ticker_study(ticker, panel, cfg, strategy_keys=args.strategies)
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
        chart = ticker_chart(study, output_dir / f"{ticker}.png")
        study.table.assign(params=study.table["params"].apply(json.dumps, default=str)).to_csv(
            output_dir / f"{ticker}_strategies.csv", index=False
        )
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
    (output_dir / "summary.md").write_text(_per_ticker_markdown(studies, winners, cfg, panel))
    (output_dir / "index.html").write_text(_per_ticker_html(studies, stamp))

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


def _per_ticker_markdown(studies, winners: pd.DataFrame, cfg: Config, panel) -> str:
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


def _per_ticker_html(studies, stamp: str) -> str:
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
{cards}
</body></html>
"""

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
    p_bt.add_argument("--start")
    p_bt.add_argument("--train-days", type=int)
    p_bt.add_argument("--test-days", type=int)
    p_bt.add_argument("--step-days", type=int)
    p_bt.add_argument("--cost-bps", type=float)
    p_bt.add_argument("--output")
    p_bt.add_argument("--no-plot", action="store_true")
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
    p_pt.add_argument("--strategies", nargs="+", help="restrict the search (default: all)")
    p_pt.add_argument("--train-days", type=int)
    p_pt.add_argument("--test-days", type=int)
    p_pt.add_argument("--cost-bps", type=float)
    p_pt.add_argument("--output", help="root directory (default: data/)")
    p_pt.add_argument("--stamp", help="folder name (default: YYYYMMDDHHMM)")
    p_pt.set_defaults(func=cmd_per_ticker)
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
