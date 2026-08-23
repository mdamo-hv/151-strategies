"""``s151`` command line: load bars, inspect the catalog, run the study."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from strategies151 import catalog
from strategies151.backtest.engine import buy_and_hold, common_folds, walk_forward
from strategies151.backtest.report import (
    leaderboard,
    markdown_summary,
    plot_equity_curves,
    write_results,
)
from strategies151.config import Config
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
            "%-42s sharpe %6.2f  cagr %7.2f%%  mdd %7.2f%%  (%.1fs)",
            result.key,
            result.stats.get("sharpe", float("nan")),
            result.stats.get("cagr", float("nan")) * 100,
            result.stats.get("max_drawdown", float("nan")) * 100,
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
    paths = write_results(results, output_dir, context=context)
    if not args.no_plot:
        plot_equity_curves(results, output_dir / "equity_curves.png")
    board = leaderboard(results)
    print(markdown_summary(board, context))
    print(f"artifacts written to {output_dir}")
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
        "--no-align-folds",
        dest="align_folds",
        action="store_false",
        help="let each strategy use as many folds as its warmup allows "
             "(default: share one fold schedule so results are comparable)",
    )
    p_bt.set_defaults(func=cmd_backtest, align_folds=True)
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
