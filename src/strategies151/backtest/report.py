"""Persist and render walk-forward results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from strategies151.backtest.engine import WalkForwardResult

LEADERBOARD_COLUMNS = [
    "section",
    "key",
    "title",
    "style",
    "folds",
    "days",
    # Headline four, as fractions: annualised return, Sharpe, drawdown, Calmar.
    "cagr",
    "sharpe",
    "max_drawdown",
    "calmar",
    "ann_return",
    "ann_volatility",
    "sortino",
    "hit_rate",
    "ann_turnover",
    "gross_sharpe",
    "cost_drag_ann",
    "avg_gross_exposure",
    "avg_net_exposure",
]


def leaderboard(results: Sequence[WalkForwardResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {"section": res.section, "key": res.key, "title": res.title, "style": res.style}
        row.update(res.stats)
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = [c for c in LEADERBOARD_COLUMNS if c in frame.columns]
    frame = frame.loc[:, columns + [c for c in frame.columns if c not in columns]]
    return frame.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)


def daily_matrix(results: Sequence[WalkForwardResult]) -> pd.DataFrame:
    """One column of net daily returns per strategy, aligned on date."""
    return pd.DataFrame({res.key: res.net_returns for res in results}).sort_index()


def write_results(
    results: Sequence[WalkForwardResult],
    output_dir: Path,
    context: dict | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    board = leaderboard(results)
    paths["leaderboard"] = output_dir / "leaderboard.csv"
    board.to_csv(paths["leaderboard"], index=False)

    returns = daily_matrix(results)
    paths["daily_returns"] = output_dir / "daily_returns.csv"
    returns.to_csv(paths["daily_returns"])

    equity = (1.0 + returns.fillna(0.0)).cumprod()
    paths["equity_curves"] = output_dir / "equity_curves.csv"
    equity.to_csv(paths["equity_curves"])

    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    for res in results:
        if res.folds.empty:
            continue
        frame = res.folds.copy()
        if "params" in frame:
            frame["params"] = frame["params"].apply(json.dumps, default=str)
        for column in ("fitted_pair", "clusters", "combo_weights"):
            if column in frame:
                frame[column] = frame[column].apply(json.dumps, default=str)
        frame.to_csv(folds_dir / f"{res.key}.csv", index=False)
    paths["folds"] = folds_dir

    if context:
        paths["context"] = output_dir / "run_context.json"
        paths["context"].write_text(json.dumps(context, indent=2, default=str))

    paths["summary"] = output_dir / "summary.md"
    paths["summary"].write_text(markdown_summary(board, context))
    return paths


#: Display name -> (source statistic, formatting). ``pct`` multiplies by 100.
DISPLAY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("section", "section", "raw"),
    ("title", "title", "raw"),
    ("style", "style", "raw"),
    ("folds", "folds", "int"),
    ("days", "days", "int"),
    # The four headline metrics.
    ("ann_return_%", "cagr", "pct"),
    ("sharpe", "sharpe", "ratio"),
    ("max_drawdown_%", "max_drawdown", "pct"),
    ("calmar", "calmar", "ratio"),
    # Supporting detail.
    ("ann_vol_%", "ann_volatility", "pct"),
    ("hit_rate_%", "hit_rate", "pct"),
    ("ann_turnover_x", "ann_turnover", "ratio"),
)


def display_frame(board: pd.DataFrame) -> pd.DataFrame:
    """Rename and rescale the leaderboard for human consumption.

    ``leaderboard.csv`` keeps every statistic as a raw fraction so downstream
    analysis does not have to undo formatting; this is the presentation view,
    with units carried in the column names.
    """
    out = pd.DataFrame(index=board.index)
    for label, source, kind in DISPLAY_COLUMNS:
        if source not in board.columns:
            continue
        column = board[source]
        if kind == "pct":
            out[label] = (column.astype(float) * 100).round(2)
        elif kind == "ratio":
            out[label] = column.astype(float).round(2)
        elif kind == "int":
            out[label] = column.astype(float).round().astype("Int64")
        else:
            out[label] = column
    return out


def markdown_summary(board: pd.DataFrame, context: dict | None = None) -> str:
    display = display_frame(board)
    keep = list(display.columns)
    lines = ["# 151 Strategies - walk-forward out-of-sample results", ""]
    if context:
        lines += [
            f"* Universe: `{', '.join(context.get('tickers', []))}`",
            f"* Bars: `{context.get('table')}` in QuestDB, "
            f"{context.get('bars')} rows, {context.get('first_bar')} to {context.get('last_bar')}",
            f"* Windows: {context.get('train_days')} training days -> "
            f"{context.get('test_days')} test days, stepped by {context.get('step_days')}",
            f"* Out-of-sample span: {context.get('oos_start')} to {context.get('oos_end')} "
            f"({context.get('folds')} folds)",
            f"* Transaction cost: {context.get('cost_bps')} bps per unit traded, "
            f"signals executed with delay {context.get('delay')}",
            "",
            "",
            "`ann_return_%` is the annualised compound return (CAGR), "
            "`max_drawdown_%` the worst peak-to-trough loss, and "
            "`calmar` the ratio of the two. `sharpe` is "
            "`mean/sd * sqrt(252)`, the statistic used in Appendix A of the "
            "paper. All figures are net of costs and entirely out-of-sample: "
            "parameters were chosen on the preceding training window only.",
            "",
        ]
    lines.append(display.loc[:, keep].to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def plot_equity_curves(
    results: Sequence[WalkForwardResult],
    output_path: Path,
    top_n: int = 10,
) -> Path | None:
    """Equity curves for the best strategies plus any benchmark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(
        [r for r in results if r.style != "benchmark"],
        key=lambda r: (r.stats.get("sharpe") if r.stats.get("sharpe") == r.stats.get("sharpe") else -9e9),
        reverse=True,
    )[:top_n]
    benchmarks = [r for r in results if r.style == "benchmark"]
    if not ranked and not benchmarks:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    for res in ranked:
        ax.plot(res.equity_curve(), linewidth=1.2,
                label=f"{res.section} {res.title} (SR {res.stats.get('sharpe', float('nan')):.2f})")
    for res in benchmarks:
        ax.plot(res.equity_curve(), linewidth=2.0, color="black", linestyle="--",
                label=f"{res.title} (SR {res.stats.get('sharpe', float('nan')):.2f})")
    ax.set_yscale("log")
    ax.set_title("Out-of-sample equity curves (1y train / 1m test, walked forward)")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
