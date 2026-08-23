"""Chart rendering for the per-ticker study.

Colours come from a validated categorical palette: slot 1 blue and slot 2
orange for identity, the blue-red diverging pair for long/short polarity, and
recessive greys for chrome.  The three-slot categorical set clears the
all-pairs CVD (worst 9.2) and normal-vision (worst 24.0) floors on the light
surface, so the charts stay readable under colour-vision deficiency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --------------------------------------------------------------------- palette --
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_1 = "#2a78d6"  # blue  - the strategy
SERIES_2 = "#eb6834"  # orange - the buy & hold reference
POLARITY_LONG = "#2a78d6"  # diverging pair, cool pole
POLARITY_SHORT = "#e34948"  # diverging pair, warm pole
RECESSIVE = "#c3c2b7"


def _style_axes(ax, *, grid_axis: str = "y") -> None:
    """Hairline, recessive chrome: solid gridlines one shade off the surface."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)


def _polarity_bands(ax, index: pd.DatetimeIndex, position: pd.Series) -> list[str]:
    """Shade the periods the strategy was long or short.

    Returns the polarities actually drawn, so the legend only names states the
    reader can see - a "short" swatch beside a long-only strategy is noise.
    """
    if position.empty:
        return []
    sign = np.sign(position.reindex(index).fillna(0.0).to_numpy())
    drawn: list[str] = []
    for value, colour, label in ((1.0, POLARITY_LONG, "long"), (-1.0, POLARITY_SHORT, "short")):
        mask = sign == value
        if not mask.any():
            continue
        ax.fill_between(
            index, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
            color=colour, alpha=0.12, linewidth=0, zorder=1,
        )
        drawn.append(label)
    return drawn


def _format_params(params: dict, per_line: int = 3) -> str:
    items = [f"{k} = {v}" for k, v in params.items()]
    lines = [
        "    ".join(items[i : i + per_line]) for i in range(0, len(items), per_line)
    ]
    return "\n".join(lines)


def ticker_chart(study, output_path: Path) -> Path:
    """One research card per ticker: price, position, equity, ranking, parameters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(14.0, 11.0), facecolor=SURFACE)
    grid = fig.add_gridspec(
        3, 2, height_ratios=[3.0, 2.0, 1.15], width_ratios=[1.30, 1.0],
        hspace=0.42, wspace=0.42, left=0.065, right=0.975, top=0.885, bottom=0.055,
    )
    ax_price = fig.add_subplot(grid[0, :])
    ax_equity = fig.add_subplot(grid[1, 0])
    ax_rank = fig.add_subplot(grid[1, 1])
    ax_text = fig.add_subplot(grid[2, :])

    best = study.best
    strategy_label = (
        f"{best.section} {best.title}" if best is not None else "no applicable strategy"
    )
    fig.suptitle(
        f"{study.ticker}  —  best out-of-sample strategy: {strategy_label}",
        x=0.065, ha="left", fontsize=17, fontweight="bold", color=INK_PRIMARY, y=0.965,
    )
    subtitle = (
        f"Walk-forward: {study.train_days} training days → {study.test_days} test days, "
        f"stepped forward {study.folds} times   |   out-of-sample "
        f"{study.oos_start} to {study.oos_end}   |   {study.cost_bps:.0f} bps costs"
    )
    fig.text(0.065, 0.925, subtitle, ha="left", fontsize=10, color=INK_SECONDARY)

    # ------------------------------------------------------ price + position --
    close = study.close
    oos = study.oos_index
    _style_axes(ax_price)

    labels: list[str] = []
    if len(oos):
        split = close.index.get_indexer([oos[0]], method="nearest")[0]
        # In-sample history is context, so it recedes; the traded period is inked.
        ax_price.plot(close.index[: split + 1], close.to_numpy()[: split + 1],
                      color=INK_MUTED, linewidth=1.1, zorder=3)
        ax_price.plot(close.index[split:], close.to_numpy()[split:],
                      color=INK_PRIMARY, linewidth=1.3, zorder=3)
        ax_price.axvline(oos[0], color=AXIS, linewidth=1.0, zorder=2)
        ax_price.annotate(
            f"out-of-sample from {study.oos_start}", xy=(oos[0], 0.96),
            xycoords=("data", "axes fraction"), fontsize=9, color=INK_MUTED,
            ha="left", va="top", xytext=(7, 0), textcoords="offset points",
        )
        if best is not None:
            labels = _polarity_bands(ax_price, oos, study.best_position())
    else:
        ax_price.plot(close.index, close.to_numpy(), color=INK_PRIMARY, linewidth=1.3, zorder=3)

    ax_price.set_yscale("log")
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_price.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_price.set_ylabel(f"{study.ticker} close (log, adjusted)", fontsize=10, color=INK_SECONDARY)

    if labels:
        colours = {"long": POLARITY_LONG, "short": POLARITY_SHORT}
        legend = ax_price.legend(
            handles=[Patch(facecolor=colours[k], alpha=0.42, label=f"strategy {k}") for k in labels],
            loc="upper left", frameon=False, fontsize=9, ncol=len(labels),
            bbox_to_anchor=(0.0, -0.09),
        )
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    # ------------------------------------------------------------- equity --
    _style_axes(ax_equity)
    if best is not None:
        curves = {
            f"{best.section} strategy": (study.best_equity(), SERIES_1),
            "buy & hold": (study.benchmark_equity(), SERIES_2),
        }
        for label, (curve, colour) in curves.items():
            ax_equity.plot(
                curve.index, curve.to_numpy(), color=colour, linewidth=2.0, zorder=3,
                label=f"{label}  —  {curve.iloc[-1]:.1f}x",
            )
            # A ringed endpoint marker anchors the legend entry to its line
            # without a floating label that can collide with the next panel.
            ax_equity.plot(
                [curve.index[-1]], [curve.iloc[-1]], marker="o", markersize=7,
                color=colour, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
            )
        equity_legend = ax_equity.legend(
            loc="upper left", frameon=False, fontsize=9.5, handlelength=1.6,
            borderpad=0.0, labelspacing=0.35,
        )
        for text in equity_legend.get_texts():
            text.set_color(INK_SECONDARY)
        ax_equity.set_yscale("log")
        ax_equity.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax_equity.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax_equity.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
        index = study.best_equity().index
        span = index[-1] - index[0]
        ax_equity.set_xlim(index[0], index[-1] + span * 0.04)  # breathing room past the marker
    ax_equity.set_ylabel("growth of 1 (log)", fontsize=10, color=INK_SECONDARY)
    ax_equity.set_title(
        "Out-of-sample equity, net of costs", fontsize=11, color=INK_PRIMARY,
        loc="left", pad=8, fontweight="bold",
    )

    # ------------------------------------------------------------ ranking --
    _style_axes(ax_rank, grid_axis="x")
    ranked = study.ranked.head(8).iloc[::-1]
    if not ranked.empty:
        colours = [
            SERIES_1 if key == (best.key if best else None) else RECESSIVE
            for key in ranked["key"]
        ]
        positions = np.arange(len(ranked))
        ax_rank.barh(positions, ranked["sharpe"].to_numpy(), color=colours, height=0.62, zorder=2)
        ax_rank.set_yticks(positions)
        ax_rank.set_yticklabels(
            [f"{s}  {t if len(t) <= 30 else t[:29] + chr(0x2026)}"
             for s, t in zip(ranked["section"], ranked["title"])],
            fontsize=8.5,
        )
        for tick in ax_rank.get_yticklabels():
            tick.set_color(INK_SECONDARY)
        ax_rank.axvline(0.0, color=AXIS, linewidth=0.8, zorder=1)
        for pos, value in zip(positions, ranked["sharpe"].to_numpy()):
            offset = 4 if value >= 0 else -4
            ax_rank.annotate(
                f"{value:.2f}", xy=(value, pos), xytext=(offset, 0), textcoords="offset points",
                va="center", ha="left" if value >= 0 else "right", fontsize=8.5,
                color=INK_SECONDARY,
            )
        ax_rank.margins(x=0.18)
    ax_rank.set_xlabel("out-of-sample Sharpe", fontsize=10, color=INK_SECONDARY)
    ax_rank.set_title(
        f"Applicable strategies ranked ({study.applicable} of {study.tested} tested)",
        fontsize=11, color=INK_PRIMARY, loc="left", pad=8, fontweight="bold",
    )

    # --------------------------------------------------------- parameters --
    ax_text.axis("off")
    ax_text.set_facecolor(SURFACE)
    if best is not None:
        ax_text.text(
            0.0, 0.92, "Parameters selected in-sample", fontsize=11, fontweight="bold",
            color=INK_PRIMARY, va="top", transform=ax_text.transAxes,
        )
        ax_text.text(
            0.0, 0.70, _format_params(study.best_params), fontsize=10, color=INK_SECONDARY,
            va="top", family="monospace", transform=ax_text.transAxes, linespacing=1.7,
        )
        ax_text.text(
            0.0, 0.06,
            f"most frequent choice across {study.folds} training windows; "
            f"per-fold detail in {best.key}.csv",
            fontsize=8.5, color=INK_MUTED, va="bottom", transform=ax_text.transAxes,
        )
        ax_text.text(
            0.60, 0.92, "Out-of-sample performance", fontsize=11, fontweight="bold",
            color=INK_PRIMARY, va="top", transform=ax_text.transAxes,
        )
        rows = [
            ("", "strategy", "buy & hold"),
            ("annualised return", f"{study.best_stats['cagr'] * 100:>8.1f}%",
             f"{study.benchmark_stats['cagr'] * 100:>8.1f}%"),
            ("Sharpe", f"{study.best_stats['sharpe']:>9.2f}",
             f"{study.benchmark_stats['sharpe']:>9.2f}"),
            ("max drawdown", f"{study.best_stats['max_drawdown'] * 100:>8.1f}%",
             f"{study.benchmark_stats['max_drawdown'] * 100:>8.1f}%"),
            ("Calmar", f"{study.best_stats['calmar']:>9.2f}",
             f"{study.benchmark_stats['calmar']:>9.2f}"),
        ]
        for i, (label, strategy_value, bench_value) in enumerate(rows):
            y = 0.70 - i * 0.165
            weight = "bold" if i == 0 else "normal"
            colour = INK_PRIMARY if i == 0 else INK_SECONDARY
            ax_text.text(0.60, y, label, fontsize=9.5, color=INK_SECONDARY,
                         va="top", transform=ax_text.transAxes)
            ax_text.text(0.80, y, strategy_value, fontsize=9.5, color=colour, va="top",
                         ha="right", family="monospace", fontweight=weight,
                         transform=ax_text.transAxes)
            ax_text.text(0.95, y, bench_value, fontsize=9.5, color=colour, va="top",
                         ha="right", family="monospace", fontweight=weight,
                         transform=ax_text.transAxes)
    else:
        ax_text.text(
            0.0, 0.8, "No strategy in the library takes a position on a single-name universe.",
            fontsize=11, color=INK_SECONDARY, va="top", transform=ax_text.transAxes,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def summary_chart(winners: pd.DataFrame, output_path: Path, subtitle: str = "") -> Path:
    """Best strategy per ticker: one bar each, with the buy & hold marker.

    Takes the winners table rather than the study objects so the chart can be
    re-rendered from ``best_per_ticker.csv`` without re-running the backtest.
    Values and strategy names sit in fixed columns to the right of the plot;
    anchoring them to the bar ends would collide with the benchmark markers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = winners.dropna(subset=["sharpe"]).sort_values("sharpe").reset_index(drop=True)
    if ranked.empty:
        return output_path

    n = len(ranked)
    fig, ax = plt.subplots(figsize=(13.0, 0.68 * n + 2.9), facecolor=SURFACE)
    fig.suptitle("Best out-of-sample strategy per ticker", x=0.045, ha="left",
                 fontsize=17, fontweight="bold", color=INK_PRIMARY, y=0.975)
    if subtitle:
        fig.text(0.045, 0.905, subtitle, ha="left", va="top", fontsize=10,
                 color=INK_SECONDARY, wrap=True)

    _style_axes(ax, grid_axis="x")
    positions = np.arange(n)
    sharpes = ranked["sharpe"].to_numpy(dtype=float)
    benchmarks = ranked["buy_hold_sharpe"].to_numpy(dtype=float)

    ax.barh(positions, sharpes, color=SERIES_1, height=0.42, zorder=2, label="best strategy")
    ax.scatter(benchmarks, positions, color=SERIES_2, s=58, zorder=4,
               edgecolor=SURFACE, linewidth=2, label="buy & hold")
    ax.set_yticks(positions)
    ax.set_yticklabels(ranked["ticker"], fontsize=11.5, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_color(INK_PRIMARY)

    limit = float(max(np.nanmax(sharpes), np.nanmax(benchmarks)))
    value_x, name_x = limit * 1.10, limit * 1.24
    ax.set_xlim(min(0.0, float(np.nanmin(sharpes)) * 1.1), limit * 2.05)
    for pos, row in ranked.iterrows():
        ax.text(value_x, pos, f"{row['sharpe']:.2f}", va="center", ha="right",
                fontsize=10, color=INK_PRIMARY, fontweight="bold")
        ax.text(name_x, pos, f"{row['section']} {row['best_strategy']}", va="center",
                ha="left", fontsize=10, color=INK_SECONDARY)
    ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    # Ticks stop at the data; the space to the right belongs to the text columns.
    low = min(0.0, float(np.nanmin(sharpes)))
    step = 0.5 if limit <= 3 else 1.0
    ax.set_xticks(np.arange(np.floor(low / step) * step, limit + step * 0.5, step))
    ax.set_xlabel("out-of-sample Sharpe ratio", fontsize=10, color=INK_SECONDARY)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.16 - 0.02 * n),
                       frameon=False, fontsize=9.5, ncol=2)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.87))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def significance_chart(
    significance: pd.DataFrame,
    output_path: Path,
    alpha: float = 0.05,
    subtitle: str = "",
) -> Path:
    """Effect size with its uncertainty, per ticker.

    A p-value alone says "significant or not"; what a reader needs is how large
    the edge is and how wide the error bar around it. Each row is the winning
    strategy's annualised return in excess of that ticker's own buy & hold, with
    a Newey-West 95% interval, so an interval straddling zero is visible as
    such. The selection-corrected p-value is printed alongside.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = significance.dropna(subset=["excess_ann_return"]).copy()
    if frame.empty:
        return output_path
    frame = frame.sort_values("excess_ann_return").reset_index(drop=True)

    # The t-statistic already embeds the Newey-West standard error.
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.abs(frame["excess_ann_return"] / frame["t_stat_vs_buy_hold"])
    half = 1.96 * se

    n = len(frame)
    fig, ax = plt.subplots(figsize=(13.0, 0.72 * n + 3.1), facecolor=SURFACE)
    fig.suptitle("Is the edge real, or the best of many tries?", x=0.045, ha="left",
                 fontsize=17, fontweight="bold", color=INK_PRIMARY, y=0.975)
    if subtitle:
        fig.text(0.045, 0.90, subtitle, ha="left", va="top", fontsize=10, color=INK_SECONDARY)

    _style_axes(ax, grid_axis="x")
    positions = np.arange(n)
    centre = frame["excess_ann_return"].to_numpy() * 100
    ax.errorbar(
        centre, positions, xerr=half.to_numpy() * 100, fmt="o", markersize=8,
        color=SERIES_1, ecolor=RECESSIVE, elinewidth=2.0, capsize=0, zorder=3,
        markeredgecolor=SURFACE, markeredgewidth=2,
        label="excess annualised return, 95% Newey-West interval",
    )
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.2, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(frame["ticker"], fontsize=11.5, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_color(INK_PRIMARY)

    span = float(np.nanmax(np.abs(centre) + half.to_numpy() * 100))
    ax.set_xlim(-span * 1.25, span * 3.1)
    label_x = span * 1.45
    ax.text(label_x, n - 0.35, "selection-corrected p", fontsize=9, color=INK_MUTED,
            ha="left", va="bottom", fontweight="bold")
    for pos, row in frame.iterrows():
        p_value = row["spa_p"]
        colour = POLARITY_SHORT if p_value < alpha else INK_SECONDARY
        ax.text(label_x, pos, f"p = {p_value:.2f}", fontsize=10, color=colour,
                va="center", ha="left", fontweight="bold" if p_value < alpha else "normal")
        ax.text(label_x + span * 0.55, pos, row["verdict"], fontsize=9.5,
                color=INK_SECONDARY, va="center", ha="left")
    ax.set_xlabel("annualised return in excess of buy & hold (%)", fontsize=10,
                  color=INK_SECONDARY)
    step = 5.0 if span <= 25 else 10.0
    ax.set_xticks(np.arange(-np.ceil(span / step) * step, span + step * 0.5, step))
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.17 - 0.02 * n),
                       frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path
