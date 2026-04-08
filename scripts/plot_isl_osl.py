#!/usr/bin/env python3
"""
plot_isl_osl.py — Visualize ISL/OSL sweep benchmark results.

Generates a throughput vs. TTFT latency curve in Jensen/GTC 2026 style:
- Dark background, NVIDIA green accent palette
- One curve per ISL/OSL combo
- Points along each curve = increasing concurrency levels
- X-axis: output throughput (tokens/sec)
- Y-axis: TTFT p50 (ms)

Usage:
  python scripts/plot_isl_osl.py results/gemma4-26b-isl-osl-2026-04-07.json
  python scripts/plot_isl_osl.py results/*.json --out results/custom.png
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Style constants ───────────────────────────────────────────────────────────

BG_COLOR      = "#0a0a0a"
PANEL_COLOR   = "#111111"
GRID_COLOR    = "#1e1e1e"
TEXT_COLOR    = "#e0e0e0"
ACCENT_COLOR  = "#76b900"   # NVIDIA green

# Curve palette — NVIDIA green + complementary hues for multi-line readability
CURVE_COLORS = [
    "#76b900",  # NVIDIA green      — short/short
    "#00b4d8",  # cyan               — short/long
    "#f77f00",  # amber              — long/short
    "#e63946",  # red                — long/long
    "#c77dff",  # violet             — very-long
]

LABEL_MAP = {
    "short-short": "ISL=128 / OSL=128",
    "short-long":  "ISL=128 / OSL=512",
    "long-short":  "ISL=1024 / OSL=128",
    "long-long":   "ISL=1024 / OSL=512",
    "very-long":   "ISL=4096 / OSL=512",
}

MARKER_STYLE = dict(marker="o", markersize=6, linewidth=2, alpha=0.92)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(paths: list[Path]) -> dict:
    """Merge one or more result JSON files. Last file wins on duplicate keys."""
    merged: dict[str, dict] = {}
    meta = {}
    for path in paths:
        data = json.loads(path.read_text())
        meta = data  # keep last for header info
        for row in data["results"]:
            key = (row["label"], row["concurrency"])
            merged[key] = row
    return meta, merged


def group_by_label(merged: dict) -> dict[str, list[dict]]:
    """Group rows by ISL/OSL label, sorted by concurrency."""
    groups: dict[str, list[dict]] = {}
    for row in merged.values():
        groups.setdefault(row["label"], []).append(row)
    for label in groups:
        groups[label].sort(key=lambda r: r["concurrency"])
    return groups


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(meta: dict, groups: dict[str, list[dict]], out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # Grid
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # Tick styling
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT_COLOR)

    # Plot each combo
    ordered_labels = [l for l, _, _ in [
        ("short-short",  128,  128),
        ("short-long",   128,  512),
        ("long-short",  1024,  128),
        ("long-long",   1024,  512),
        ("very-long",   4096,  512),
    ] if l in groups]

    for idx, label in enumerate(ordered_labels):
        rows = groups[label]
        x = [r["throughput_tok_s"] for r in rows]
        y = [r["ttft_p50_ms"]      for r in rows]
        concs = [r["concurrency"]  for r in rows]
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        display = LABEL_MAP.get(label, label)

        ax.plot(x, y, color=color, label=display, **MARKER_STYLE)

        # Annotate every other concurrency point to avoid clutter
        for i, (xi, yi, c) in enumerate(zip(x, y, concs)):
            if i == 0 or i == len(x) - 1 or c in (8, 32):
                ax.annotate(
                    f"c={c}",
                    xy=(xi, yi),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=color,
                    alpha=0.85,
                )

    # Axis labels
    ax.set_xlabel("Output Throughput (tokens/sec)", color=TEXT_COLOR, fontsize=12, labelpad=8)
    ax.set_ylabel("TTFT p50 (ms)", color=TEXT_COLOR, fontsize=12, labelpad=8)

    # Format axes with comma thousands separator
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Title
    model = meta.get("model", "gemma-4-26B-A4B")
    ts    = meta.get("timestamp", "")[:10]
    ax.set_title(
        f"{model}  |  ISL/OSL Throughput–Latency Sweep  |  {ts}",
        color=TEXT_COLOR,
        fontsize=13,
        pad=14,
        fontweight="bold",
    )

    # Subtitle annotation
    ax.annotate(
        "DGX Spark PP=2  ·  GB10 sm_121  ·  vLLM  ·  each point = one concurrency level",
        xy=(0.5, 1.025),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="#888888",
    )

    # Legend
    leg = ax.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.3,
        facecolor=BG_COLOR,
        edgecolor="#333333",
        labelcolor=TEXT_COLOR,
    )

    # NVIDIA green accent bar along top edge
    fig.add_artist(
        plt.Line2D(
            [0.0, 1.0], [1.0, 1.0],
            transform=fig.transFigure,
            color=ACCENT_COLOR,
            linewidth=3,
        )
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"[plot] Saved to {out_path}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot ISL/OSL sweep results (throughput vs TTFT)"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to benchmark_isl_osl.py JSON output file(s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: results/gemma4-26b-isl-osl-<date>.png)",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            sys.exit(f"ERROR: file not found: {p}")

    meta, merged = load_results(input_paths)
    if not merged:
        sys.exit("ERROR: no results found in input files")

    groups = group_by_label(merged)
    print(f"[plot] Loaded {len(merged)} data points across {len(groups)} combos", flush=True)

    if args.out:
        out_path = Path(args.out)
    else:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        # Infer repo root from script location
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "results"
        out_path = results_dir / f"gemma4-26b-isl-osl-{date_str}.png"

    make_plot(meta, groups, out_path)
    print(f"[plot] Done. View: {out_path}", flush=True)


if __name__ == "__main__":
    main()
