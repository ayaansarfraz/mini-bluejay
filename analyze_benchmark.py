"""Turn benchmark_results.json into a comparison table, the headline answers, and a chart.

Uses the latest sweep for each label (see NAMES) and takes the median across repeats at
each concurrency level. Writes two charts: API vs self-hosted, and quantization before/after.

  uv run python analyze_benchmark.py
"""

import json
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from benchmark import GPU_PER_HOUR, HAIKU_IN_PER_TOK, HAIKU_OUT_PER_TOK, RESULTS_PATH  # noqa: E402

VOICE_TTFT_BUDGET_S = 1.0  # a caller should start "speaking" within ~1s
CHART_PATH = "benchmark_chart.png"
QUANT_CHART_PATH = "quantization_chart.png"

NAMES = {
    "haiku": "Claude Haiku 4.5 (API)",
    "qwen3-8b-a10g": "Qwen3-8B 16-bit on 1x A10G",
    "qwen3-8b-awq-a10g": "Qwen3-8B 4-bit AWQ on 1x A10G",
    "qwen3-8b-awq-kvfp8-a10g": "Qwen3-8B 4-bit AWQ + 8-bit KV cache on 1x A10G",
    "qwen3-8b-l40s": "Qwen3-8B 16-bit on 1x L40S",
    "qwen3-8b-fp8-l40s": "Qwen3-8B 8-bit FP8 on 1x L40S",
}
# Validated categorical palette, slots 1-3, assigned in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# (label, short line label, color, marker)
MODELS_CHART = [
    ("haiku", "Claude Haiku 4.5", BLUE, "o"),
    ("qwen3-8b-a10g", "Qwen A10G", ORANGE, "s"),
    ("qwen3-8b-l40s", "Qwen L40S", AQUA, "^"),
]
QUANT_CHART = [  # one row per GPU; blue = original 16-bit weights ("before")
    ("A10G (24GB, $1.10/hr)", [
        ("qwen3-8b-a10g", "16-bit", BLUE, "o"),
        ("qwen3-8b-awq-a10g", "4-bit AWQ", ORANGE, "s"),
        ("qwen3-8b-awq-kvfp8-a10g", "4-bit + fp8 KV", AQUA, "^"),
    ]),
    ("L40S (48GB, $1.95/hr)", [
        ("qwen3-8b-l40s", "16-bit", BLUE, "o"),
        ("qwen3-8b-fp8-l40s", "8-bit FP8", ORANGE, "s"),
    ]),
]
PANELS = [
    ("throughput_tok_s", "Throughput (output tokens/s)", False),
    ("ttft_p95", "Time to first token, p95 (s)", True),
    ("cost_per_1k_calls", "Cost per 1,000 simulated calls ($)", True),
]
METRICS = [
    "throughput_tok_s", "requests_per_s", "ttft_p50", "ttft_p95", "latency_p50", "latency_p95",
    "tok_s_per_request", "cost_per_1k_calls", "peak_running", "peak_waiting", "peak_kv_cache",
    "avg_in_tokens", "avg_out_tokens", "caller_turns_per_call",
]


def load() -> dict[str, list[dict]]:
    """label -> one aggregated row per concurrency level (medians across repeats)."""
    with open(RESULTS_PATH) as f:
        records = json.load(f)
    latest = {}
    for rec in records:
        if rec.get("label") in NAMES:
            latest[rec["label"]] = rec  # later records win

    out = {}
    for label, rec in latest.items():
        by_c = defaultdict(list)
        for row in rec["levels"]:
            by_c[row["concurrency"]].append(row)
        rows = []
        for c, reps in sorted(by_c.items()):
            agg = {"concurrency": c, "repeats": len(reps),
                   "errors": sum(r["errors"] for r in reps),
                   "requests": sum(r["requests"] for r in reps)}
            for m in METRICS:
                vals = [r[m] for r in reps if r.get(m) is not None]
                agg[m] = statistics.median(vals) if vals else None
            tput = [r["throughput_tok_s"] for r in reps if r.get("throughput_tok_s")]
            agg["throughput_spread"] = (max(tput) - min(tput)) / statistics.median(tput) if len(tput) > 1 else 0
            rows.append(agg)
        out[label] = rows
    return out


def headline(label: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r["throughput_tok_s"]]
    peak = max(ok, key=lambda r: r["throughput_tok_s"])
    knee = next(r for r in ok if r["throughput_tok_s"] >= 0.9 * peak["throughput_tok_s"])
    in_budget = [r for r in ok if r["ttft_p95"] <= VOICE_TTFT_BUDGET_S]
    best_in_budget = max(in_budget, key=lambda r: r["throughput_tok_s"]) if in_budget else None
    return {"peak": peak, "knee": knee, "best_in_budget": best_in_budget}


def print_tables(data: dict[str, list[dict]]) -> None:
    for label, rows in data.items():
        print(f"\n=== {NAMES[label]} (median of {rows[0]['repeats']} repeat(s)) ===")
        print(f"{'conc':>5}{'thru tok/s':>12}{'req/s':>8}{'ttft p50':>10}{'ttft p95':>10}"
              f"{'lat p95':>9}{'tok/s/req':>11}{'$/1k calls':>12}{'errors':>8}"
              f"{'running':>9}{'waiting':>9}{'kv':>6}{'spread':>8}")
        for r in rows:
            if r["throughput_tok_s"] is None:
                print(f"{r['concurrency']:>5}  all failed ({r['errors']} errors)")
                continue
            server = (f"{r['peak_running']:>9.0f}{r['peak_waiting']:>9.0f}{r['peak_kv_cache']:>6.0%}"
                      if r["peak_running"] is not None else f"{'':>24}")
            print(
                f"{r['concurrency']:>5}{r['throughput_tok_s']:>12.0f}{r['requests_per_s']:>8.1f}"
                f"{r['ttft_p50']:>10.2f}{r['ttft_p95']:>10.2f}{r['latency_p95']:>9.2f}"
                f"{r['tok_s_per_request']:>11.1f}{r['cost_per_1k_calls']:>12.2f}"
                f"{r['errors']:>8}{server}{r['throughput_spread']:>8.0%}"
            )


def print_answers(data: dict[str, list[dict]]) -> None:
    print("\n=== Headline answers ===")
    for label, rows in data.items():
        h = headline(label, rows)
        name = NAMES[label]
        p, k, b = h["peak"], h["knee"], h["best_in_budget"]
        print(f"\n{name}")
        print(f"  peak throughput: {p['throughput_tok_s']:.0f} tok/s ({p['requests_per_s']:.1f} req/s)"
              f" at concurrency {p['concurrency']}")
        print(f"  saturates (90% of peak) at concurrency {k['concurrency']}")
        if b:
            print(f"  best within {VOICE_TTFT_BUDGET_S:.0f}s TTFT p95: concurrency {b['concurrency']},"
                  f" {b['requests_per_s']:.1f} req/s, ${b['cost_per_1k_calls']:.2f} per 1k calls")
        else:
            print(f"  never meets {VOICE_TTFT_BUDGET_S:.0f}s TTFT p95")

    # Break-even: how busy must a GPU be to beat the API on cost?
    if "haiku" in data:
        h_rows = [r for r in data["haiku"] if r["avg_in_tokens"]]
        api_per_req = statistics.median(
            r["avg_in_tokens"] * HAIKU_IN_PER_TOK + r["avg_out_tokens"] * HAIKU_OUT_PER_TOK
            for r in h_rows
        )
        print(f"\nHaiku costs ~${api_per_req * 1000:.2f} per 1,000 caller turns.")
        for label in data:
            if label == "haiku":
                continue
            gpu = label.rsplit("-", 1)[1].upper()
            per_s = GPU_PER_HOUR[gpu] / 3600
            breakeven_rps = per_s / api_per_req
            peak_rps = headline(label, data[label])["peak"]["requests_per_s"]
            print(f"  {NAMES[label]}: cheaper than Haiku above {breakeven_rps:.2f} req/s sustained"
                  f" = {breakeven_rps / peak_rps:.0%} of its peak capacity"
                  f" (~{breakeven_rps * 3600:,.0f} caller turns/hour)")


def style_axis(ax, metric: str, title: str, logy: bool, levels: list[int], muted: str,
               grid: str) -> None:
    if metric == "ttft_p95":
        ax.axhline(VOICE_TTFT_BUDGET_S, color=muted, linewidth=1, linestyle="--")
        ax.annotate("1s voice budget", (levels[0], VOICE_TTFT_BUDGET_S), xytext=(0, 4),
                    textcoords="offset points", fontsize=8.5, color=muted)
    ax.set_xscale("log")
    ticks = [c for c in levels if c in (1, 5, 10, 20, 50, 100, 200)] or levels
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(c) for c in ticks], fontsize=8.5)
    ax.xaxis.set_minor_locator(NullLocator())
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        if metric == "ttft_p95":
            ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xlabel("Concurrent requests")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(levels[0] * 0.8, levels[-1] * 2.4)  # room for end labels


def chart(data: dict[str, list[dict]], rows: list[tuple[str, list]], title: str, path: str) -> None:
    """One row of three panels per (row title, series list); series missing from data are skipped."""
    rows = [(t, [s for s in series if s[0] in data]) for t, series in rows]
    rows = [(t, series) for t, series in rows if series]
    if not rows:
        return
    text, muted, grid, bg = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    plt.rcParams.update({
        "text.parse_math": False,  # "$" in titles is money, not LaTeX
        "font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted,
        "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": text,
    })
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.6 * len(rows)), facecolor=bg,
                             squeeze=False)
    levels = sorted({r["concurrency"] for v in data.values() for r in v})
    for row_axes, (row_title, series) in zip(axes, rows):
        for ax, (metric, panel_title, logy) in zip(row_axes, PANELS):
            ax.set_facecolor(bg)
            ends = []
            for label, short, color, marker in series:
                pts = [(r["concurrency"], r[metric]) for r in data[label] if r[metric] is not None]
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6,
                        markeredgecolor=bg, markeredgewidth=1.5, label=NAMES[label])
                ends.append((xs[-1], ys[-1], short))
            style_axis(ax, metric, "", logy, levels, muted, grid)  # fix scales before placing labels
            fig.canvas.draw()
            placed: list[float] = []  # label y positions in screen points, nudged apart
            for x, y, short in sorted(ends, key=lambda e: e[1]):
                y_pt = ax.transData.transform((x, y))[1] * 72 / fig.dpi
                target = max([y_pt] + [p + 11 for p in placed])
                placed.append(target)
                ax.annotate(short, (x, y), xytext=(6, target - y_pt), textcoords="offset points",
                            va="center", fontsize=8.5, color=muted)
            full_title = f"{row_title}: {panel_title}" if len(rows) > 1 else panel_title
            style_axis(ax, metric, full_title, logy, levels, muted, grid)
        row_axes[0].legend(frameon=False, fontsize=8.5, labelcolor=text, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=text)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    print(f"chart saved to {path}")


def print_quant_comparison(data: dict[str, list[dict]]) -> None:
    """Before/after at the levels that matter: in the voice budget, and flat out."""
    print("\n=== Quantization before/after ===")
    for row_title, series in QUANT_CHART:
        present = [s for s in series if s[0] in data]
        if len(present) < 2:
            continue
        print(f"\n{row_title}")
        print(f"{'':<18}{'peak tok/s':>12}{'peak req/s':>12}{'TTFT p95@50':>13}"
              f"{'max conc <1s':>14}{'$/1k @ that':>13}{'peak KV':>9}{'tok/s/req@1':>13}")
        for label, short, *_ in present:
            rows = data[label]
            h = headline(label, rows)
            at50 = next((r for r in rows if r["concurrency"] == 50), None)
            first = rows[0]
            b = h["best_in_budget"]
            kv = max((r["peak_kv_cache"] or 0) for r in rows)
            print(f"{short:<18}{h['peak']['throughput_tok_s']:>12.0f}"
                  f"{h['peak']['requests_per_s']:>12.1f}"
                  f"{(at50['ttft_p95'] if at50 else float('nan')):>13.2f}"
                  f"{(b['concurrency'] if b else 0):>14}"
                  f"{(b['cost_per_1k_calls'] if b else float('nan')):>13.2f}"
                  f"{kv:>9.0%}{first['tok_s_per_request']:>13.1f}")


def main() -> None:
    data = load()
    if not data:
        raise SystemExit("no labelled sweeps in benchmark_results.json; run benchmark.py first")
    print_tables(data)
    print_answers(data)
    print_quant_comparison(data)
    print()
    chart(data, [("", MODELS_CHART)],
          "Persona model: hosted API vs self-hosted Qwen3-8B (vLLM, 1 GPU)", CHART_PATH)
    chart(data, QUANT_CHART, "Quantization: same model, smaller numbers (vLLM, 1 GPU)",
          QUANT_CHART_PATH)


if __name__ == "__main__":
    main()
