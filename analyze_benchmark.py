"""Turn benchmark_results.json into a comparison table, the headline answers, and a chart.

Uses the latest sweep for each label (haiku, qwen3-8b-a10g, qwen3-8b-l40s) and takes the
median across repeats at each concurrency level.

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

SERIES = {  # fixed order and colors (validated categorical palette, slots 1-3)
    "haiku": ("Claude Haiku 4.5 (API)", "#2a78d6", "o"),
    "qwen3-8b-a10g": ("Qwen3-8B on 1x A10G", "#eb6834", "s"),
    "qwen3-8b-l40s": ("Qwen3-8B on 1x L40S", "#1baf7a", "^"),
}
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
        if rec.get("label") in SERIES:
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
        print(f"\n=== {SERIES[label][0]} (median of {rows[0]['repeats']} repeat(s)) ===")
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
        name = SERIES[label][0]
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
        for label in ("qwen3-8b-a10g", "qwen3-8b-l40s"):
            if label not in data:
                continue
            gpu = label.rsplit("-", 1)[1].upper()
            per_s = GPU_PER_HOUR[gpu] / 3600
            breakeven_rps = per_s / api_per_req
            peak_rps = headline(label, data[label])["peak"]["requests_per_s"]
            print(f"  {gpu}: cheaper than Haiku above {breakeven_rps:.2f} req/s sustained"
                  f" = {breakeven_rps / peak_rps:.0%} of its peak capacity"
                  f" (~{breakeven_rps * 3600:,.0f} caller turns/hour)")


def chart(data: dict[str, list[dict]]) -> None:
    text, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted,
        "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": text,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="#fcfcfb")
    panels = [
        ("throughput_tok_s", "Throughput (output tokens/s)", False),
        ("ttft_p95", "Time to first token, p95 (s)", True),
        ("cost_per_1k_calls", "Cost per 1,000 simulated calls ($)", True),
    ]
    levels = sorted({r["concurrency"] for rows in data.values() for r in rows})
    for ax, (metric, title, logy) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for label in SERIES:
            if label not in data:
                continue
            name, color, marker = SERIES[label]
            pts = [(r["concurrency"], r[metric]) for r in data[label] if r[metric] is not None]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
            ax.annotate(name.split(" (")[0].replace("Qwen3-8B on 1x ", "Qwen "), (xs[-1], ys[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=muted)
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
                ax.set_yticks([0.2, 0.5, 1, 2, 5])
        ax.set_xlabel("Concurrent requests")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(levels[0] * 0.8, levels[-1] * 1.9)  # room for end labels
    axes[0].legend(frameon=False, fontsize=9, labelcolor=text, loc="upper left")
    fig.suptitle("Persona model: hosted API vs self-hosted Qwen3-8B (vLLM, 1 GPU)",
                 x=0.01, ha="left", fontsize=13, color=text)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160, facecolor=fig.get_facecolor())
    print(f"\nchart saved to {CHART_PATH}")


def main() -> None:
    data = load()
    if not data:
        raise SystemExit("no labelled sweeps in benchmark_results.json; run benchmark.py first")
    print_tables(data)
    print_answers(data)
    chart(data)


if __name__ == "__main__":
    main()
