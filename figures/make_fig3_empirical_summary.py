"""Regenerate Figure 3: empirical summary panels.

The figure is generated from the checked-in JSON artifacts. The plotting code
keeps the reported numerical values unchanged and only controls layout,
annotation placement, and visual distinguishability.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"


COLORS = {
    "chain": "#0072B2",
    "naive": "#E69F00",
    "naive_dark": "#D55E00",
    "holm": "#009E73",
    "bonf": "#CC79A7",
    "gray": "#6E6E6E",
    "black": "#222222",
}


def load_json(path: str) -> dict:
    with (ROOT / path).open() as f:
        return json.load(f)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 24,
            "axes.labelsize": 21,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 14,
            "axes.linewidth": 1.7,
        }
    )


def panel_a(ax: plt.Axes) -> None:
    rows = load_json("results/e1_validity.json")["rows"]
    alphas = sorted({r["alpha"] for r in rows})
    deltas = [0.05, 0.10]
    offsets = {0.05: -0.0015, 0.10: 0.0015}

    for delta in deltas:
        subset = [r for r in rows if abs(r["delta"] - delta) < 1e-12]
        subset.sort(key=lambda r: r["alpha"])
        x = [r["alpha"] + offsets[delta] for r in subset]
        chain_y = [r["chain"]["violation_rate"] for r in subset]
        naive_y = [r["naive"]["violation_rate"] for r in subset]

        chain_style = "-" if delta == 0.05 else "--"
        naive_style = "-" if delta == 0.05 else (0, (4, 3))
        naive_marker = "s" if delta == 0.05 else "^"
        naive_color = COLORS["naive"] if delta == 0.05 else COLORS["naive_dark"]

        ax.plot(
            x,
            chain_y,
            color=COLORS["chain"],
            linestyle=chain_style,
            marker="o",
            markersize=8,
            linewidth=3.2,
            label=rf"chain $\delta={delta:.2f}$",
            zorder=3,
        )
        ax.plot(
            x,
            naive_y,
            color=naive_color,
            linestyle=naive_style,
            marker=naive_marker,
            markersize=8,
            linewidth=3.2,
            label=rf"naive $\delta={delta:.2f}$",
            zorder=3,
        )

    ax.set_title("A. Synthetic validity", pad=14)
    ax.set_xlabel(r"Risk target $\alpha$", labelpad=10)
    ax.set_ylabel("Violation rate", labelpad=10)
    ax.set_xlim(min(alphas) - 0.008, max(alphas) + 0.022)
    ax.set_ylim(0.0, 0.40)
    ax.set_xticks([0.08, 0.10, 0.12, 0.14, 0.16, 0.18])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.grid(False)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.115),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.0,
    )


def panel_b(ax: plt.Axes) -> None:
    rows = load_json("results/e2_chain_vs_bonferroni.json")["rows"]
    hb_rows = {r["method"]: r for r in rows if r["pvalue"] == "hb"}
    methods = ["chain", "holm", "bonferroni"]
    labels = ["chain", "Holm", "Bonf."]
    costs = [hb_rows[m]["mean_cost"] for m in methods]
    colors = [COLORS["chain"], COLORS["holm"], COLORS["bonf"]]

    e1_rows = load_json("results/e1_validity.json")["rows"]
    naive_cost = next(
        r["naive"]["mean_cost"]
        for r in e1_rows
        if abs(r["alpha"] - 0.1188475) < 1e-12 and abs(r["delta"] - 0.10) < 1e-12
    )

    bars = ax.bar(labels, costs, color=colors, edgecolor=COLORS["black"], linewidth=1.3, width=0.62)
    for bar, val in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.008,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=17,
            color=COLORS["black"],
        )

    ax.axhline(naive_cost, color=COLORS["black"], linestyle=(0, (5, 3)), linewidth=2.4, zorder=2)
    ax.annotate(
        f"naive baseline\n{naive_cost:.3f}",
        xy=(0.50, naive_cost),
        xytext=(0.50, naive_cost + 0.045),
        ha="center",
        va="center",
        fontsize=15,
        color=COLORS["black"],
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.86),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=COLORS["black"]),
    )

    ax.set_title("B. Cost of certification", pad=14)
    ax.set_ylabel("Expected cost", labelpad=10)
    ax.set_ylim(3.10, 3.36)
    ax.set_xlim(-0.60, 2.45)
    ax.set_yticks([3.10, 3.15, 3.20, 3.25, 3.30, 3.35])
    ax.grid(axis="y", color="#E3E3E3", linewidth=1.1)


def _cifar_summary() -> tuple[list[str], list[float], list[float], list[float]]:
    fixed = load_json("results/cifar_branchy_real_cache_eval.json")["rows"]
    exploratory = load_json("results/cifar_branchy_real_cache_eval_exploratory.json")["rows"]

    specs = [
        (".30 pre", 0.30, fixed),
        (".35 expl.", 0.35, exploratory),
        (".40 expl.", 0.40, exploratory),
    ]

    labels: list[str] = []
    chain_viol: list[float] = []
    naive_viol: list[float] = []
    chain_feas: list[float] = []

    for label, alpha, rows in specs:
        labels.append(label)
        chain = [r for r in rows if r["method"] == "chain" and abs(r["alpha"] - alpha) < 1e-12]
        naive = [r for r in rows if r["method"] == "naive" and abs(r["alpha"] - alpha) < 1e-12]
        chain_viol.append(mean(r["violation_rate"] for r in chain))
        naive_viol.append(mean(r["violation_rate"] for r in naive))
        chain_feas.append(mean(r["feasible_rate"] for r in chain))

    return labels, chain_viol, naive_viol, chain_feas


def panel_c(ax: plt.Axes) -> None:
    labels, chain_viol, naive_viol, chain_feas = _cifar_summary()
    x = list(range(len(labels)))
    width = 0.34
    chain_x = [i - width / 2 for i in x]
    naive_x = [i + width / 2 for i in x]

    ax.bar(chain_x, chain_viol, width=width, color=COLORS["chain"], edgecolor=COLORS["black"], linewidth=1.2, label="chain")
    ax.bar(naive_x, naive_viol, width=width, color=COLORS["naive"], edgecolor=COLORS["black"], linewidth=1.2, label="naive")
    ax.axhline(0.10, color=COLORS["gray"], linestyle=":", linewidth=2.4, label=r"$\delta=.10$", zorder=1)

    for cx, viol, feas in zip(chain_x, chain_viol, chain_feas):
        ax.text(
            cx,
            max(viol + 0.018, 0.040),
            f"feas.\n{feas:.2f}",
            ha="center",
            va="bottom",
            fontsize=12.5,
            color=COLORS["chain"],
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=1.0),
            zorder=5,
        )

    ax.set_title("C. CIFAR finite-pool", pad=14)
    ax.set_xlabel(r"Target $\alpha$", labelpad=10)
    ax.set_ylabel("Mean violation rate", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.62)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.grid(axis="y", color="#E3E3E3", linewidth=1.1)
    handles = [
        Line2D([0], [0], color=COLORS["gray"], linestyle=":", linewidth=2.4, label=r"$\delta=.10$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["chain"], edgecolor=COLORS["black"], label="chain"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["naive"], edgecolor=COLORS["black"], label="naive"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)


def panel_d(ax: plt.Axes) -> None:
    data = load_json("results/e6_shift_eta_sweep.json")
    eta_rows = data["eta_sweep"]
    eta = [r["eta"] for r in eta_rows]
    feas = [r["estimated_weight"]["feasible_rate"] for r in eta_rows]
    mean_l1 = eta_rows[0]["mean_l1"]
    unweighted = data["unweighted"]["target_violation_rate_feasible"]
    failure = data["failure_mode"]
    biased_eta = failure["eta"]
    biased_fail = failure["target_violation_rate_feasible"]

    ax.plot(eta, feas, color=COLORS["chain"], marker="o", markersize=8.5, linewidth=3.4, label="feasibility")
    ax.axvline(mean_l1, color=COLORS["gray"], linestyle=":", linewidth=2.6, label=r"mean $L^1$")
    ax.scatter([0.0], [unweighted], color=COLORS["naive_dark"], marker="x", s=210, linewidths=3.2, label="unweighted fail", zorder=4)
    ax.scatter([biased_eta], [biased_fail], color=COLORS["naive"], marker="x", s=210, linewidths=3.2, label="biased fail", zorder=4)

    ax.set_title("D. Shift sensitivity", pad=14)
    ax.set_xlabel(r"Weight-error budget $\eta$", labelpad=10)
    ax.set_ylabel("Rate", labelpad=10)
    ax.set_xlim(-0.004, 0.125)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color="#E3E3E3", linewidth=1.1)
    ax.legend(loc="center right", frameon=False)


def finish_axes(axs: list[plt.Axes]) -> None:
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=9, width=1.7, pad=7)


def main() -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 10.05), dpi=180)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)
    finish_axes([ax_a, ax_b, ax_c, ax_d])

    fig.subplots_adjust(left=0.115, right=0.985, top=0.940, bottom=0.140, wspace=0.36, hspace=0.64)
    fig.savefig(OUT_DIR / "fig3_empirical_summary.png", dpi=180)
    fig.savefig(OUT_DIR / "fig3_empirical_summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
