"""Regenerate Figure 2: compute-chain boundary band schematic.

This is a schematic, not a data-derived result. It visualizes the theorem
semantics: the unsafe prefix lies to the left of k0, the validity frontier is
the first safe point, and the efficiency band is the compute span from k0 to k1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    x = [1.00, 1.32, 1.65, 1.97, 2.28, 2.60, 2.92, 3.25, 3.57, 3.90, 4.18]
    y = [0.365, 0.280, 0.218, 0.172, 0.139, 0.114, 0.096, 0.082, 0.072, 0.065, 0.061]
    alpha = 0.140
    alpha_minus_delta = 0.104
    k0 = 2.28
    k1 = 2.76

    colors = {
        "risk": "#0072B2",
        "alpha": "#D55E00",
        "margin": "#CC79A7",
        "band": "#F4C45D",
        "band_edge": "#E0A31D",
        "safe": "#009E73",
        "unsafe": "#6E6E6E",
        "guide": "#5B5B5B",
        "eff": "#E69F00",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "axes.linewidth": 1.8,
        }
    )

    fig, ax = plt.subplots(figsize=(10.75, 6.10), dpi=180)

    ax.axvspan(
        k0,
        k1,
        facecolor=colors["band"],
        edgecolor=colors["band_edge"],
        alpha=0.28,
        linewidth=2.0,
        zorder=0,
    )
    ax.vlines([k0, k1], 0.025, 0.405, colors=colors["guide"], linestyles=":", linewidth=2.2)

    risk_line, = ax.plot(
        x,
        y,
        color=colors["risk"],
        linewidth=4.2,
        marker="o",
        markersize=9.5,
        label="true chain risk",
        zorder=3,
    )
    alpha_line = ax.axhline(alpha, color=colors["alpha"], linewidth=3.4, label=r"$\alpha$")
    margin_line = ax.axhline(
        alpha_minus_delta,
        color=colors["margin"],
        linewidth=3.0,
        linestyle=(0, (5, 3)),
        label=r"$\alpha-\Delta_n$",
    )

    ax.text(k0, 0.382, r"$k_0$", ha="center", va="top", fontsize=24)
    ax.text(k1, 0.382, r"$k_1$", ha="center", va="top", fontsize=24)

    ax.annotate(
        "unsafe prefix",
        xy=(1.65, 0.218),
        xytext=(1.18, 0.327),
        color=colors["unsafe"],
        fontsize=20,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            lw=2.1,
            color=colors["unsafe"],
            connectionstyle="arc3,rad=-0.16",
        ),
    )

    ax.annotate(
        "validity\nfrontier",
        xy=(k0, alpha),
        xytext=(2.12, 0.262),
        color=colors["safe"],
        fontsize=20,
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=2.3, color=colors["safe"]),
    )

    # The efficiency band is the compute interval [k0, k1], not a point.
    band_y = 0.185
    ax.annotate(
        "",
        xy=(k1, band_y),
        xytext=(k0, band_y),
        arrowprops=dict(arrowstyle="<->", lw=2.4, color=colors["eff"], shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.annotate(
        "efficiency band\ncompute span",
        xy=((k0 + k1) / 2, band_y),
        xytext=(3.02, 0.258),
        color=colors["eff"],
        fontsize=20,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=2.1, color=colors["eff"]),
    )

    ax.set_xlim(0.90, 4.28)
    ax.set_ylim(0.025, 0.405)
    ax.set_xlabel("Compute along pre-specified chain", labelpad=8)
    ax.set_ylabel("True risk", labelpad=10)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.grid(axis="y", color="#E5E5E5", linewidth=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=10, width=1.8, pad=7)

    band_patch = Patch(
        facecolor=colors["band"],
        edgecolor=colors["band_edge"],
        alpha=0.28,
        linewidth=2.0,
        label="boundary band",
    )
    ax.legend(
        handles=[risk_line, alpha_line, margin_line, band_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.205),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.095, right=0.985, top=0.955, bottom=0.285)
    fig.savefig(OUT_DIR / "fig2_chain_band.png", dpi=180)
    fig.savefig(OUT_DIR / "fig2_chain_band.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
