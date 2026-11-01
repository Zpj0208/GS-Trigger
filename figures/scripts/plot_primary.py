"""Plot the primary architecture comparisons and RNG-isolated controls."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_support import configure, save_figure

WIDTH = 159.2 / 25.4
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#727272"


def main():
    args, DATA, _ = configure(__doc__)

    def save(fig, stem, **options):
        return save_figure(fig, stem, args, **options)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    def panel(ax, letter, descriptor):
        ax.text(
            -0.16,
            1.08,
            letter,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
            va="bottom",
        )
        ax.text(0, 1.08, descriptor, transform=ax.transAxes, va="bottom", fontsize=8.5)
        ax.tick_params(direction="out", length=3, pad=3)

    def legend(fig):
        fig.legend(
            [
                Line2D([], [], marker="o", color=ORANGE, ls="", markersize=4),
                Line2D([], [], marker="D", color=BLUE, ls="", markersize=4),
            ],
            ["Fixed-seed WaNet", "GS-Trigger"],
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
            ncol=2,
            columnspacing=1.3,
            handlelength=1.4,
            handletextpad=0.5,
            frameon=False,
        )

    def pairpanel(
        ax, groups, getter, metric, labels, summary=True, xlim=None, mark_new=False
    ):
        for i, g in enumerate(groups):
            fixed, gs = getter(g, metric)
            n = len(fixed)
            yy = i + np.linspace(-0.20, 0.20, n)
            for y, f, v in zip(yy, fixed, gs):
                ax.plot([f, v], [y, y], color="#C4C4C4", lw=0.7, zorder=1)
            ax.scatter(fixed, yy, c=ORANGE, s=14, marker="o", zorder=3)
            ax.scatter(gs, yy, c=BLUE, s=17, marker="D", zorder=3)
            if mark_new and i == 0:
                ax.annotate(
                    "*",
                    (gs[-1], yy[-1]),
                    xytext=(7, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    color=BLUE,
                )
            if summary:
                for vals, c, dy in ((fixed, ORANGE, 0.32), (gs, BLUE, 0.43)):
                    ax.errorbar(
                        np.mean(vals),
                        i + dy,
                        xerr=np.std(vals, ddof=1),
                        fmt="D",
                        color=c,
                        ms=4,
                        elinewidth=1.2,
                        capsize=2,
                        zorder=4,
                    )
        ax.set_yticks(range(len(groups)), labels)
        ax.set_ylim(len(groups) - 0.35, -0.45)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel("ASR (%)" if metric == "asr" else "CACC (%)")
        ax.grid(axis="x", color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)

    primary = pd.read_csv(DATA / "Fig_02_primary_exact_endpoints.csv")
    hosts = ["preactresnet18", "wrn34_10", "resnext29"]
    assert len(primary) == 22 and (primary.pratio == 0.01).all()
    rng = pd.read_csv(DATA / "Fig_02_rng_endpoints_unchanged.csv")
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 115 / 25.4))
    fig.subplots_adjust(
        left=0.19, right=0.98, bottom=0.11, top=0.86, wspace=0.72, hspace=0.65
    )

    def pg(host, metric):
        z = primary[primary.model == host]
        return (
            z[z.method == "Fixed"].sort_values("seed")[metric].to_numpy(),
            z[z.method == "GS"].sort_values("seed")[metric].to_numpy(),
        )

    for j, m in enumerate(("asr", "cacc")):
        pairpanel(
            axs[0, j],
            hosts,
            pg,
            m,
            ["PreAct-18 (n = 5)", "WRN-34-10 (n = 3)", "ResNeXt-29 (n = 3)"],
            xlim=(-5, 110) if j == 0 else (81, 95),
            mark_new=False,
        )
        panel(axs[0, j], "ab"[j], "Primary protocol")

    def rg(seed, metric):
        z = rng[rng.seed == seed]
        c = "ASR_final_pct" if metric == "asr" else "CACC_final_pct"
        return (
            z[z.condition == "fixed"][c].to_numpy(),
            z[z.condition == "free"][c].to_numpy(),
        )

    for j, m in enumerate(("asr", "cacc")):
        pairpanel(
            axs[1, j],
            [41, 42, 43],
            rg,
            m,
            ["Seed 41", "Seed 42", "Seed 43"],
            summary=False,
            xlim=(-5, 105) if j == 0 else (88, 93),
        )
        panel(axs[1, j], "cd"[j], "Isolated evaluation RNG")
    legend(fig)
    save(fig, "Fig_02")


if __name__ == "__main__":
    main()
