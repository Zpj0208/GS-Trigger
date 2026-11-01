"""Plot target-label and single-seed dataset extensions."""

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
MARKERS = ["o", "s", "^", "v", "P"]


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

    def grid(rows, cols, height):
        fig, axs = plt.subplots(
            rows, cols, figsize=(WIDTH, height / 25.4), squeeze=False
        )
        fig.subplots_adjust(
            left=0.105, right=0.98, bottom=0.16, top=0.90, wspace=0.36, hspace=0.74
        )
        return fig, axs

    def panel(ax, letter, descriptor):
        ax.text(
            -0.10 if ax.get_position().width > 0.6 else -0.16,
            1.08,
            letter,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
            va="bottom",
        )
        ax.text(0, 1.08, descriptor, transform=ax.transAxes, va="bottom", fontsize=8.5)
        ax.tick_params(direction="out", length=3, pad=3)

    def legend(fig, handles=None, labels=None, y=0.99):
        if handles is None:
            handles = [
                Line2D([], [], marker="o", color=ORANGE, ls="", markersize=4),
                Line2D([], [], marker="D", color=BLUE, ls="", markersize=4),
            ]
            labels = ["Fixed-seed WaNet", "GS-Trigger"]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.54, y),
            ncol=len(labels),
            columnspacing=1.3,
            handlelength=1.4,
            handletextpad=0.5,
            frameon=False,
        )

    def pairpanel(ax, groups, getter, metric, labels, summary=True, xlim=None):
        for i, g in enumerate(groups):
            fixed, gs = getter(g, metric)
            n = len(fixed)
            yy = i + np.linspace(-0.20, 0.20, n)
            for y, f, v in zip(yy, fixed, gs):
                ax.plot([f, v], [y, y], color="#C4C4C4", lw=0.7, zorder=1)
            ax.scatter(fixed, yy, c=ORANGE, s=14, marker="o", zorder=3)
            ax.scatter(gs, yy, c=BLUE, s=17, marker="D", zorder=3)
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

    def dots(ax, df, key, cats, labels, colors, ylim=None):
        for i, c in enumerate(cats):
            v = df[df.condition == c].sort_values("seed")[key].to_numpy()
            xx = i + np.linspace(-0.15, 0.15, len(v))
            for j, (x, y) in enumerate(zip(xx, v)):
                ax.scatter(x, y, s=20, color=colors[i], marker=MARKERS[j], zorder=3)
            ax.errorbar(
                i + 0.26,
                v.mean(),
                yerr=v.std(ddof=1),
                fmt="D",
                color=colors[i],
                ms=3.5,
                capsize=2,
                elinewidth=1,
                zorder=2,
            )
        ax.set_xticks(range(len(cats)), labels)
        ax.set_xlim(-0.45, len(cats) - 0.4)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)

    targets = pd.read_csv(DATA / "target_endpoints.csv")
    cross = pd.read_csv(DATA / "cross_dataset_author_table.csv")
    fig, axs = grid(2, 2, 125)
    fig.subplots_adjust(
        left=0.21, right=0.98, bottom=0.12, top=0.86, wspace=0.76, hspace=0.65
    )

    def tg(target, metric):
        z = targets[targets.target == target]
        key = metric + "_pct"
        return (
            z[z.condition == "Fixed"].sort_values("seed")[key].to_numpy(),
            z[z.condition == "Free"].sort_values("seed")[key].to_numpy(),
        )

    for j, m in enumerate(("asr", "cacc")):
        pairpanel(
            axs[0, j],
            [0, 3, 7],
            tg,
            m,
            ["Target 0", "Target 3", "Target 7"],
            summary=True,
            xlim=(-5, 110) if j == 0 else (87, 94),
        )
        panel(axs[0, j], "ab"[j], "Targets (n = 3)")

    def cg(i, metric):
        z = cross.iloc[i]
        return np.array([z["fixed_" + metric]]), np.array([z["gs_" + metric]])

    labels = [f"{r.dataset}\n{r.architecture}" for r in cross.itertuples()]
    for j, m in enumerate(("asr", "cacc")):
        pairpanel(
            axs[1, j],
            range(4),
            cg,
            m,
            labels,
            summary=False,
            xlim=(-5, 105) if j == 0 else (58, 101),
        )
        panel(axs[1, j], "cd"[j], "Datasets (n = 1)")
    legend(fig, y=0.995)
    save(fig, "Fig_04")


if __name__ == "__main__":
    main()
