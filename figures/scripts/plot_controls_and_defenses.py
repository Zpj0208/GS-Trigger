"""Plot amplitude/PSNR controls and repair/STRIP results."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_support import configure, save_figure

WIDTH = 159.2 / 25.4
BLUE = "#0072B2"
GREY = "#727272"
COLORS = [GREY, "#E69F00", "#009E73", BLUE, "#CC79A7"]
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

    def dots(ax, df, key, cats, labels, colors, ylim):
        for i, c in enumerate(cats):
            vals = df[df.condition == c].sort_values("seed")[key].to_numpy()
            assert len(vals) == 3
            for x, y, m in zip(i + np.linspace(-0.15, 0.15, 3), vals, MARKERS):
                ax.scatter(x, y, s=20, color=colors[i], marker=m, zorder=3)
            ax.errorbar(
                i + 0.26,
                vals.mean(),
                yerr=vals.std(ddof=1),
                fmt="D",
                color=colors[i],
                ms=3.5,
                capsize=2,
                elinewidth=1,
                zorder=2,
            )
        ax.set_xticks(range(len(cats)), labels)
        ax.set_xlim(-0.45, len(cats) - 0.4)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)

    PINK = "#CC79A7"
    MARKERS = ["o", "s", "^"]

    def load(name, cols=None):
        df = pd.read_csv(DATA / name)
        return df if cols is None else df[cols]

    end = load("Fig_03_standard_controls_unchanged.csv")
    joint = load("Fig_03_joint_rng_endpoints.csv")
    psnr = load("Fig_03_joint_psnr_verified.csv")
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 118 / 25.4))
    fig.subplots_adjust(
        left=0.11, right=0.98, bottom=0.13, top=0.92, wspace=0.39, hspace=0.75
    )
    conditions = ["Fixed", "Amplified", "Projected", "Free", "PSNR-match"]
    labels = ["Fixed", "Static\namp.", "Projected", "Free", "PSNR-\nmatched"]
    dots(axs[0, 0], end, "asr_pct", conditions, labels, COLORS, (0, 109))
    axs[0, 0].set_ylabel("Epoch 100 ASR (%)")
    panel(axs[0, 0], "a", "Standard evaluation")
    dots(axs[1, 0], end, "cacc_pct", conditions, labels, COLORS, (87, 93))
    axs[1, 0].set_ylabel("Epoch 100 CACC (%)")
    panel(axs[1, 0], "c", "Standard evaluation")
    # The unit-magnitude Fixed comparison appears in Fig. 2c.
    cats = ["PSNR-match", "Free"]
    for seed, offset, m in zip((41, 42, 43), np.linspace(-0.15, 0.15, 3), MARKERS):
        vals = [
            joint[(joint.seed == seed) & (joint.condition == c)].asr_pct.item()
            for c in cats
        ]
        axs[0, 1].plot([offset, 1 + offset], vals, color="#C7C7C7", lw=0.7, zorder=1)
    dots(
        axs[0, 1],
        joint,
        "asr_pct",
        cats,
        ["PSNR-matched\nFixed", "Free GS"],
        [PINK, BLUE],
        (85, 101),
    )
    axs[0, 1].set_yticks([85, 90, 95, 100])
    axs[0, 1].set_ylabel("Epoch 100 ASR (%)")
    panel(axs[0, 1], "b", "Isolated RNG")
    for seed, m in zip((41, 42, 43), MARKERS):
        z = psnr[psnr.seed == seed].iloc[0]
        vals = [z.psnr_free_first1000_nontarget, z.psnr_fixed_first1000_nontarget]
        axs[1, 1].plot([0, 1], vals, color="#BDBDBD", lw=0.8)
        axs[1, 1].scatter([0, 1], vals, c=[BLUE, PINK], marker=m, s=24, zorder=3)
    axs[1, 1].set_xticks([0, 1], ["Free GS", "PSNR-matched\nFixed"])
    axs[1, 1].set_xlim(-0.35, 1.35)
    axs[1, 1].set_ylim(26.8, 28.25)
    axs[1, 1].set_ylabel("Mean PSNR (dB)")
    panel(axs[1, 1], "d", "Isolated RNG: matching")
    save(fig, "Fig_03")
    repair = load(
        "Fig_S03_repair_endpoints.csv",
        ["method", "condition", "source_attack_seed", "ASR_pct", "CACC_pct"],
    )
    budget = load(
        "Fig_S03_strip_budget_mapping.csv",
        [
            "model",
            "calibration_n",
            "overlay_n",
            "threshold_low",
            "TN",
            "FP",
            "FN",
            "TP",
            "TPR_pct",
            "FPR_pct",
        ],
    )
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 82 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.23, top=0.83, wspace=0.5)
    teal = "#138986"
    muted = "#697B82"
    light = "#D9E1E4"
    methods = [
        "Fine-Pruning",
        "Fine-tuning",
        "Spectral",
        "NAD",
        "ABL",
        "CLP",
        "ANP",
        "I-BAU",
        "NPD",
    ]
    for method in methods:
        pair = repair[repair.method == method].set_index("condition")
        f = pair.loc["Fixed"]
        g = pair.loc["GS"]
        axs[0].plot(
            [f.ASR_pct, g.ASR_pct],
            [f.CACC_pct, g.CACC_pct],
            color=light,
            lw=0.85,
            zorder=1,
        )
        axs[0].scatter(
            f.ASR_pct,
            f.CACC_pct,
            color=muted,
            s=22,
            marker="o",
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
        axs[0].scatter(
            g.ASR_pct,
            g.CACC_pct,
            color=teal,
            s=24,
            marker="D",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    axs[0].set(
        xlim=(-4, 72),
        ylim=(48, 97),
        xlabel="Residual ASR (%)",
        ylabel="CACC after repair (%)",
    )
    axs[0].set_xticks([0, 20, 40, 60])
    axs[0].set_yticks([50, 60, 70, 80, 90])
    axs[0].grid(color=light, lw=0.5)
    b = budget[budget.model == "GS AvpT"].sort_values("calibration_n")
    x = b.calibration_n.to_numpy()
    assert list(x) == [100, 500, 1000, 2000]
    axs[1].plot(x, b.TPR_pct, "o-", color="#CF7A2D", lw=1, ms=4, label="TPR")
    axs[1].plot(x, b.FPR_pct, "s--", color=muted, lw=1, ms=4, label="FPR")
    axs[1].set(
        xlim=(0, 2100),
        ylim=(0, 22),
        xlabel="Calibration images",
        ylabel="Detection rate (%)",
    )
    axs[1].set_xticks(x, [str(v) for v in x])
    axs[1].set_yticks([0, 5, 10, 15, 20])
    axs[1].grid(axis="y", color=light, lw=0.5)
    axs[1].legend(
        loc="center right", bbox_to_anchor=(1, 0.52), frameon=False, handlelength=1.5
    )
    for i, ax in enumerate(axs):
        ax.set_axisbelow(True)
        panel(ax, "ab"[i], ["Post-repair trade-off", "STRIP calibration"][i])
    fig.legend(
        [
            Line2D([], [], marker="D", color=teal, ls="", markersize=4),
            Line2D([], [], marker="o", color=muted, ls="", markersize=4),
        ],
        ["GS-Trigger", "Fixed WaNet"],
        loc="lower left",
        bbox_to_anchor=(0.13, 0.01),
        ncol=2,
        frameon=False,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    save(fig, "Fig_S03")


if __name__ == "__main__":
    main()
