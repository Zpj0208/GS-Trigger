"""Plot the archived post-epoch seed diagnostics."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_support import configure, save_figure

WIDTH = 159.2 / 25.4
BLUE = "#0072B2"
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
    probes = pd.read_csv(DATA / "Fig_S02_amplitude_trajectories.csv")
    outcomes = pd.read_csv(DATA / "Fig_S02_probe1pct_outcomes.csv")
    grad = pd.read_csv(DATA / "Fig_S02_probe1pct_gradients.csv")
    probe_paths = {
        label: None for label in ["0.5% / k=4", "1% / k=4", "2% / k=4", "10% / k=8"]
    }
    with plt.rc_context(
        {
            "font.size": 6.7,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.edgecolor": "#8C9BA3",
            "axes.labelcolor": "#263840",
            "xtick.color": "#60737F",
            "ytick.color": "#60737F",
        }
    ):
        fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 59.258 / 25.4))
        fig.subplots_adjust(left=0.066, right=0.98, bottom=0.24, top=0.80, wspace=0.42)
        teal = "#138986"
        muted = "#697B82"
        ochre = "#CF7A2D"
        pcolors = ["#6595B9", teal, "#D08B4B", "#927CB6"]
        for label, color in zip(probe_paths, pcolors):
            z = probes[probes.setting == label]
            axs[0].plot(
                z.epoch,
                z.mean_abs_seed,
                color=color,
                lw=1.1 if label != "1% / k=4" else 1.7,
                label=label,
            )
        axs[0].set(xlim=(1, 100), ylim=(0.985, 1.355), ylabel="Seed magnitude")
        axs[0].set_yticks([1, 1.1, 1.2, 1.3])
        axs[1].plot(
            np.arange(1, 101), outcomes.test_asr * 100, color=teal, lw=1.3, label="ASR"
        )
        axs[1].plot(
            np.arange(1, 101),
            outcomes.test_acc * 100,
            color=muted,
            lw=1.0,
            label="CACC",
        )
        axs[1].axvline(24, color=teal, lw=0.8, ls="--")
        axs[1].set(xlim=(1, 100), ylim=(0, 106), ylabel="Percentage (%)")
        axs[1].set_yticks([0, 25, 50, 75, 100])
        axs[1].legend(
            loc="lower right", frameon=False, handlelength=1.4, labelspacing=0.25
        )
        axs[2].plot(grad.epoch, grad.gn_poison_seed, color=ochre, lw=1.15)
        peak = grad.loc[grad.gn_poison_seed.idxmax()]
        axs[2].axvline(peak.epoch, color=ochre, lw=0.75, ls="--")
        axs[2].scatter(
            [peak.epoch],
            [peak.gn_poison_seed],
            s=18,
            color=ochre,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        axs[2].set(xlim=(1, 100), ylim=(-0.02, 1.05), ylabel="Gradient L2 norm")
        axs[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
        titles = ["Seed magnitude", "Attack emergence", "Poison-only gradient"]
        subtitles = [
            "Four runs; seed 42",
            "1% / k=4; ASR > 50%: ep24",
            "Peak 0.95 at epoch 53",
        ]
        for i, ax in enumerate(axs):
            ax.set_xlabel("Training epoch")
            ax.set_xticks([1, 25, 50, 75, 100])
            ax.grid(axis="y", color="#D9E1E4", lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(length=2.5, pad=2)
            ax.text(
                -0.23,
                1.18,
                "abc"[i],
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=8,
                va="bottom",
                color="#263840",
            )
            ax.text(
                0,
                1.18,
                titles[i],
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=7.3,
                va="bottom",
                color="#263840",
            )
            ax.text(
                0,
                1.055,
                subtitles[i],
                transform=ax.transAxes,
                fontsize=6.5,
                color="#60737F",
                va="bottom",
            )
        fig.legend(
            *axs[0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.535, 0.005),
            ncol=4,
            fontsize=6.7,
            frameon=False,
            handlelength=1.3,
            handletextpad=0.4,
            columnspacing=1.15
        )
        save(fig, "Fig_S02")


if __name__ == "__main__":
    main()
