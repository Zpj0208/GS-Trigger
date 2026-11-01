"""Plot the mixed-loss seed-gradient measurements."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    grad = pd.read_csv(DATA / "mixed_gradient_events.csv")
    fig, axs = grid(2, 1, 140)
    fig.subplots_adjust(left=0.14, bottom=0.11, top=0.83, hspace=0.90)
    for k, (col, label, color) in enumerate(
        [
            ("gn_seed_bd", "Poison", ORANGE),
            ("gn_seed_cross", "Cross-mix", BLUE),
            ("gn_seed_clean", "Clean", GREY),
        ]
    ):
        z = grad.copy()
        x = z.epoch + np.where(z.batch == 10, -0.12, 0.12)
        axs[0, 0].plot(x, z[col], color=color, lw=0.6, marker=".", ms=1.8, label=label)
    axs[0, 0].set(xlim=(0.5, 100.5), ylabel="Seed-gradient L2 norm", xlabel="Epoch")
    axs[0, 0].legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.20),
        fontsize=8,
        frameon=False,
    )
    panel(axs[0, 0], "a", "Loss-component gradients")
    for col, label, color in [
        ("cos_bd_cross", "Poison / cross-mix", BLUE),
        ("cos_bd_clean", "Poison / clean", GREY),
    ]:
        axs[1, 0].plot(
            x, grad[col], color=color, lw=0.6, marker=".", ms=1.8, label=label
        )
    axs[1, 0].axhline(0, color="#B0B0B0", lw=0.6)
    axs[1, 0].set(
        xlim=(0.5, 100.5), ylim=(-1.1, 1.1), ylabel="Gradient cosine", xlabel="Epoch"
    )
    axs[1, 0].legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.20),
        fontsize=8,
        frameon=False,
    )
    panel(axs[1, 0], "b", "Gradient alignment")
    save(fig, "Fig_S06")


if __name__ == "__main__":
    main()
