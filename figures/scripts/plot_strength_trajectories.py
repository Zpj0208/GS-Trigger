"""Plot control-run accuracy and seed-magnitude trajectories."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_support import configure, save_figure

BLUE = "#0072B2"
GREY = "#727272"


def main():
    args, DATA, _ = configure(__doc__)

    def save(fig, stem, **options):
        return save_figure(fig, stem, args, **options)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
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
    curves = pd.read_csv(DATA / "control_epoch_values.csv")
    amp = pd.read_csv(DATA / "amplitude_epoch_values.csv")
    conditions = ["Fixed", "Amplified", "Projected", "Free", "PSNR-match"]
    display = ["Fixed", "Amplified", "Projected GS", "Free GS", "PSNR-matched"]
    colors = {
        "Fixed": "#727272",
        "Amplified": "#E69F00",
        "Projected": "#009E73",
        "Free": "#0072B2",
        "PSNR-match": "#CC79A7",
    }
    styles = {41: "-", 42: "--", 43: ":"}
    assert len(curves) == 1500 and len(amp) == 600
    assert set(curves.condition) == set(conditions) and set(amp.condition) == {
        "Free",
        "Projected",
    }
    assert set(curves.seed) == set(amp.seed) == {41, 42, 43}
    for df in (curves, amp):
        for _, g in df.groupby(["condition", "seed"]):
            assert g.epoch.tolist() == list(range(1, 101))
    assert (
        float(
            curves[
                (curves.condition == "Free")
                & (curves.seed == 42)
                & (curves.epoch == 100)
            ].asr_pct.iloc[0]
        )
        == 90.0
    )
    assert (amp[amp.condition == "Projected"].mean_abs_seed == 1).all()
    WMM, HMM = 159.2, 130
    fig = plt.figure(figsize=(WMM / 25.4, HMM / 25.4))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.115,
        right=0.985,
        bottom=0.10,
        top=0.82,
        wspace=0.34,
        hspace=0.66,
        height_ratios=[1, 1],
    )
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
    ]
    for ax in axs:
        ax.tick_params(direction="out", length=3, pad=3)
        ax.set_xlim(1, 100)
        ax.set_xticks([1, 20, 40, 60, 80, 100])
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color="#ECECEC", linewidth=0.5)
        ax.set_axisbelow(True)
    for ax, key, ylab, ylim, yticks in [
        (axs[0], "asr_pct", "ASR (%)", (0, 102), [0, 20, 40, 60, 80, 100]),
        (axs[1], "cacc_pct", "CACC (%)", (50, 100), [50, 60, 70, 80, 90, 100]),
    ]:
        for cond in conditions:
            for seed, ls in styles.items():
                z = curves[(curves.condition == cond) & (curves.seed == seed)]
                ax.plot(
                    z.epoch,
                    z[key],
                    color=colors[cond],
                    linestyle=ls,
                    linewidth=0.75,
                    alpha=0.95,
                )
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
    for cond in ["Free", "Projected"]:
        for seed, ls in styles.items():
            z = amp[(amp.condition == cond) & (amp.seed == seed)]
            axs[2].plot(
                z.epoch, z.mean_abs_seed, color=colors[cond], linestyle=ls, linewidth=1
            )
    axs[2].set_ylabel("Mean absolute seed")
    axs[2].set_ylim(0.98, 1.36)
    axs[2].set_yticks([1.0, 1.1, 1.2, 1.3])
    labels = ["a", "b", "c"]
    titles = ["Attack success", "Clean accuracy", "Seed magnitude"]
    fig.canvas.draw()
    for ax, letter, title in zip(axs, labels, titles):
        width_pt = ax.get_position().width * WMM / 25.4 * 72
        ax.text(
            -28 / width_pt,
            1.07,
            letter,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        ax.text(0, 1.07, title, transform=ax.transAxes, fontsize=8.5, va="bottom")
    method_handles = [
        Line2D([], [], color=colors[c], linewidth=1.2) for c in conditions
    ]
    seed_handles = [
        Line2D([], [], color="#303030", linestyle=styles[s], linewidth=1)
        for s in styles
    ]
    fig.legend(
        method_handles,
        display,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.992),
        frameon=False,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0,
    )
    fig.legend(
        seed_handles,
        [f"Seed {s}" for s in styles],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.945),
        frameon=False,
        handlelength=2.5,
        handletextpad=0.6,
        columnspacing=2.0,
        borderaxespad=0,
    )
    fig.canvas.draw()
    save(
        fig,
        "Fig_S04",
        panel_ids=labels,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
    )


if __name__ == "__main__":
    main()
