"""Plot all six RNG-isolated control training trajectories."""

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
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axs = plt.subplots(
        1, 2, figsize=(183 / 25.4, 80 / 25.4), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.80, wspace=0.30)
    colors = {41: "#147F83", 42: "#4F6D99", 43: "#BA762D"}
    records = pd.read_csv(DATA / "Fig_S07_rng_trajectories.csv")
    assert len(records) == 600
    for condition in ["fixed", "free"]:
        for seed in [41, 42, 43]:
            data = records[
                (records.condition == condition) & (records.seed == seed)
            ].sort_values("epoch")
            epochs = data.epoch.to_numpy()
            asr = data.asr_pct.to_numpy()
            cacc = data.cacc_pct.to_numpy()
            assert epochs.tolist() == list(range(1, 101))
            for ax, values in zip(axs, [asr, cacc]):
                ax.plot(
                    epochs,
                    values,
                    color=colors[seed],
                    ls="-" if condition == "free" else "--",
                    lw=1.05 if condition == "free" else 0.85,
                    alpha=0.95 if condition == "free" else 0.8,
                    zorder=3 if condition == "free" else 2,
                )
    for index, ax in enumerate(axs):
        ax.set_xlim(1, 100)
        ax.set_xticks([1, 25, 50, 75, 100])
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color="#E2E5E7", lw=0.45, zorder=0)
        ax.annotate(
            chr(97 + index),
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-20, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    axs[0].set_ylim(0, 102)
    axs[0].set_yticks([0, 25, 50, 75, 100])
    axs[0].set_ylabel("ASR (%)")
    axs[1].set_ylim(0, 102)
    axs[1].set_yticks([0, 25, 50, 75, 100])
    axs[1].set_ylabel("CACC (%)")
    handles = [
        Line2D([], [], color=colors[seed], lw=1.3, label=f"Seed {seed}")
        for seed in [41, 42, 43]
    ]
    handles += [
        Line2D([], [], color="#50565B", lw=1.1, ls="-", label="Free GS"),
        Line2D([], [], color="#50565B", lw=0.95, ls="--", label="Fixed"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.15,
        handletextpad=0.45,
    )
    save(
        fig,
        "Fig_S07",
        require_panel_labels=True,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
    )


if __name__ == "__main__":
    main()
