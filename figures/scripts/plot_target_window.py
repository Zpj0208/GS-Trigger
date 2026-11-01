"""Plot the two recorded target-3 final-ten-epoch trajectories."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_support import configure, save_figure

BLUE = "#0072B2"
GREY = "#727272"


def main():
    args, DATA, _ = configure(__doc__)

    def save(fig, stem, **options):
        return save_figure(fig, stem, args, **options)

    records = pd.read_csv(DATA / "Fig_S05_last_ten_epochs.csv").to_dict("records")
    assert len(records) == 20
    WIDTH_MM = 159.2
    HEIGHT_MM = WIDTH_MM * 3.1 / 6.5
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.89)
    for method, color, marker in [
        ("GS-Trigger", "#007F7A", "o"),
        ("Fixed-seed WaNet", "#747B86", "s"),
    ]:
        data = [row for row in records if row["method"] == method]
        assert len(data) == 10
        ax.plot(
            [row["epoch"] for row in data],
            [row["ASR_percent"] for row in data],
            color=color,
            marker=marker,
            markersize=3.6,
            linewidth=1.2,
            label=method,
            zorder=3,
        )
    ax.set_xticks(range(91, 101))
    ax.set_ylim(55, 101)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_xlim(90.55, 100.45)
    ax.set_xlabel("Training epoch", labelpad=5)
    ax.set_ylabel("ASR (%)", labelpad=5)
    ax.set_title("Target 3, seed 42: final ten epochs", loc="left", pad=7)
    ax.tick_params(length=3, width=0.7, pad=3)
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower left",
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=0.5,
        borderaxespad=0.8,
    )

    save(fig, "Fig_S05")


if __name__ == "__main__":
    main()
