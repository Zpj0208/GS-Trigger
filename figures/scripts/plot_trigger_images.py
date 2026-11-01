"""Plot the saved common-input trigger and spatial-difference arrays."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_support import configure, save_figure

BLUE = "#0072B2"
GREY = "#727272"

import matplotlib as mpl
from matplotlib.colors import Normalize


def main():
    args, DATA, _ = configure(__doc__)

    def save(fig, stem, **options):
        return save_figure(fig, stem, args, **options)

    visual = DATA.parent / "visual_inputs"
    with np.load(
        visual / "reconstructed_inputs_float32.npz", allow_pickle=False
    ) as source:
        imgs = source["rgb"]
        shown = source["displayed_difference"]
        indices = source["test_indices"]
    assert imgs.shape == (2, 6, 32, 32, 3) and shown.shape == (2, 6, 32, 32)
    display = [
        "Clean",
        "Fixed",
        "Static\namplified",
        "Projected\nGS",
        "Free\nGS",
        "PSNR-matched\nFixed",
    ]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": "black",
        }
    )
    fig, axes = plt.subplots(4, 6, figsize=(159.2 / 25.4, 119 / 25.4), squeeze=False)
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.115, top=0.905, wspace=0.065, hspace=0.14
    )
    for r in range(4):
        example = r // 2
        is_diff = r % 2
        for c in range(6):
            ax = axes[r, c]
            if is_diff:
                ax.imshow(
                    shown[example, c],
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
            else:
                ax.imshow(imgs[example, c], interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.canvas.draw()
    for c, label in enumerate(display):
        b = axes[0, c].get_position()
        fig.text(b.x0 + b.width / 2, 0.975, label, ha="center", va="top", fontsize=7.2)
    for r in range(4):
        b = axes[r, 0].get_position()
        label = f"Test {int(indices[r//2])}: input" if r % 2 == 0 else "10× difference"
        fig.text(
            0.039,
            b.y0 + b.height / 2,
            label,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=7.2,
        )
    for r, letter in [(0, "a"), (2, "b")]:
        b = axes[r, 0].get_position()
        fig.text(
            0.012, b.y1, letter, fontweight="bold", ha="left", va="top", fontsize=8.5
        )
    cax = fig.add_axes([0.36, 0.070, 0.35, 0.016])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap="magma"),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 0.5, 1],
    )
    cb.ax.tick_params(labelsize=6.8, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    fig.text(
        0.535,
        0.013,
        "10 × mean RGB |Δ| (clipped at 1)",
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    save(
        fig,
        "Fig_05",
        exclude_axes=[cax],
        panel_ids=[f"cell_{r}_{c}" for r in range(4) for c in range(6)],
    )


if __name__ == "__main__":
    main()
