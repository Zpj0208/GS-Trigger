"""Command-line paths, fonts, and exports shared by the figure scripts."""

import argparse
from pathlib import Path
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from lxml import etree
from audit_panel_alignment import require_matplotlib_panel_alignment

TIFF_DPI = 1000
PNG_DPI = 300


def configure(description):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=root / "plot_data")
    parser.add_argument("--output-dir", type=Path, default=root / "generated")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "svg", "png", "tiff"],
        default=["pdf", "svg", "tiff"],
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        help="Optional folder containing licensed Arial font files.",
    )
    args = parser.parse_args()
    if args.font_dir:
        for file in args.font_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(file)
    font_manager.findfont("Arial", fallback_to_default=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qa").mkdir(exist_ok=True)
    return args, args.data_dir.resolve(), args.output_dir.resolve()


def save_figure(fig, stem, args, **alignment):
    """Save .pdf, .svg, .tiff, or .png files after checking panel alignment."""
    destination = args.output_dir
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=destination / "qa" / (stem + ".alignment.json"),
        overlay_svg=destination / "qa" / (stem + ".alignment.svg"),
        strict=True,
        **alignment,
    )
    for extension in args.formats:
        path = destination / (stem + "." + extension)
        options = (
            {"dpi": TIFF_DPI, "pil_kwargs": {"compression": "tiff_lzw"}}
            if extension == "tiff"
            else {}
        )
        if extension == "png":
            options["dpi"] = PNG_DPI
        if extension == "pdf":
            options["metadata"] = {"CreationDate": None, "ModDate": None}
        fig.savefig(path, **options)
        if extension == "svg":
            text = re.sub(r"<!DOCTYPE[^>]*>", "", path.read_text("utf-8"), flags=re.S)
            tree = etree.fromstring(
                text.encode("utf-8"),
                etree.XMLParser(
                    resolve_entities=False, no_network=True, load_dtd=False
                ),
            )
            for element in tree.iter():
                style = element.get("style", "")
                match = re.search(r"font-size:\s*([\d.]+)px", style)
                if match:
                    element.set("font-size", match[1])
                    element.set(
                        "style", re.sub(r"font-size:\s*[\d.]+px;?\s*", "", style)
                    )
            path.write_bytes(
                etree.tostring(tree, encoding="utf-8", xml_declaration=True)
            )
        if extension == "tiff":
            with Image.open(path) as original:
                rgb = original.convert("RGB")
            rgb.save(path, dpi=(TIFF_DPI, TIFF_DPI), compression="tiff_lzw")
    plt.close(fig)
    print(stem, flush=True)
