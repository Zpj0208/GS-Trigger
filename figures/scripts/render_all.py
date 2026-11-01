"""Render the eleven data figures; edit the method schematic in its PPTX."""

from pathlib import Path
import subprocess
import sys


def main():
    folder = Path(__file__).resolve().parent
    scripts = [
        "plot_primary.py",
        "plot_controls_and_defenses.py",
        "plot_extensions.py",
        "plot_trigger_images.py",
        "plot_sensitivity.py",
        "plot_probe_diagnostics.py",
        "plot_strength_trajectories.py",
        "plot_target_window.py",
        "plot_gradient_decomposition.py",
        "plot_rng_trajectories.py",
    ]
    for script in scripts:
        subprocess.run(
            [sys.executable, "-B", str(folder / script), *sys.argv[1:]], check=True
        )


if __name__ == "__main__":
    main()
