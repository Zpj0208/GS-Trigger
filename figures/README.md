# Figure sources

The current manuscript has five main figures and seven supplementary figures. Figure 1 is an explanatory method schematic and is not included in this code/data repository. The other eleven figures are rendered with Python and Matplotlib. `figure_sources.csv` maps each figure to its entry point.

## Render the data figures

Use a separate Python 3.12 environment for the plotting requirements, leaving the historical training environment unchanged. Install `requirements-plotting.txt` and ensure that Arial is available. If Arial is not installed, provide a folder containing Arial TTF files with `--font-dir`. The scripts stop rather than silently substituting a font.

Run from the reproduction-package root:

```text
python -m pip install -r figures/requirements-plotting.txt
python figures/scripts/render_all.py
```

The default output folder is `figures/generated/`. Outputs are embedded-font PDF, SVG with editable text, and RGB TIFF at 1000 dpi. To choose another destination or export only vector files:

```text
python figures/scripts/render_all.py --output-dir figure_exports --formats pdf svg
```

Each plotting entry accepts the same `--data-dir`, `--output-dir`, `--formats`, and `--font-dir` options. The controls/defenses entry produces both Figure 3 and Figure S3. The scripts read package-relative inputs and do not use a machine-specific skill installation.

## Data and uncertainty

`plot_data/` contains the inputs used by the eleven data figures. Run `python figures/scripts/check_plot_inputs.py` to verify their hashes and table sizes. `plot_data/data_index.json` records the source hashes, retained columns, and row counts. Removing machine-specific provenance columns from these plotting views does not alter measurement cells or row order. The original numerical records and prior source CSVs remain under `results/` and `figures/source_data/`.

Means and sample SD are computed over training runs where the plot contains run aggregates. Trajectories, gradient events, image pairs, and calibration settings retain their original roles. They are not counted as additional independent training runs. No smoothing or new training is introduced by these scripts.

`visual_inputs/` stores the two CIFAR-10 example inputs and their recorded transformed arrays, together with the seed records used. It contains a processed illustration subset, not the full test dataset. The images are the fixed first two non-target examples used in the manuscript. To check the RGB and difference arrays against the recorded seeds, use the study's PyTorch 2.4.1 and NumPy 1.26.4 environment and run:

```text
python figures/scripts/verify_visual_inputs.py
```

This optional check reconstructs the image warps on CPU. It neither trains a model nor evaluates attack success. The ordinary plotting commands read the saved arrays and do not require PyTorch.

## Source and assistance

Plotting statements were adapted from the scripts used for the manuscript, with relative paths and shared export handling. The unmodified panel-alignment helper comes from the nature-figure tooling used for those plots. Existing file-level notices and source attribution are retained; no new blanket license is assigned to third-party components.

AI assistance was used for plotting code, diagram layout, and release documentation. The authors supplied the experimental records. The package retains the measurements and the distinction between original observations, derived tables, and illustrative reconstructions.
