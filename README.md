# GS-Trigger

Reproduction code and supporting data for **GS-Trigger: Gradient-steered seed trigger for backdoor implantation at low per-batch poisoning rates**.

GS-Trigger learns a shared two-channel geometric seed during classifier training. The seed is expanded into a dense displacement field and applied through the WaNet sampling operator. The learned seed is frozen for inference. The Fixed comparison uses the corresponding normalized frozen seed.

## Contents

| Folder | Contents |
|---|---|
| `code/BackdoorBench/` | Training code, model definitions, data readers, and defense implementations |
| `configs/` | Recorded arguments, original YAML defaults, and run-to-result indexes |
| `results/` | Original numerical records and logs, log-derived epoch data, and validated result tables |
| `figures/` | Python plotting entries, figure input tables, and saved trigger-illustration arrays |
| `data_manifests/` | File hashes, source indexes, and GTSRB annotations/subset indexes |
| `environment/` | Training and defense dependency lists |
| `licenses/` | Retained upstream license texts |

The manuscript, supplementary document, cover letter, and submission artwork are separate from this repository. Full image datasets and trained model checkpoints are not bundled.

## Check the supplied results

Python 3.9 or later is sufficient for the following checks. They use the standard library and require no GPU:

```text
python scripts/check_inputs.py
python scripts/recompute_results.py
python scripts/verify_gtsrb.py
python figures/scripts/check_plot_inputs.py
```

The result script verifies all 2,200 epochs from the 22 primary training logs and recomputes the endpoint means and sample SD. It also recomputes the nine distinct RNG-isolated control runs. Reports are written to `results/recomputed/`.

| CIFAR-10 comparison | GS ASR (%) | Fixed ASR (%) | Runs per method |
|---|---:|---:|---:|
| PreActResNet-18 | 98.35 ± 1.69 | 75.72 ± 7.39 | 4 |
| WRN-34-10 | 85.85 ± 13.31 | 39.92 ± 26.10 | 3 |
| ResNeXt-29 | 94.09 ± 3.51 | 42.76 ± 26.60 | 3 |

The PreActResNet-18 row gives the four same-environment runs reported in the manuscript; the additional cross-environment seed-45 pair is reported separately in Supplementary Table S2.

The RNG-isolated Free-minus-PSNR-matched Fixed mean ASR difference is 3.45 percentage points. The same three Free runs contribute to the two RNG-isolated comparisons. Variability is sample SD across training runs, not across epochs or images.

## Train models

The experimental records use PyTorch 2.4.1+cu121. The added primary GS seed-45 run used an RTX 3090 with zero workers; the other primary runs used the author-reported RTX 4090 environment with eight workers. See `environment/README.md` for environment setup and the scope of the code checks.

Prepare the datasets according to `data_manifests/README.md`. Use the official CIFAR train/test splits. GTSRB requires the recorded 26,640-image online-competition training subset and 12,630 test images; keep the supplied subset indexes.

From the repository root, a Free GS seed-41 run is:

```text
cd code/BackdoorBench
python -m backdoorbench.attack.wanet --yaml_path backdoorbench/config/attack/prototype/cifar10.yaml --bd_yaml_path backdoorbench/config/attack/wanet/default.yaml --model preactresnet18 --dataset cifar10 --dataset_path <data-root> --pratio 0.01 --gs_trigger 1 --random_seed 41 --num_workers 8 --device cuda --epochs 100
```

Replace `<data-root>` with the parent directory of the dataset folders. Use `--gs_trigger 0` for Fixed WaNet and `--gs_trigger 1 --gs_project 1` for Projected GS. The supplied YAML defaults use a 0.1 poisoning rate, so retain the explicit `--pratio 0.01` override for the 1% comparisons.

`backdoorbench.attack.wanet_rng_control` is the RNG-isolated evaluation entry. The PSNR-matched C2 scales for seeds 41/42/43 are 1.5830, 1.5654, and 1.8643. Keep `GS_RNG_CONTROL_STRICT` unset or set to `0` for those scaled controls. The optional strict whitelist requires unit magnitude; disabling that whitelist does not disable evaluation-RNG isolation. Run indexes and recorded settings are listed in `configs/README.md`.

## Rebuild the data figures

Use a separate plotting environment, described in `figures/README.md`:

```text
python -m pip install -r figures/requirements-plotting.txt
python figures/scripts/render_all.py
```

The scripts produce eleven data figures as PDF, SVG, and 1000-dpi RGB TIFF. Figure 1 is a schematic rather than a data plot and is not needed to execute the method. Plot inputs and recorded measurements remain separate from generated outputs.

## Validation and reuse

The distributed records have been checked for file integrity and statistical consistency. The retained training code passed CPU import, model-construction, forward-pass, and command-line checks. The eleven plotting outputs were regenerated from a relocated directory and matched the approved reference renderings. These checks do not represent a new independent training run of every experiment.

The original Free seed-42 strength-control endpoint is 90.00% ASR. The separate mixed-gradient diagnostic endpoint is 97.07%; it is used for that diagnostic. Individual reversals, clean-accuracy costs, and effective repairs remain in the data. `configs/run_reuse.json` identifies shared training runs.

The method builds on [WaNet](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release) and [BackdoorBench](https://github.com/SCLBD/BackdoorBench). File-level notices and upstream license texts are retained; see `LICENSE.md`. AI assistance (GLM-5.3, accessed through the ZCode editor) was used for language editing and document layout. The schematic in Fig. 1 was drafted with the same tool and refined manually by the authors. All training code, plotting code, and experimental records were produced and checked by the authors, who are responsible for the reported results.
