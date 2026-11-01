# Training and code-check environment

The training records identify PyTorch 2.4.1+cu121 and torchvision 0.19.1+cu121. The added seed-45 note identifies Python 3.12.10 and NumPy 1.26.4. Install a PyTorch/torchvision build appropriate to the intended CPU or CUDA 12.1 machine, then install `requirements-core.txt`. Defense modules also require `requirements-defense.txt`.

The retained dependency lists describe the CPU compatibility environment used for import and model-construction checks. They are separate from `figures/requirements-plotting.txt`; keep plotting requirements in another environment.

From the repository root, check imports, command-line entries, and model construction:

```text
python code/code_import_check.py --include-defense --report results/recomputed/code_checks.json
```

These checks use synthetic 2×3×32×32 inputs for the three study architectures. They do not train models, download datasets, or evaluate attack endpoints. Training requires the prepared datasets and the recorded run settings.
