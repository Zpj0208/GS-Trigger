"""Verify the saved visualisation arrays from their clean inputs and seed records."""

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "visual_inputs",
    )
    args = parser.parse_args()
    with np.load(
        args.input_dir / "reconstructed_inputs_float32.npz", allow_pickle=False
    ) as data:
        expected = data["rgb"].copy()
        difference = data["difference_unclipped"].copy()
        displayed = data["displayed_difference"].copy()
        conditions = data["conditions"].tolist()
    with (args.input_dir / "seed_records_used.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        seed_rows = {row["condition"]: row for row in csv.DictReader(stream)}
    clean = torch.from_numpy(expected[:, 0].copy()).permute(0, 3, 1, 2).contiguous()
    axis = torch.linspace(-1, 1, 32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((x, y), dim=2)[None]
    outputs = [clean]
    with torch.no_grad():
        for condition in conditions[1:]:
            values = np.fromstring(
                seed_rows[condition]["32_values"], sep=" ", dtype=np.float32
            )
            assert values.size == 32 and np.isfinite(values).all()
            seed = torch.from_numpy(values.reshape(1, 2, 4, 4).copy())
            noise = F.interpolate(
                seed, size=(32, 32), mode="bicubic", align_corners=True
            ).permute(0, 2, 3, 1)
            grid = (identity + 0.5 * noise / 32).clamp(-1, 1)
            outputs.append(
                F.grid_sample(
                    clean,
                    grid.expand(clean.shape[0], -1, -1, -1),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            )
    actual = torch.stack(outputs, dim=1).permute(0, 1, 3, 4, 2).numpy()
    actual_difference = np.abs(actual - actual[:, 0:1]).mean(axis=-1) * 10
    actual_displayed = np.minimum(actual_difference, 1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_difference, difference)
    np.testing.assert_array_equal(actual_displayed, displayed)
    print(
        "Verified all 12 RGB arrays and their displayed differences exactly; no training performed."
    )


if __name__ == "__main__":
    main()
