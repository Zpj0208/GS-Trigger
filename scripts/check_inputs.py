"""Verify repository inputs, configuration links, and GTSRB annotations."""

import argparse
import json
from pathlib import Path
from evidence_io import ROOT, read_csv, sha256, write_json
from verify_gtsrb import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/recomputed/input_checks.json"
    )
    args = parser.parse_args()
    records = read_csv(ROOT / "data_manifests/package_inputs_sha256.csv")
    for row in records:
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert path.stat().st_size == int(row["bytes"]), row["path"]
        assert sha256(path) == row["sha256"], row["path"]
    references = 0
    for name, keys in [
        ("primary_run_manifest.csv", ("log", "config", "epoch_csv")),
        ("isolated_control_manifest.csv", ("csv", "manifest", "config")),
        ("log_configuration_index.csv", ("log", "configuration_json")),
    ]:
        for row in read_csv(ROOT / "configs" / name):
            for key in keys:
                assert (ROOT / row[key]).is_file(), (name, key, row[key])
                references += 1
    source_records = read_csv(ROOT / "data_manifests/source_files.csv")
    for row in source_records:
        assert sha256(ROOT / row["package_path"]) == row["sha256"], row["package_path"]
    result = {
        "input_files_checked": len(records),
        "input_hash_failures": 0,
        "configuration_references_checked": references,
        "original_source_files_checked": len(source_records),
        "GTSRB": verify(),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
