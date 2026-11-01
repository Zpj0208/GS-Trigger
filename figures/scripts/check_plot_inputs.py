"""Verify the plotting tables and saved image/seed inputs with SHA-256."""

import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "inputs_sha256.csv").open(encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream))
    for entry in entries:
        path = root / entry["path"]
        data = path.read_bytes()
        assert len(data) == int(entry["bytes"]), entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    index = json.loads((root / "plot_data/data_index.json").read_text("utf-8"))
    for entry in index:
        with (root / entry["file"]).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == entry["columns"], entry["file"]
            assert len(list(reader)) == entry["rows"], entry["file"]
    print(json.dumps({"files_checked": len(entries), "tables_checked": len(index),
                      "hash_failures": 0}))


if __name__ == "__main__":
    main()
