"""Check GTSRB labels and hashes, with optional image-file validation."""

import argparse
import csv
import json
from pathlib import Path
import re

from evidence_io import ROOT, read_csv, sha256, write_json


def annotation_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=';'))


def verify(image_root=None):
    base = ROOT / 'data_manifests/gtsrb'
    archived = {}
    for line in (base / 'original_server_sha256.txt').read_text(
        encoding='utf-8-sig',
    ).splitlines():
        match = re.match(r'^([a-f0-9]{64})\s+.*?/gtsrb/(.+)$', line)
        if match:
            archived[match[2]] = match[1]
    label_files = list((base / 'labels/Train').glob('*/GT-*.csv'))
    assert len(label_files) == 43
    small_files = label_files + [
        base / 'labels/Train/Readme.txt',
        base / 'labels/Test/GT-final_test.csv',
        base / 'labels/Test/GT-final_test.test.csv',
    ]
    for path in small_files:
        relative = path.relative_to(base / 'labels').as_posix()
        assert sha256(path) == archived[relative], relative
    totals = {}
    checked_images = 0
    expected_paths = {p.relative_to(base / 'labels').as_posix() for p in small_files}
    for split in ('training', 'test'):
        index = read_csv(base / f'{split}_index.csv')
        assert len(index) == (26640 if split == 'training' else 12630)
        assert len({r['relative_path'] for r in index}) == len(index)
        assert {int(r['ClassId']) for r in index} == set(range(43))
        observed = {}
        files = (
            label_files if split == 'training'
            else [base / 'labels/Test/GT-final_test.csv']
        )
        for file in files:
            directory = file.parent.relative_to(base / 'labels')
            for row in annotation_rows(file):
                relative = (directory / row['Filename']).as_posix()
                observed[relative] = (
                    int(row['ClassId']), int(row['Width']), int(row['Height']),
                )
        assert len(observed) == len(index)
        for row in index:
            relative = row['relative_path']
            expected_paths.add(relative)
            assert observed[relative] == (
                int(row['ClassId']), int(row['Width']), int(row['Height']),
            )
            assert row['sha256_archived_server'] == archived[relative]
            if image_root:
                image = image_root / relative
                if not image.is_file():
                    raise FileNotFoundError(image)
                if sha256(image) != row['sha256_archived_server']:
                    raise ValueError(
                        f'Image differs from the archived server hash: {image}',
                    )
                checked_images += 1
        totals[split] = len(index)
    assert set(archived) == expected_paths, (
        'The server hash inventory and supplied subset paths differ.'
    )
    return {
        'label_files_matching_archived_hashes': len(small_files),
        'training_images_indexed': totals['training'],
        'test_images_indexed': totals['test'],
        'classes': 43,
        'unique_paths_and_labels_match': True,
        'archived_hash_entries_covering_exactly_the_indexed_subset': len(archived),
        'images_checked_now': checked_images,
        'image_root': str(image_root) if image_root else None,
        'image_hash_scope': (
            'Current file hashes compared with the archived server inventory'
            if image_root else
            'Historical image hashes indexed; no image bytes supplied or '
            'checked in this run'
        ),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--gtsrb-root', type=Path,
        help=(
            'Dataset directory containing Train/ and Test/. '
            'Images stay outside this package.'
        ),
    )
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results/recomputed/gtsrb_checks.json',
    )
    args = parser.parse_args()
    result = verify(args.gtsrb_root)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
