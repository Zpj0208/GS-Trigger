"""Read archived tabular and logged records without importing training code."""

from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )


def logged_dicts(path):
    text = Path(path).read_text(encoding='utf-8-sig', errors='replace')
    start = 0
    while True:
        opening = text.find('{', start)
        if opening < 0:
            return
        depth, quote, escape, end = 0, None, False, opening
        for end in range(opening, len(text)):
            char = text[end]
            if quote:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    break
        if depth:
            return
        try:
            record = ast.literal_eval(text[opening:end+1])
        except (ValueError, SyntaxError):
            record = None
        if isinstance(record, dict):
            yield text.count('\n', 0, opening) + 1, record
        start = end + 1


def read_training_log(path):
    args, epochs = None, {}
    for line, record in logged_dicts(path):
        if (
            args is None and 'model' in record
            and 'dataset' in record and 'save_path' in record
        ):
            args = record
        if all(key in record for key in ('epoch', 'test_asr', 'test_acc')):
            epoch = int(record['epoch'])
            values = (float(record['test_asr']), float(record['test_acc']))
            if epoch in epochs and epochs[epoch]['values'] != values:
                raise ValueError(
                    f'Conflicting training metrics at epoch {epoch}: {path}',
                )
            epochs[epoch] = {'values': values, 'source_line': line}
    if args is None:
        raise ValueError(f'No logged training configuration: {path}')
    return args, epochs
