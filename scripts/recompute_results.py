"""Recompute primary and isolated-control results from archived records, on CPU."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics

from evidence_io import ROOT, read_csv, read_training_log, sha256, write_csv, write_json


def close(actual, expected, label, tolerance=1e-9):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f'{label}: {actual} != {expected}')


def summarize(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summaries = []
    for group, observations in sorted(groups.items()):
        result = dict(zip(group_keys, group))
        result['n_training_runs'] = len(observations)
        for metric in ('ASR_pct', 'CACC_pct'):
            values = [float(row[metric]) for row in observations]
            result[metric + '_mean'] = statistics.mean(values)
            result[metric + '_sample_SD'] = statistics.stdev(values)
        summaries.append(result)
    return summaries


def primary(output):
    manifest = read_csv(ROOT / 'configs/primary_run_manifest.csv')
    assert len(manifest) == 22
    assert len({(r['model'], r['method'], r['seed']) for r in manifest}) == 22
    endpoints = []
    for row in manifest:
        log = ROOT / row['log']
        assert sha256(log) == row['log_sha256'], log
        args, epochs = read_training_log(log)
        assert sorted(epochs) == list(range(100)), log
        assert args['pratio'] == 0.01 and args['k'] == 4
        assert (
            args['model'] == row['model']
            and int(args['random_seed']) == int(row['seed'])
        )
        csv_rows = read_csv(ROOT / row['epoch_csv'])
        assert [int(r['epoch']) for r in csv_rows] == list(range(100))
        for r in csv_rows:
            e = int(r['epoch'])
            close(
                float(r['test_asr']), epochs[e]['values'][0],
                f'{row["run_id"]} epoch {e} ASR',
            )
            close(
                float(r['test_acc']), epochs[e]['values'][1],
                f'{row["run_id"]} epoch {e} CACC',
            )
        asr, cacc = epochs[99]['values']
        close(asr * 100, float(row['ASR_pct']), row['run_id'] + ' final ASR')
        close(cacc * 100, float(row['CACC_pct']), row['run_id'] + ' final CACC')
        endpoints.append(
            {
                key: row[key] for key in (
                    'run_id', 'model', 'method', 'seed',
                    'gpu_author_reported', 'num_workers',
                )
            }
            | {'ASR_pct': asr * 100, 'CACC_pct': cacc * 100}
        )
    assert not any(r['run_id'] == 'hR7x' for r in endpoints)
    assert next(r for r in endpoints if r['run_id'] == 'mopR')['ASR_pct'] > 99
    summary = summarize(endpoints, ('model', 'method'))
    # The reference CSV comes from earlier log-based figure preparation, not Word.
    reference = ROOT / 'figures/source_data/Fig_02_primary_exact_statistics.csv'
    with reference.open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.reader(stream))[3:]
    expected = {(r[0], r[1]): r for r in original if r and r[0]}
    for row in summary:
        ref = expected[(row['model'], row['method'])]
        assert row['n_training_runs'] == int(ref[2])
        for key, col in [
            ('ASR_pct_mean', 3), ('ASR_pct_sample_SD', 4),
            ('CACC_pct_mean', 6), ('CACC_pct_sample_SD', 7),
        ]:
            close(
                row[key], float(ref[col]),
                f'{row["model"]}/{row["method"]}/{key}',
            )
    common = [
        r for r in endpoints
        if r['model'] == 'preactresnet18' and int(r['seed']) in (41, 42, 43, 44)
    ]
    common_summary = summarize(common, ('model', 'method'))
    write_csv(output / 'primary_endpoints.csv', endpoints)
    write_csv(output / 'primary_mean_sample_sd.csv', summary)
    write_csv(output / 'preact_common_setting_four_pairs.csv', common_summary)
    return {
        'runs': len(endpoints),
        'verified_training_epochs': len(endpoints) * 100,
        'summary': summary,
        'four_common_setting_pairs': common_summary,
        'reference_statistics_match': True,
        'old_10pct_hR7x_excluded': True,
        'replacement_1pct_mopR_included': True,
    }


def isolated_controls(output):
    manifest = read_csv(ROOT / 'configs/isolated_control_manifest.csv')
    assert len(manifest) == 9
    endpoints = []
    for row in manifest:
        data = read_csv(ROOT / row['csv'])
        assert [int(float(r['epoch'])) for r in data] == list(range(100)), row['csv']
        last = data[-1]
        endpoints.append({
            'condition': row['condition'], 'seed': int(row['seed']),
            'ASR_pct': float(last['test_asr']) * 100,
            'CACC_pct': float(last['test_acc']) * 100,
            'ASR_last10_pct': statistics.mean(
                float(r['test_asr']) * 100 for r in data[-10:]
            ),
            'CACC_last10_pct': statistics.mean(
                float(r['test_acc']) * 100 for r in data[-10:]
            ),
        })
    summary = summarize(endpoints, ('condition',))
    for item in summary:
        runs = [r for r in endpoints if r['condition'] == item['condition']]
        for metric in ('ASR_last10_pct', 'CACC_last10_pct'):
            item[metric + '_mean'] = statistics.mean(r[metric] for r in runs)
            item[metric + '_sample_SD'] = statistics.stdev(r[metric] for r in runs)
    expected = read_csv(ROOT / 'results/validated/joint_control_training_metrics.csv')
    for row in expected:
        actual = next(
            r for r in endpoints
            if r['condition'] == row['method'] and r['seed'] == int(row['seed'])
        )
        for a, b in [
            ('ASR_pct', 'final_ASR_pct'), ('CACC_pct', 'final_CACC_pct'),
            ('ASR_last10_pct', 'last10_ASR_pct'),
            ('CACC_last10_pct', 'last10_CACC_pct'),
        ]:
            close(actual[a], float(row[b]), f'{row["method"]}/{row["seed"]}/{a}')
    paired = []
    for seed in (41, 42, 43):
        free = next(
            r for r in endpoints if r['condition'] == 'Free' and r['seed'] == seed
        )
        matched = next(
            r for r in endpoints
            if r['condition'] == 'Matched Fixed' and r['seed'] == seed
        )
        paired.append({
            'seed': seed,
            'Free_minus_matched_ASR_pp': free['ASR_pct'] - matched['ASR_pct'],
        })
    mean_gain = statistics.mean(r['Free_minus_matched_ASR_pp'] for r in paired)
    close(
        round(mean_gain, 2), 3.45,
        'joint-control mean difference rounded to two decimals',
    )
    write_csv(output / 'isolated_control_endpoints.csv', endpoints)
    write_csv(output / 'isolated_control_mean_sample_sd.csv', summary)
    write_csv(output / 'isolated_control_paired_differences.csv', paired)
    return {
        'distinct_training_runs': 9,
        'Free_runs_reused_in_two_comparisons': True,
        'summary': summary,
        'Free_minus_matched_mean_ASR_pp': mean_gain,
        'prior_verified_endpoints_match': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir', type=Path, default=ROOT / 'results/recomputed',
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'primary': primary(args.output_dir),
        'isolated_controls': isolated_controls(args.output_dir),
        'statistical_unit': (
            'independent training runs within each condition; sample SD uses n-1'
        ),
        'execution': (
            'Archived-result recalculation only. No model is loaded '
            'and no training is started.'
        ),
    }
    write_json(args.output_dir / 'result_checks.json', report)
    for row in report['primary']['summary']:
        print(
            f"{row['model']:18s} {row['method']:6s} n={row['n_training_runs']} "
            f"ASR={row['ASR_pct_mean']:.2f}±{row['ASR_pct_sample_SD']:.2f}% "
            f"CACC={row['CACC_pct_mean']:.2f}±{row['CACC_pct_sample_SD']:.2f}%"
        )
    print(
        f"Joint control Free-minus-matched ASR: "
        f"{report['isolated_controls']['Free_minus_matched_mean_ASR_pp']:.6f} "
        f"percentage points"
    )
    print('Archived-result checks passed.')


if __name__ == '__main__':
    main()
