"""Check source parsing, imports, CPU model construction, and CLI help.

This check uses synthetic inputs. It does not load checkpoints, download data,
train a model, or validate the reported attack and defense endpoints.
"""

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-path', action='append', default=[],
                        help='Additional dependency directory; may be repeated.')
    parser.add_argument('--include-defense', action='store_true',
                        help='Also import the twelve archived defense modules.')
    parser.add_argument('--skip-models', action='store_true',
                        help='Skip CPU forward passes for the three study architectures.')
    parser.add_argument('--report', type=Path, help='Write the check results as JSON.')
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parent / 'BackdoorBench'
    extra = [str(Path(p).resolve()) for p in args.extra_path]
    sys.path[:0] = [str(code_root)] + extra
    sys.dont_write_bytecode = True
    os.environ.setdefault('MPLBACKEND', 'Agg')
    started = time.time()
    result = {'python': sys.version, 'platform': platform.platform(),
              'code_root': str(code_root), 'dependency_directories': extra,
              'checks': [], 'training_performed': False,
              'checkpoints_loaded': False, 'dataset_downloaded': False}

    def check(name, fn):
        t = time.time()
        entry = {'name': name}
        try:
            entry['details'] = fn()
            entry['passed'] = True
        except Exception as error:
            entry.update(passed=False, error=repr(error), traceback=traceback.format_exc())
        entry['seconds'] = round(time.time() - t, 3)
        result['checks'].append(entry)

    def parse_sources():
        files = sorted(code_root.rglob('*.py'))
        for file in files:
            ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
        return {'python_files': len(files)}

    check('parse_python_sources', parse_sources)
    for name in ['numpy', 'torch', 'torchvision', 'yaml', 'pandas', 'matplotlib',
                 'numba', 'kornia', 'imageio', 'tqdm']:
        def imported(name=name):
            module = importlib.import_module(name)
            return {'version': getattr(module, '__version__', None),
                    'path': getattr(module, '__file__', None)}
        check('dependency:' + name, imported)

    for name in ['backdoorbench.attack.wanet', 'backdoorbench.attack.wanet_rng_control',
                 'backdoorbench.utils.aggregate_block.model_trainer_generate',
                 'backdoorbench.models.wrn_cifar',
                 'backdoorbench.utils.defense_utils.npd.utils',
                 'dataset.GTSRB', 'backdoorbench.dataset.GTSRB']:
        check('import:' + name, lambda name=name: {'path': importlib.import_module(name).__file__})

    def construct_models():
        import torch
        from backdoorbench.utils.aggregate_block.model_trainer_generate import generate_cls_model
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        outcomes = []
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(42)
            inputs = torch.zeros(2, 3, 32, 32)
            for name in ['preactresnet18', 'wrn34_10', 'resnext29']:
                model = generate_cls_model(name, num_classes=10, image_size=32).cpu().eval()
                output = model(inputs)
                assert tuple(output.shape) == (2, 10), (name, output.shape)
                assert torch.isfinite(output).all().item(), name
                outcomes.append({'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
                                 'input_shape': list(inputs.shape), 'output_shape': list(output.shape),
                                 'finite_output': True, 'device': 'cpu'})
                del model, output
        return outcomes

    if not args.skip_models:
        check('cpu_model_construction_and_forward', construct_models)

    def help_entry(module):
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(code_root)] + extra + [env.get('PYTHONPATH', '')])
        proc = subprocess.run([sys.executable, '-B', '-m', module, '--help'], cwd=code_root,
                              env=env, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=90)
        assert proc.returncode == 0, proc.stderr[-5000:]
        for flag in ['--pratio', '--random_seed', '--gs_trigger']:
            assert flag in proc.stdout, flag
        return {'returncode': proc.returncode, 'stdout': proc.stdout,
                'stderr': proc.stderr, 'command': [sys.executable, '-B', '-m', module, '--help']}

    for module in ['backdoorbench.attack.wanet', 'backdoorbench.attack.wanet_rng_control']:
        check('help:' + module, lambda module=module: help_entry(module))

    if args.include_defense:
        for name in ['abl', 'anp', 'clp', 'fp', 'ft', 'i_bau', 'nad', 'nc',
                     'npd_ours', 'sentinet_ours', 'spectral', 'strip_ours']:
            module = 'backdoorbench.defense.' + name
            check('import:' + module, lambda module=module: {'path': importlib.import_module(module).__file__})

    result['all_passed'] = all(x['passed'] for x in result['checks'])
    result['seconds'] = round(time.time() - started, 3)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(result['checks']),
                      'failed': [x['name'] for x in result['checks'] if not x['passed']],
                      'seconds': result['seconds']}, ensure_ascii=False))
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
