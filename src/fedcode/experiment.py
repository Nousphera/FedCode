"""Launch native Flower experiments from the installed source checkout."""

import argparse
from contextlib import contextmanager
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import torch
from .config import DATASETS, METHODS, MODELS, defaults, paper_defaults, validate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, default='fedcode')
    p.add_argument('--dataset', choices=DATASETS, default='cifar10')
    p.add_argument('--model', choices=MODELS, help='Default: ResNet20 for CIFAR, YAMNet for Speech Commands')
    p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--nr-clients', type=int, default=10)
    for key in ('rounds', 'warmup-rounds', 'downstream-period'):
        p.add_argument(f'--{key}', type=int, help='Default: paper IID settings for the chosen model')
    for key in ('local-epochs', 'batch-size', 'num-clusters', 'upstream-period', 'seed',
                'data-seed', 'kmeans-seed', 'torch-threads', 'num-workers'):
        p.add_argument(f'--{key}', type=int, default=defaults()[key])
    for key in ('participation', 'dirichlet-alpha', 'learning-rate', 'registration-timeout', 'reply-timeout'):
        p.add_argument(f'--{key}', type=float, default=defaults()[key])
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path)
    p.add_argument('--parallel-clients', type=int, default=1)
    p.add_argument('--client-cpus', type=int, default=1)
    p.add_argument('--client-gpus', type=float)
    p.add_argument('--superlink', default='local')
    p.add_argument('--dry-run', action='store_true', help='Print resolved settings without training')
    return p


def build_config(args):
    model = args.model or ('yamnet' if args.dataset == 'speech_commands' else 'resnet20')
    cfg = defaults() | paper_defaults(model) | {'model': model}
    for key in cfg:
        attribute = {'num-clients': 'nr_clients', 'num-rounds': 'rounds'}.get(key, key.replace('-', '_'))
        value = getattr(args, attribute, None)
        if value is not None:
            cfg[key] = value
    cfg['data-root'] = str(Path(cfg['data-root']).expanduser().resolve())
    name = f"{args.method}-{model}-{args.dataset}-seed{args.seed}.json"
    cfg['output'] = str((args.output or Path('results') / name).expanduser().resolve())
    if args.parallel_clients < 1 or args.client_cpus < 1:
        raise ValueError('Client resources must be positive')
    return validate(cfg)


@contextmanager
def flower_app(cfg):
    """Generate Flower's descriptor temporarily; retain only source in the repo."""
    import tomli_w
    dist = distribution('fedcode')
    package = Path(__file__).resolve().parent
    assets = package.parent.parent
    if not (assets / 'setup.py').exists():
        assets = Path(sys.prefix) / 'share/fedcode'
    descriptor = {
        'build-system': {'requires': ['setuptools>=77', 'wheel'], 'build-backend': 'setuptools.build_meta'},
        'project': {'name': 'fedcode', 'version': dist.version, 'description': dist.metadata['Summary'],
                    'requires-python': dist.metadata['Requires-Python'], 'dependencies': dist.requires or [],
                    'license': 'MIT AND BSD-2-Clause',
                    'readme': {'text': dist.metadata['Summary'], 'content-type': 'text/plain'}},
        'tool': {'flwr': {'app': {'publisher': 'nousphera', 'components': {
            'serverapp': 'fedcode.server_app:app', 'clientapp': 'fedcode.client_app:app'}, 'config': cfg}}},
    }
    with TemporaryDirectory(prefix='fedcode-flower-') as directory:
        root = Path(directory)
        for source in package.rglob('*.py'):
            target = root / 'fedcode' / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for name in ('setup.py', 'LICENSE'):
            shutil.copy2(assets / name, root / name)
        (root / 'pyproject.toml').write_text(tomli_w.dumps(descriptor))
        yield root


def flower_command(args, cfg, config_file, application):
    executable = Path(sys.executable).with_name('flwr')
    if not executable.exists():
        raise RuntimeError('Install requirements.txt in the active environment first')
    gpus = args.client_gpus if args.client_gpus is not None else (1. if cfg['device'] == 'cuda' else 0.)
    if not math.isfinite(gpus) or gpus < 0 or (cfg['device'] == 'cuda' and gpus <= 0):
        raise ValueError('CUDA clients require positive --client-gpus')
    nodes = 1 if cfg['method'] == 'centralized' else cfg['num-clients']
    federation = (f'num-supernodes={nodes} client-resources-num-cpus={args.client_cpus} '
                  f'client-resources-num-gpus={gpus} init-args-num-cpus={args.client_cpus * args.parallel_clients}')
    return [str(executable), 'run', str(application), args.superlink, '--run-config', str(config_file),
            '--federation-config', federation, '--stream']


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        path = Path(cfg['output'])
        config_file = path.with_suffix('.run.toml')
        if args.dry_run:
            with flower_app(cfg) as application:
                print(json.dumps({'config': cfg, 'command': flower_command(args, cfg, config_file, application)}, indent=2))
            return
        from .data import load_datasets
        from .preparation import verify_dataset
        from .client_app import partitions
        verify_dataset(cfg['dataset'], cfg['data-root'])
        data, _ = load_datasets(cfg['dataset'], cfg['data-root'])
        partitions(data, cfg)
        if cfg['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; choose --device cpu or --device mps')
        if cfg['device'] == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable; choose --device cpu')
        for target in (path, path.with_suffix('.pt'), config_file):
            if target.exists():
                raise FileExistsError(f'Output exists: {target}')
        path.parent.mkdir(parents=True, exist_ok=True)
        import tomli_w
        with config_file.open('xb') as stream:
            tomli_w.dump(cfg, stream)
        env = os.environ.copy()
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        env.setdefault('FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION', '1')
        with flower_app(cfg) as application:
            command = flower_command(args, cfg, config_file, application)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True, env=env)
        if not path.exists() or not path.with_suffix('.pt').exists():
            raise RuntimeError('Flower ended without saved results; inspect the run logs')
        print(f'Saved {path}')
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f'Experiment failed: {exc}\n')


if __name__ == '__main__':
    main()
