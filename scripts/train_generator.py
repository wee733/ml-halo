"""Resolve a local training run and enter HALO's original generator trainer."""
import argparse
import os
from pathlib import Path
import shlex
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/training/generator_5090.yaml')
    parser.add_argument('--data', type=Path, required=True, help='Stitched dataset containing data/, videos/ and meta/.')
    parser.add_argument('--run', required=True, help='New run name under HALO_RUNS_ROOT/generator.')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.run or Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('--run must be a single directory name')
    if any(value is not None and value < 1 for value in (args.batch_size, args.epochs)):
        parser.error('batch size and epochs must be positive')
    cfg = yaml.safe_load(args.config.read_text())
    data = args.data.expanduser().resolve()
    runs = Path(os.environ.get('HALO_RUNS_ROOT', ROOT / '.local/runs')).expanduser().resolve()
    out = runs / 'generator' / args.run
    cfg['data']['data_dir'] = str(data)
    cfg['data']['cache_dir'] = str(Path(os.environ.get('HALO_LOCAL_ROOT', ROOT / '.local')) / 'cache/generator')
    cfg['training']['checkpoint_dir'] = str(out)
    cfg['experiment_name'] = args.run
    if args.batch_size is not None:
        cfg['data']['batch_size'] = args.batch_size
    if args.epochs is not None:
        cfg['training']['num_epochs'] = args.epochs
    resolved = out / 'run_config.yaml'
    command = [sys.executable, str(ROOT / 'generator/scripts/train.py'), '--config', str(resolved)]
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        print(shlex.join(command))
        return
    missing = [name for name in ('data', 'videos', 'meta') if not (data / name).is_dir()]
    if missing:
        parser.error(f'Dataset is incomplete at {data}, missing {missing}. Generate and stitch data first.')
    out.mkdir(parents=True, exist_ok=False)
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    os.chdir(ROOT / 'generator')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
