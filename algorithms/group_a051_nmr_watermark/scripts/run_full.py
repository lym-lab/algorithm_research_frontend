"""Resume the verified-data -> search -> evaluation pipeline.

Requires explicit frozen diagnostics; never silently promotes pilot weights.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import torch
from nmr_watermark.data import prepare_fma, prepare_rwc
from nmr_watermark.training import cache_masks, search, select
from nmr_watermark.evaluation import evaluate, figures
from scripts.windows_training_qos import enable_training_performance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matlab', default='D:/Program Files/MATLAB/R2025a/bin/matlab.exe')
    args = parser.parse_args()
    try:
        print(json.dumps({'training_qos': enable_training_performance()}), flush=True)
    except OSError as exc:
        print(json.dumps({'training_qos_warning': str(exc)}), flush=True)
    torch.set_num_threads(4)
    root = Path.cwd()
    status_path = root/'outputs/pipeline_status.json'
    status_path.parent.mkdir(exist_ok=True)

    def status(stage, **values):
        result = dict(stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'), **values)
        status_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)

    try:
        frozen = json.loads(Path('outputs/frozen_config.json').read_text())
        if not frozen.get('approved_diagnostics') or not frozen.get('stft_normalized'):
            raise RuntimeError('Formal configuration has not passed diagnostic gates')
        for key in ('fma', 'rwc'):
            while not Path(f'data/{key}.verified.json').exists():
                status('waiting_for_verified_data', dataset=key)
                time.sleep(30)
        status('building_manifests')
        manifests = Path('data/manifests/full')
        if not (manifests/'summary.json').exists():
            prepare_fma('data/fma_medium', manifests)
        if not Path('data/manifests/rwc.json').exists():
            prepare_rwc('data/RWC-G', 'data/manifests/rwc.json')
        for split in ('train', 'validation'):
            status('mask_cache', split=split)
            cache_masks(manifests/f'{split}.json', Path('data/cache/full')/split,
                        'cuda', frozen['batch_size'], normalized=True)
        status('training_60_candidates')
        for name, expected in frozen['implementation_sha256'].items():
            if hashlib.sha256((Path('nmr_watermark')/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Implementation changed after configuration freeze: '+name)
        search(manifests, 'data/cache/full', 'runs/paper', batch_size=frozen['batch_size'])
        status('selecting_four_models')
        selected = select('runs/paper')
        matlab_env = dict(os.environ, MATLAB_PREFDIR=str(root/'.cache/matlab'))
        for name, record in selected.items():
            out = root/'outputs/evaluation'/name
            status('objective_evaluation', model=name)
            if not (out/'summary.json').exists():
                evaluate(record['checkpoint'], 'data/manifests/rwc.json', out, 'cuda', peaq_audio=True)
            if not (out/'peaq.csv').exists():
                def quote(path):
                    return str(path).replace('\\', '/').replace("'", "''")
                command = f"addpath('{quote(root/'scripts')}'); evaluate_peaq('{quote(out/'peaq_pairs.csv')}','{quote(out/'peaq.csv')}','{quote(root/'vendor/PEAQ_matlab/PQevalAudio')}');"
                subprocess.run([args.matlab, '-batch', command], check=True, env=matlab_env)
        figures('outputs/evaluation')
        subprocess.run([sys.executable, '-m', 'scripts.build_results'], check=True)
        subprocess.run([sys.executable, '-m', 'scripts.plot_case'], check=True)
        subprocess.run([sys.executable, 'scripts/export_models.py', 'runs/paper/selected.json', 'weights'], check=True)
        status('objective_pipeline_finished', remaining=['report_and_slide_finalization', 'cloud_deployment', 'optional_human_listening'])
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
