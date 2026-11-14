"""Final checkpoint audit; run after the 60-candidate search ends, on trusted local files."""
import argparse
import json
from pathlib import Path
import torch


def verify_run(folder):
    folder = Path(folder)
    result = json.loads((folder/'result.json').read_text())
    history = json.loads((folder/'history.json').read_text())
    best = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    last = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
    if not last['complete'] or last['epoch']+1 != result['epochs'] or history[-1]['epoch'] != last['epoch']:
        raise ValueError(f'{folder.name}: incomplete or inconsistent final checkpoint')
    if result['config'] != best['config'] or result['config'] != last['config']:
        raise ValueError(f'{folder.name}: checkpoint configuration differs')
    if result['metrics'] != best['metrics'] or last['best'] != best['metrics']['loss']:
        raise ValueError(f'{folder.name}: best checkpoint differs from reported metrics')
    rows = [r for r in history if r['epoch'] == best['epoch']]
    if len(rows) != 1 or any(rows[0][k] != v for k, v in best['metrics'].items()):
        raise ValueError(f'{folder.name}: best epoch is absent or disagrees with history')
    if 'optimizer' not in last or 'rng_cpu' not in last:
        raise ValueError(f'{folder.name}: missing recovery state')
    return dict(config=result['config'], metrics=result['metrics'],
                best_epoch=best['epoch'], final_epoch=last['epoch'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, default=Path('runs/paper'))
    p.add_argument('--output', type=Path, default=Path('outputs/final_training_audit.json'))
    args = p.parse_args()
    files = sorted(args.runs.glob('*/result.json'))
    # Fail before expensive checkpoint IO while the experiment is incomplete.
    if len(files) != 60:
        raise ValueError(f'Require 60 completed candidates; found {len(files)}')
    from nmr_watermark.training import candidates
    key = lambda c: (c['bits'], c['loss'], c['seed'], c['alpha'])
    expected = {key(c): c for c in candidates()}
    records = {p.parent.name: verify_run(p.parent) for p in files}
    if {key(r['config']) for r in records.values()} != set(expected):
        raise ValueError('Candidate identities do not cover the exact 60-model grid')
    for record in records.values():
        config = record['config']
        if config.get('pilot') or any(config.get(k) != v for k, v in expected[key(config)].items()):
            raise ValueError('Candidate does not use the frozen formal configuration')
    selected = json.loads((args.runs/'selected.json').read_text())
    if set(selected) != {'MSE256', 'NMR256', 'MSE512', 'NMR512'}:
        raise ValueError('Require all four selected models')
    for bits in (256, 512):
        subset = [(name, r) for name, r in records.items() if r['config']['bits'] == bits]
        _, mse = max(((n, r) for n, r in subset if r['config']['loss'] == 'mse'),
                            key=lambda pair: pair[1]['metrics']['snr'])
        selected_mse = records.get(Path(selected[f'MSE{bits}']['checkpoint']).parent.name)
        if selected_mse is None:
            raise ValueError(f'Missing selected MSE{bits} candidate')
        baseline_ber = selected_mse['metrics']['ber']
        eligible = [(n, r) for n, r in subset if r['config']['loss'] == 'nmr'
                    and r['metrics']['ber'] < baseline_ber]
        if not eligible:
            raise ValueError(f'No eligible NMR{bits}')
        _, nmr = min(eligible, key=lambda pair: pair[1]['metrics']['nmr'])
        # Accept ties only when the selected candidate itself satisfies the rule.
        for name, target in [(f'MSE{bits}', mse['metrics']['snr']), (f'NMR{bits}', nmr['metrics']['nmr'])]:
            path = Path(selected[name]['checkpoint'])
            path = path if path.is_absolute() else args.runs/path
            actual = records.get(path.parent.name)
            metric = 'snr' if name.startswith('MSE') else 'nmr'
            if (actual is None or path.resolve() != (args.runs/path.parent.name/'best.pt').resolve()
                    or actual['config']['bits'] != bits or actual['config']['loss'] != name[:-3].lower()
                    or actual['metrics'] != selected[name]['metrics'] or actual['metrics'][metric] != target
                    or (metric == 'nmr' and actual['metrics']['ber'] >= baseline_ber)):
                raise ValueError(f'{name}: selection rule or checkpoint mismatch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(runs=records, selected=selected), indent=2))
    print('Verified 60 checkpoint histories and four-model selection. No files were changed in runs/.')


if __name__ == '__main__':
    main()
