"""Inspect formal run metadata and memory-mapped checkpoints without training."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from nmr_watermark.training import candidates


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inspect(folder, warmup=False):
    cfg = read(folder/'config.json')
    history = read(folder/'history.json') if (folder/'history.json').exists() else []
    warm = [0.5] if cfg['loss'] == 'mse' else [1e-4, .01, .1]
    errors = []
    if history:
        start = 0 if warmup else len(warm)
        if [r['epoch'] for r in history] != list(range(start, history[-1]['epoch']+1)):
            errors.append('Non-contiguous epoch history')
    best, stale = math.inf, 0
    for row in history:
        alpha = warm[row['epoch']] if row['epoch'] < len(warm) else cfg['alpha']
        if row['alpha'] != alpha:
            errors.append('Alpha schedule differs')
        expected = alpha*row['distortion']+(1-alpha)*row['bce']
        if not math.isclose(row['loss'], expected, rel_tol=2e-6, abs_tol=1e-7):
            errors.append('Logged validation objective differs from weighted loss')
        if row['epoch'] >= len(warm):
            if row['loss'] < best:
                best, stale = row['loss'], 0
            else:
                stale += 1
            if stale >= 2 and row != history[-1]:
                errors.append('Training continued after two non-improving formal epochs')
    result = {'config': cfg, 'logged_epochs': len(history),
              'epoch_indices': [r['epoch'] for r in history],
              'alphas': [r['alpha'] for r in history], 'errors': errors}
    last_path = folder/'last.pt'
    if not last_path.exists():
        result['checkpoint'] = 'not_yet_written'
        return result
    # Trusted local checkpoint; mmap keeps large parameter/Adam storages off the heap.
    last = torch.load(last_path, map_location='cpu', weights_only=False, mmap=True)
    result['checkpoint'] = {'epoch': last['epoch'], 'complete': last['complete'],
                            'stale': last['stale'], 'best_loss': last['best']}
    if last['config'] != cfg:
        errors.append('Checkpoint config differs from config.json')
    groups = last['optimizer']['param_groups']
    result['optimizer'] = [{k: g[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad')}
                           for g in groups]
    for g in groups:
        if g['lr'] != 1e-4 or g['betas'] != (.9, .999) or g['eps'] != 1e-8 or g['weight_decay'] != 0:
            errors.append('Optimizer differs from frozen project defaults')
    steps = [float(s['step'].item()) for s in last['optimizer']['state'].values()]
    result['adam_step_range'] = [min(steps), max(steps)]
    expected_steps = (last['epoch']+1)*math.ceil(261120/cfg['batch_size'])
    result['expected_steps_from_full_data'] = expected_steps
    if min(steps) != expected_steps or max(steps) != expected_steps:
        errors.append('Adam step counts do not match full-data epochs')
    result['floating_model_dtypes'] = sorted({str(t.dtype) for t in last['model'].values() if t.is_floating_point()})
    result['torch_rng_saved'] = 'rng_cpu' in last and 'rng_cuda' in last
    result['resume_source'] = 'last.pt contains model, optimizer, epoch, best, stale and torch RNG'
    if not warmup and last['complete']:
        final = read(folder/'result.json')
        selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=False, mmap=True)
        result['selected_epoch'] = selected['epoch']
        if final['metrics'] != selected['metrics'] or selected['metrics']['loss'] != best:
            errors.append('Saved best checkpoint is not the minimum formal validation loss')
        if selected['epoch'] != min(history, key=lambda r:r['loss'])['epoch']:
            errors.append('Saved best epoch differs from history')
        if last['stale'] != stale or not (stale >= 2 or last['epoch']==29):
            errors.append('Completion does not match early stopping or 30-epoch limit')
        if final['epochs'] != last['epoch']+1 or last['epoch'] >= 30:
            errors.append('Total epoch count differs')
        del selected
    elif history and last['epoch'] != history[-1]['epoch']:
        result['snapshot_note'] = 'Active history may be written before its atomic last checkpoint'
    del last
    return result


def main():
    torch.set_num_threads(2)
    root = Path('runs/paper')
    expected = list(candidates())
    queue = read(root/'queue.json')
    assert queue == expected
    source = Path('nmr_watermark/training.py')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = {'checked_at': datetime.now().isoformat(timespec='seconds'),
              'queue_exactly_matches_60_frozen_candidates': True,
              'training_source_sha256': before, 'runs': {}, 'warmups': {}}
    for folder in sorted(root.iterdir()):
        if folder.name == 'warmup' or not (folder/'config.json').exists():
            continue
        result = inspect(folder)
        if not any(all(result['config'].get(k)==v for k,v in c.items()) for c in expected):
            result['errors'].append('Actual run config is outside frozen candidate grid')
        output['runs'][folder.name] = result
    for folder in sorted((root/'warmup').iterdir()):
        output['warmups'][folder.name] = inspect(folder, True)
    output['completed'] = sum(r['checkpoint'].get('complete', False)
        for r in output['runs'].values() if isinstance(r['checkpoint'], dict))
    output['errors'] = {n:r['errors'] for group in ('runs','warmups')
                        for n,r in output[group].items() if r['errors']}
    output['source_unchanged'] = before == hashlib.sha256(source.read_bytes()).hexdigest()
    dest = Path('outputs/paper_audit_20261021/training_runtime.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:output[k] for k in ('checked_at','completed','errors','source_unchanged')}, ensure_ascii=False))
    print('Inspected', len(output['runs']), 'started candidates and', len(output['warmups']), 'warmups')


if __name__ == '__main__':
    main()
