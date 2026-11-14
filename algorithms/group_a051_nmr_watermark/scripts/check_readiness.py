"""Read-only evidence audit. Does not load checkpoints, run training or select models."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifests(train, validation, test):
    groups = []
    for label, rows, expected in [('train', train, 261120), ('validation', validation, 32640),
                                   ('rwc', test, 3000)]:
        if len(rows) != expected:
            raise ValueError(f'{label}: expected {expected} segments')
        tracks = defaultdict(list)
        for row in rows:
            if row['length'] != 48640 or row['start'] < 0:
                raise ValueError(f'{label}: invalid segment')
            tracks[row['track_id']].append(row)
        per_track = 30 if label == 'rwc' else 24
        if any(len(items) != per_track for items in tracks.values()):
            raise ValueError(f'{label}: invalid segments per track')
        for items in tracks.values():
            if len({(r['path'], r['start']) for r in items}) != len(items):
                raise ValueError(f'{label}: duplicate segment')
            if label != 'rwc':
                if len({r['path'] for r in items}) != 1:
                    raise ValueError(f'{label}: inconsistent track path')
                starts = sorted(r['start'] for r in items)
                if any(b-a != 48640 for a, b in zip(starts, starts[1:])):
                    raise ValueError(f'{label}: non-contiguous FMA segments')
        groups.append(set(tracks))
    if groups[0] & groups[1]:
        raise ValueError('FMA track leakage')
    if groups[2] != {f'RWC_G{i:03}' for i in range(1, 101)}:
        raise ValueError('RWC requires all 100 works')
    return dict(train_tracks=len(groups[0]), validation_tracks=len(groups[1]),
                rwc_tracks=len(groups[2]), segments=[len(train), len(validation), len(test)])


def audit(root):
    root = Path(root)
    checks = []

    def check(name, operation):
        try:
            evidence = operation()
            checks.append(dict(name=name, status='verified', evidence=evidence))
        except FileNotFoundError as exc:
            checks.append(dict(name=name, status='pending', detail=str(exc)))
        except (ValueError, KeyError, OSError) as exc:
            checks.append(dict(name=name, status='review', detail=str(exc)))

    def frozen():
        config = read(root/'outputs/frozen_config.json')
        for name, value in config['implementation_sha256'].items():
            if digest(root/'nmr_watermark'/name) != value:
                raise ValueError('Frozen source changed: '+name)
        return config['implementation_sha256']
    check('frozen_sources', frozen)

    def manifests():
        paths = [root/'data/manifests/full/train.json', root/'data/manifests/full/validation.json',
                 root/'data/manifests/rwc.json']
        rows = [read(p) for p in paths]
        evidence = check_manifests(*rows)
        missing = [p for p in {r['path'] for part in rows for r in part} if not Path(p).is_file()]
        if missing:
            raise ValueError(f'{len(missing)} missing source audio paths')
        evidence['sha256'] = {str(p.relative_to(root)): digest(p) for p in paths}
        return evidence
    check('data_manifests', manifests)

    def caches():
        import numpy as np
        result = {}
        for split, count in [('train', 261120), ('validation', 32640)]:
            folder = root/'data/cache/full'/split
            meta = read(folder/'complete.json')
            if (meta['manifest'] != digest(root/f'data/manifests/full/{split}.json')
                    or meta['psycho'] != digest(root/'nmr_watermark/psychoacoustics.py')
                    or meta['normalized'] is not True):
                raise ValueError(f'{split}: cache identity differs')
            array = np.load(folder/'masks.npy', mmap_mode='r')
            if array.shape != (count, 109, 96) or array.dtype != np.float32:
                raise ValueError(f'{split}: invalid cache shape/dtype')
            result[split] = dict(shape=list(array.shape), dtype=str(array.dtype))
            del array
        return result
    check('mask_cache_headers_and_identity', caches)

    def candidates():
        expected = {(bits, loss, seed, alpha) for bits in (256, 512)
                    for loss, alphas in [('mse', [.8, .9, .92, .95, .98]), ('nmr', [.1, .2, .3, .4, .5])]
                    for seed in (2026, 2027, 2028) for alpha in alphas}
        key = lambda c: (c['bits'], c['loss'], c['seed'], c['alpha'])
        queue = read(root/'runs/paper/queue.json')
        if len(queue) != 60 or {key(c) for c in queue} != expected:
            raise ValueError('Invalid 60-model queue')
        keys, counts = [], Counter()
        for path in (root/'runs/paper').glob('*/result.json'):
            value = read(path)
            config = value['config']
            keys.append(key(config))
            if config.get('pilot') or not config.get('stft_normalized') or config['batch_size'] != 16:
                raise ValueError(f'{path.parent.name}: unexpected formal config')
            for file in ('best.pt', 'last.pt', 'history.json'):
                if not (path.parent/file).is_file():
                    raise ValueError(f'{path.parent.name}: missing {file}; may be writing, retry later')
            counts[f"{config['loss']}{config['bits']}"] += 1
        if len(set(keys)) != len(keys) or set(keys)-expected:
            raise ValueError('Duplicate or unexpected candidate result')
        return dict(completed_records=len(keys), expected=60, groups=dict(counts),
                    checkpoint_contents='Not loaded during active training; final consistency check still required')
    check('candidate_queue_and_completed_files', candidates)
    def registry(path):
        entries = read(root/path)
        if set(entries) != {'MSE256', 'NMR256', 'MSE512', 'NMR512'}:
            raise ValueError('Require exactly four final model entries')
        return dict(entries=sorted(entries), scope='Registry structure only; not weight integrity or final acceptance')
    for name, path in [('selected_registry_structure', 'runs/paper/selected.json'),
                       ('exported_registry_structure', 'weights/selected.json')]:
        check(name, lambda path=path: registry(path))
    return dict(timestamp=datetime.now().astimezone().isoformat(), checks=checks,
                scope='Read-only snapshot, not final acceptance. Cache payload and checkpoint contents are not fully scanned.',
                external_pending=['Huawei cloud resources and live acceptance', 'GitHub and weight publication',
                                  'Optional human listening', 'Student identity and actual contributions'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('outputs/readiness.json'))
    args = parser.parse_args()
    report = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in report['checks']:
        print(f"{item['status']}: {item['name']}")
    if any(item['status'] == 'review' for item in report['checks']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
