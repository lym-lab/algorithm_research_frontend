"""Read-only CPU audit of the uploaded paper contract; never train or select models."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from mutagen.mp3 import MP3

from nmr_watermark.model import WatermarkModel, SEGMENT_SAMPLES
from nmr_watermark.psychoacoustics import NMRLoss
from nmr_watermark.training import candidates


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2102026)
    out = Path('outputs/paper_audit_20261021')
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path('nmr_watermark').glob('*.py'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): sha(p) for p in paths}
    evidence = {'checked_at': datetime.now().isoformat(timespec='seconds'),
                'scope': 'CPU synthetic contract checks and read-only metadata; not paper performance',
                'source_sha256': before, 'network': {}}
    frozen = json.loads(Path('outputs/frozen_config.json').read_text())
    evidence['frozen_core_matches'] = {
        k: before[str(Path('nmr_watermark')/k)] == v
        for k, v in frozen['implementation_sha256'].items()}
    assert all(evidence['frozen_core_matches'].values())
    for bits in (256, 512):
        model = WatermarkModel(bits, normalized=True).eval()
        shapes, hooks = {}, []
        for name, layer in model.named_modules():
            if name in [*(f'embedder.encoder.{i}' for i in range(4)),
                        'embedder.message', *(f'embedder.decoder.{i}' for i in range(4)),
                        'embedder.output', *(f'extractor.layers.{i}' for i in range(3)),
                        'extractor.output']:
                def record(module, inputs, output, key=name):
                    shapes[key] = list(output.shape)
                hooks.append(layer.register_forward_hook(record))
        with torch.no_grad():
            x = torch.randn(1, SEGMENT_SAMPLES) * .02
            y, logits, host, marked = model(x, torch.zeros(1, bits))
            error = (model.stft.inverse(host)-x).abs().max().item()
            assert host.shape == (1, 513, 96) and torch.equal(marked, model.stft(y))
            assert shapes['extractor.layers.2'] == [1, 128, 20, 12]
            assert error < 2e-7
        for hook in hooks:
            hook.remove()
        evidence['network'][str(bits)] = {'shapes': shapes, 'roundtrip_max_abs': error,
                                         'reanalyzed_stft_used': True}
        del model
    nmr = NMRLoss().double()
    u = nmr.mapping.numpy()
    sums = u.sum(axis=0)
    partial = np.flatnonzero((sums > 1e-10) & (sums < 1-1e-10))
    evidence['mapping'] = {'shape': list(u.shape),
        'unit_sum_columns': int(np.isclose(sums, 1, atol=1e-10, rtol=0).sum()),
        'zero_sum_columns': int((sums < 1e-10).sum()),
        'partial_columns_zero_based': partial.tolist(),
        'partial_sums': sums[partial].tolist(),
        'crop_bin_centers_hz': [3*44100/1024, 130*44100/1024],
        'literal_all_columns_sum_one': bool(np.allclose(sums, 1))}
    rng = np.random.default_rng(20261021)
    host = rng.normal(size=(513, 3)) + 1j*rng.normal(size=(513, 3))
    marked = host + .01*(rng.normal(size=host.shape)+1j*rng.normal(size=host.shape))
    mask = rng.uniform(1, 5, size=(109, 3))
    # Independent real/imag expansion of the complex squared magnitude in (2).
    diff = host-marked
    expected_noise = u @ ((diff.real**2+diff.imag**2)*nmr.ear_gain.numpy()[:, None])
    expected = float((expected_noise/mask).sum()/(109*3))
    tensor = lambda a: torch.from_numpy(a).unsqueeze(0)
    actual_noise = nmr.noise(tensor(marked), tensor(host))[0].numpy()
    actual = nmr(tensor(marked), tensor(host), tensor(mask)).item()
    np.testing.assert_allclose(actual_noise, expected_noise, rtol=1e-12, atol=1e-10)
    assert np.isclose(actual, expected, rtol=1e-12)
    evidence['equations_2_3'] = {'passed_given_current_U_and_gain': True,
        'max_noise_abs_difference': float(np.max(np.abs(actual_noise-expected_noise))),
        'nmr_relative_difference': abs(actual-expected)/expected,
        'limit': 'Does not establish author calibration or U boundary equivalence'}
    grid = list(candidates())
    evidence['grid'] = {'count': len(grid),
        'unique': len({(c['bits'], c['loss'], c['alpha'], c['seed']) for c in grid}),
        'mse_alphas': sorted({c['alpha'] for c in grid if c['loss']=='mse'}),
        'nmr_alphas': sorted({c['alpha'] for c in grid if c['loss']=='nmr'})}
    manifests = {k: json.loads(Path(f'data/manifests/full/{k}.json').read_text(encoding='utf-8'))
                 for k in ('train', 'validation')}
    tracks = {k: {r['track_id'] for r in rows} for k, rows in manifests.items()}
    evidence['data'] = {'segments': {k: len(v) for k,v in manifests.items()},
        'track_counts': {k: len(v) for k,v in tracks.items()},
        'track_overlap': len(tracks['train'] & tracks['validation']),
        'all_tracks_have_24_segments': all(set(Counter(r['track_id'] for r in rows).values())=={24}
                                         for rows in manifests.values())}
    unique = {r['path'] for rows in manifests.values() for r in rows}
    near = []
    for path in sorted(unique):
        info = MP3(path).info
        if info.bitrate < 192000:
            near.append({'track': Path(path).stem, 'bitrate': info.bitrate,
                         'bitrate_mode': str(info.bitrate_mode)})
    evidence['data']['selected_below_literal_192000'] = near
    evidence['data']['minimum_selected_bitrate'] = min((v['bitrate'] for v in near), default=192000)
    rwc = json.loads(Path('data/manifests/rwc.json').read_text(encoding='utf-8'))
    evidence['data']['rwc'] = {'segments': len(rwc), 'tracks': len({r['track_id'] for r in rwc}),
        'per_track_counts': sorted(set(Counter(r['track_id'] for r in rwc).values()))}
    assert evidence['data']['track_overlap'] == 0
    after = {str(p): sha(p) for p in paths}
    assert before == after
    evidence['source_unchanged_during_check'] = True
    (out/'contract.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'mapping': evidence['mapping'], 'data_counts': evidence['data']['segments'],
                      'near_bitrate_count': len(near), 'equations': evidence['equations_2_3'],
                      'source_unchanged': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
