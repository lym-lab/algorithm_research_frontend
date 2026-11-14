import json
import numpy as np
import pytest
import torch
from nmr_watermark.audio import save_audio, read_audio
from nmr_watermark.model import WatermarkModel, SEGMENT_SAMPLES
from nmr_watermark.inference import embed_array, extract_array, parse_message
from nmr_watermark.training import candidates, select


def test_wav_roundtrip_preserves_predictions(tmp_path):
    torch.manual_seed(72)
    model = WatermarkModel(256).eval()
    audio = np.random.default_rng(4).normal(0, .02, SEGMENT_SAMPLES+100).astype('float32')
    message = parse_message('01'*128, 256)
    marked, original = embed_array(model, audio, message)
    assert original == len(audio) and len(marked) == 2*SEGMENT_SAMPLES
    path = tmp_path/'marked.wav'
    save_audio(path, marked)
    restored = read_audio(path)
    np.testing.assert_array_equal(restored, marked)
    np.testing.assert_array_equal(extract_array(model, restored), extract_array(model, marked))


def test_selection_uses_snr_then_strict_ber_then_nmr(tmp_path):
    for i, cfg in enumerate(candidates()):
        folder = tmp_path/str(i)
        folder.mkdir()
        mse = cfg['loss'] == 'mse'
        scores = dict(snr=cfg['alpha'] if mse else 1000., ber=.1 if mse else .09,
                      nmr=cfg['alpha'])
        (folder/'result.json').write_text(json.dumps(dict(config=cfg, metrics=scores)))
    chosen = select(tmp_path)
    assert chosen['MSE256']['metrics']['snr'] == .98
    assert chosen['NMR256']['metrics']['nmr'] == .1
    for p in tmp_path.glob('*/result.json'):
        value = json.loads(p.read_text())
        if value['config']['loss'] == 'nmr':
            value['metrics']['ber'] = .1
            p.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='BER constraint'):
        select(tmp_path)


def test_pilot_split_has_no_track_leakage():
    from pathlib import Path
    root = Path('data/manifests/pilot')
    if not root.exists():
        pytest.skip('Local data not included in source distribution')
    train = json.loads((root/'train.json').read_text(encoding='utf-8'))
    val = json.loads((root/'validation.json').read_text(encoding='utf-8'))
    assert not ({r['track_id'] for r in train} & {r['track_id'] for r in val})
