"""Freeze the verified numerical convention before starting paper training."""
import hashlib
import json
from pathlib import Path


def main():
    def read(path):
        return json.loads(Path(path).read_text())
    assert read('outputs/reference_validation.json')['passed']
    assert read('outputs/training_validation_unitary.json')['resume_exact']
    fixed = read('runs/pilot_unitary/diagnostics.json')
    assert len(fixed) == 4 and all(r['passed'] for r in fixed)
    learning = [r for r in read('outputs/message_learning_unitary.json') if r['bits']==256]
    learning += read('outputs/message_learning_unitary_extended.json')
    assert {(r['bits'],r['loss']) for r in learning} == {(b,k) for b in (256,512) for k in ('mse','nmr')}
    assert all(r['passed'] and r['history'][-1]['fresh_message_ber'] < .1 for r in learning)
    files = ['model.py','psychoacoustics.py','training.py','data.py','live_progress.py','prefetch.py']
    hashes = {name:hashlib.sha256((Path('nmr_watermark')/name).read_bytes()).hexdigest() for name in files}
    result = dict(approved_diagnostics=True, stft_normalized=True, batch_size=16, lr=1e-4,
                  seeds=[2026,2027,2028], implementation_sha256=hashes,
                  rationale='Raw STFT MSE failed fresh-message learning; unitary STFT passed with calibrated NMR invariant.',
                  caveat='Fresh messages on fixed diagnostic audio only; not unseen-audio accuracy or paper results.',
                  fresh_message_checks=[dict(bits=r['bits'],loss=r['loss'],**r['history'][-1]) for r in learning])
    target=Path('outputs/frozen_config.json')
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
