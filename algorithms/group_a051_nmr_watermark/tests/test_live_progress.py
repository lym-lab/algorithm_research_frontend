import json
from pathlib import Path

import torch
from nmr_watermark import training
from nmr_watermark.live_progress import EpochProgress
from scripts import progress as page


def test_percentage_phase_and_write_throttle(tmp_path):
    reporter = EpochProgress(tmp_path, 3, 30)
    reporter.update('training', 0, 10, force=True)
    reporter.update('training', 2, 10)
    path = tmp_path/'progress.json'
    assert json.loads(path.read_text())['completed'] == 0
    reporter.update('training', 2, 10, force=True)
    state = json.loads(path.read_text())
    assert (state['epoch'], state['percent']) == (4, 20)
    reporter.update('validation', 0, 2, force=True)
    assert json.loads(path.read_text())['percent'] == 0
    reporter.update('validation', 2, 2)
    assert json.loads(path.read_text())['percent'] == 100


def test_stopped_progress_is_not_displayed_as_running(tmp_path, monkeypatch):
    folder = tmp_path/'runs/paper/example'
    folder.mkdir(parents=True)
    (folder/'config.json').write_text('{}')
    EpochProgress(folder, 0, 30).update('training', 1, 4, force=True)
    monkeypatch.setattr(page, 'ROOT', tmp_path)
    monkeypatch.setattr(page.CONTROL, 'status', lambda: dict(running=False, processes=[]))
    state = page.snapshot()['epoch_progress']
    assert state['percent'] == 25 and not state['active']


def test_temporary_display_lock_does_not_interrupt_training(tmp_path, monkeypatch):
    import nmr_watermark.live_progress as live
    reporter = EpochProgress(tmp_path, 0, 30)
    def locked(*args):
        raise PermissionError('dashboard reader')
    with monkeypatch.context() as patch:
        patch.setattr(live.os, 'replace', locked)
        reporter.update('training', 1, 4, force=True)
    reporter.update('training', 2, 4, force=True)
    assert json.loads((tmp_path/'progress.json').read_text())['percent'] == 50


def test_reporting_preserves_training_and_resume(tmp_path, monkeypatch):
    class Dataset:
        def __init__(self, *args):
            self.rows = [{'track_id': str(i//2)} for i in range(4)]

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return dict(audio=torch.arange(8).float()*.01+index*.02,
                        mask=torch.ones(1, 8))

    class Model(torch.nn.Module):
        def __init__(self, bits, normalized):
            super().__init__()
            self.bits = bits
            self.encoder = torch.nn.Linear(8+bits, 8)
            self.decoder = torch.nn.Linear(8, bits)

        def forward(self, audio, message):
            marked = self.encoder(torch.cat([audio, message], 1))
            return marked, self.decoder(marked), audio[:, None], marked[:, None]

    class NMR(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, marked, host, mask):
            return (marked-host).square().mean((1, 2))

    cache = tmp_path/'cache'
    for split in ('train', 'validation'):
        (cache/split).mkdir(parents=True)
        (cache/split/'complete.json').write_text(json.dumps(
            dict(manifest='fixed', psycho='fixed', normalized=True)))
    for module in (training,):
        monkeypatch.setattr(module, 'AudioDataset', Dataset)
        monkeypatch.setattr(module, 'WatermarkModel', Model)
        monkeypatch.setattr(module, 'NMRLoss', NMR)
        monkeypatch.setattr(module, 'manifest_hash', lambda _: 'fixed')
        monkeypatch.setattr(module, 'psycho_hash', lambda: 'fixed')
    config = dict(bits=4, loss='mse', seed=2027, alpha=.8,
                  batch_size=2, lr=1e-4, epochs=3, stft_normalized=True)
    class SilentProgress:
        def __init__(self, *args):
            pass

        def update(self, *args, **kwargs):
            pass

    with monkeypatch.context() as patch:
        patch.setattr(training, 'EpochProgress', SilentProgress)
        patch.setattr(training, 'prefetch_batches', iter)
        training.train(config, 'train', 'val', cache, tmp_path/'before', device='cpu')

    class Interruption(Exception):
        pass

    save = training.atomic_save

    def interrupt(value, path):
        save(value, path)
        if Path(path).name == 'last.pt' and value['epoch'] == 1:
            raise Interruption()

    monkeypatch.setattr(training, 'atomic_save', interrupt)
    try:
        training.train(config, 'train', 'val', cache, tmp_path/'after', device='cpu')
    except Interruption:
        pass
    monkeypatch.setattr(training, 'atomic_save', save)
    training.train(config, 'train', 'val', cache, tmp_path/'after', device='cpu')
    left = torch.load(tmp_path/'before/last.pt', weights_only=False)
    right = torch.load(tmp_path/'after/last.pt', weights_only=False)
    assert left['metrics'] == right['metrics']
    for key in left['model']:
        assert torch.equal(left['model'][key], right['model'][key])
    for key, values in left['optimizer']['state'].items():
        for name, value in values.items():
            assert torch.equal(value, right['optimizer']['state'][key][name])
    assert torch.equal(left['rng_cpu'], right['rng_cpu'])
    assert json.loads((tmp_path/'after/progress.json').read_text())['phase'] == 'complete'
