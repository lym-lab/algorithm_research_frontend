"""Real-audio random-message pilot and exact epoch-resume validation."""
import json
import argparse
from pathlib import Path
import torch
import nmr_watermark.training as training


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--normalized', action='store_true')
    options = p.parse_args()
    suffix = '_unitary' if options.normalized else ''
    torch.set_num_threads(4)
    manifests = Path('data/manifests/pilot')
    cache = Path('data/cache/pilot'+suffix)
    for split in ('train', 'validation'):
        training.cache_masks(manifests/f'{split}.json', cache/split, 'cuda', normalized=options.normalized)
    args = (manifests/'train.json', manifests/'validation.json', cache)
    config = dict(bits=256, loss='mse', alpha=.8, batch_size=16, seed=2026, lr=1e-4, epochs=4, pilot=True, stft_normalized=options.normalized)
    continuous = Path('runs/validation'+suffix+'/continuous')
    resumed = Path('runs/validation'+suffix+'/resumed')
    training.train(config, *args, continuous)
    original_save = training.atomic_save

    class SimulatedInterruption(Exception):
        pass

    def interrupt_at_epoch(value, path):
        original_save(value, path)
        if Path(path).name == 'last.pt' and value['epoch'] == 1:
            raise SimulatedInterruption()

    if not (resumed/'last.pt').exists():
        training.atomic_save = interrupt_at_epoch
        try:
            training.train(config, *args, resumed)
        except SimulatedInterruption:
            print('Simulated interruption at epoch boundary', flush=True)
        finally:
            training.atomic_save = original_save
    training.train(config, *args, resumed)
    left = torch.load(continuous/'last.pt', weights_only=False, map_location='cpu')
    right = torch.load(resumed/'last.pt', weights_only=False, map_location='cpu')
    for key in left['model']:
        torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0)
    for key, state in left['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, right['optimizer']['state'][key][name], rtol=0, atol=0)
    assert left['metrics'] == right['metrics']
    summaries = []
    registry = {}
    for bits in (256, 512):
        for loss, alpha in [('mse', .8), ('nmr', .1)]:
            out = Path(f'runs/pilot_random{suffix}/{loss}{bits}')
            cfg = dict(config, bits=bits, loss=loss, alpha=alpha, epochs=12)
            summaries.append(training.train(cfg, *args, out))
            registry[f'{loss.upper()}{bits}'] = {'checkpoint': str((out/'best.pt').resolve())}
    Path(f'runs/pilot_random{suffix}/selected.json').write_text(json.dumps(registry, indent=2))
    Path(f'outputs/training_validation{suffix}.json').write_text(json.dumps({
        'resume_exact': True, 'pilot': True, 'random_message_runs': summaries,
        'warning': 'Pilot weights are diagnostic only; not final paper models.'}, indent=2))


if __name__ == '__main__':
    main()
