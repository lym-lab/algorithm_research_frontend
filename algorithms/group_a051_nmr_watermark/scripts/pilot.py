"""Fixed-message overfit diagnostic, not paper results or final weights."""
import json
import argparse
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from nmr_watermark.data import AudioDataset
from nmr_watermark.model import WatermarkModel
from nmr_watermark.psychoacoustics import NMRLoss
from nmr_watermark.training import seed_all, atomic_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path('runs/pilot_unitary' if args.normalized else 'runs/pilot')
    output.mkdir(parents=True, exist_ok=True)
    dataset = AudioDataset('data/manifests/pilot/train.json')
    audio = torch.stack([dataset[i]['audio'] for i in range(16)]).cuda()
    results = []
    for bits in (256, 512):
        for kind in ('mse', 'nmr'):
            seed_all(2026)
            model = WatermarkModel(bits, args.normalized).cuda()
            nmr = NMRLoss(args.normalized).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            message = torch.randint(0, 2, (16, bits), device='cuda').float()
            with torch.no_grad():
                mask = nmr.masking(model.stft(audio))
            torch.cuda.reset_peak_memory_stats()
            history = []
            start = time.monotonic()
            for step in range(400):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                marked_audio, logits, host, marked = model(audio, message)
                distortion = nmr(marked, host, mask).mean() if kind == 'nmr' else (marked-host).abs().square().mean()
                alpha = 0.0001 if kind == 'nmr' else 0.5
                loss = alpha*distortion + (1-alpha)*F.binary_cross_entropy_with_logits(logits, message)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Pilot loss is nonfinite')
                loss.backward()
                optimizer.step()
                if step % 10 == 0 or step == 399:
                    model.eval()
                    with torch.no_grad():
                        wave, logits, _, _ = model(audio, message)
                        ber = ((logits >= 0) != message.bool()).float().mean().item()
                    row = dict(step=step, loss=loss.item(), ber=ber, seconds=time.monotonic()-start)
                    history.append(row)
                    print(bits, kind, row, flush=True)
                    if step >= 30 and ber <= .01:
                        break
            config = dict(bits=bits, loss=kind, pilot=True, fixed_message=True, seed=2026, batch_size=16, stft_normalized=args.normalized)
            atomic_save(dict(model=model.cpu().state_dict(), config=config), output/f'{kind}{bits}.pt')
            result = dict(**config, passed=ber <= .01, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                          seconds_per_step=(time.monotonic()-start)/(step+1), history=history)
            results.append(result)
            (output/'diagnostics.json').write_text(json.dumps(results, indent=2))
            del model, optimizer, nmr
            torch.cuda.empty_cache()
    if not all(r['passed'] for r in results):
        raise RuntimeError('Fixed-message overfit failed; full training remains gated')


if __name__ == '__main__':
    main()
