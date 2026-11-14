"""Check that fresh messages can be decoded on a small real-audio batch.

This stronger diagnostic prevents fixed-message memorization from being
mistaken for a functioning message channel. It is not a paper experiment.
"""
import json
import argparse
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from nmr_watermark.data import AudioDataset
from nmr_watermark.model import WatermarkModel, SpectralTransform
from nmr_watermark.psychoacoustics import NMRLoss
from nmr_watermark.training import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--bits', type=int, choices=[256,512])
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = AudioDataset('data/manifests/pilot/train.json')
    audio = torch.stack([data[i]['audio'] for i in range(16)]).cuda()
    results = []
    for bits in ([args.bits] if args.bits else (256, 512)):
        for loss_kind in ('mse', 'nmr'):
            seed_all(2026)
            model, nmr = WatermarkModel(bits).cuda(), NMRLoss().cuda()
            if args.normalized:
                class UnitarySTFT(SpectralTransform):
                    def forward(self, x):
                        return super().forward(x)/32
                    def inverse(self, x):
                        return super().inverse(x*32)
                model.stft = UnitarySTFT().cuda()
                nmr.ear_gain.mul_(1024)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            with torch.no_grad():
                mask = nmr.masking(model.stft(audio))
            history = []
            started = time.monotonic()
            for step in range(args.steps):
                model.train()
                message = torch.randint(0, 2, (16, bits), device='cuda').float()
                optimizer.zero_grad(set_to_none=True)
                _, logits, host, marked = model(audio, message)
                distortion = nmr(marked, host, mask).mean() if loss_kind == 'nmr' else (marked-host).abs().square().mean()
                alpha = 1e-4 if loss_kind == 'nmr' else .5
                loss = alpha*distortion + (1-alpha)*F.binary_cross_entropy_with_logits(logits, message)
                loss.backward()
                optimizer.step()
                if step % 200 == 0 or step == args.steps-1:
                    model.eval()
                    with torch.no_grad():
                        # Fresh unseen messages; audio held fixed for this diagnostic.
                        fresh = torch.randint(0, 2, (16, bits), device='cuda').float()
                        _, pred, _, _ = model(audio, fresh)
                        ber = ((pred >= 0) != fresh.bool()).float().mean().item()
                    row = dict(step=step, fresh_message_ber=ber, seconds=time.monotonic()-started)
                    history.append(row)
                    print(bits, loss_kind, row, flush=True)
                    if step >= 400 and ber < .1:
                        break
            results.append(dict(bits=bits, loss=loss_kind, passed=ber < .1, history=history))
            suffix = '_unitary' if args.normalized else ''
            Path(args.output or f'outputs/message_learning{suffix}.json').write_text(json.dumps(results, indent=2))
            del model, optimizer, nmr
            torch.cuda.empty_cache()
    if not all(r['passed'] for r in results):
        raise RuntimeError('Fresh-message learning check failed; investigate before full search')


if __name__ == '__main__':
    main()
