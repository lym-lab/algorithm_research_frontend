"""Scratch benchmark; never loads or writes formal model checkpoints."""
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from nmr_watermark.data import AudioDataset
from nmr_watermark.training import TrackSampler, seed_all
from nmr_watermark.model import WatermarkModel
from nmr_watermark.psychoacoustics import NMRLoss
from nmr_watermark.prefetch import prefetch_batches


def run(prefetch):
    seed_all(2027)
    torch.set_num_threads(4)
    dataset = AudioDataset('data/manifests/full/train.json', 'data/cache/full/train')
    loader = DataLoader(dataset, batch_size=16, sampler=TrackSampler(dataset, 2040))
    model = WatermarkModel(256, True).cuda()
    nmr = NMRLoss(True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    iterator = prefetch_batches(loader) if prefetch else iter(loader)
    waits, computes, losses = [], [], []
    for step in range(96):
        started = time.perf_counter()
        batch = next(iterator)
        loaded = time.perf_counter()
        audio, mask = batch['audio'].cuda(), batch['mask'].cuda()
        message = torch.randint(0, 2, (len(audio), 256), device='cuda').float()
        optimizer.zero_grad(set_to_none=True)
        marked_audio, logits, host, marked = model(audio, message)
        loss = .2*nmr(marked, host, mask).mean()+.8*torch.nn.functional.binary_cross_entropy_with_logits(logits, message)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        ended = time.perf_counter()
        if step >= 24:
            waits.append(loaded-started)
            computes.append(ended-loaded)
    if prefetch:
        iterator.close()
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return dict(prefetch=prefetch, batches=len(waits), wait_seconds=sum(waits), compute_seconds=sum(computes), seconds=sum(waits)+sum(computes)), state, losses


if __name__ == '__main__':
    first, left, ll = run(False)
    second, right, rl = run(True)
    third, repeat, tl = run(False)
    result = dict(baseline=first, prefetch=second,
                  baseline_repeat=third,
                  model_bitwise_equal=all(torch.equal(left[k],right[k]) for k in left),
                  losses_equal=ll == rl == tl,
                  repeat_equal=all(torch.equal(left[k],repeat[k]) for k in left))
    Path('outputs/prefetch-benchmark.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
