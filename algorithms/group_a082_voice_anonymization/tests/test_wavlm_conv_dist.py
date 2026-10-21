"""Data-parallel helpers of the WavLM conversion training, with two CPU processes (gloo) and a small model."""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from a082.wavlm_conv import losses
from a082.wavlm_conv.model import WavLMConverter
from a082.wavlm_conv.train import all_reduce_grads, data_parallel, rank_share


def small_model():
    return WavLMConverter(feat_dim=8, d=8, enc_layers=1, dec_layers=1, heads=2, filt=8, dropout=0.0, postnet=False)


def test_rank_share():
    assert [rank_share(3, r, 4) for r in range(4)] == [1, 1, 1, 0]
    assert [rank_share(16, r, 4) for r in range(4)] == [4, 4, 4, 4]
    assert sum(rank_share(7, r, 3) for r in range(3)) == 7


def _worker(rank, world, init, out):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    torch.manual_seed(100 + rank)  # different initial weights: data_parallel must replace them by rank 0's
    model = data_parallel(small_model(), world, "cpu")
    torch.manual_seed(0)
    x, y = torch.randn(4, 6, 8), torch.randn(4, 6, 8)  # the same global batch on every rank
    part = slice(2 * rank, 2 * rank + 2)
    valid = torch.ones(2, 6, dtype=torch.bool)
    loss = losses.reconstruction(*model(x[part]), y[part], valid)
    loss.backward()
    all_reduce_grads(model.parameters(), world)
    out[rank] = {"w": [p.detach().clone() for p in model.parameters()],
                 "g": [p.grad.clone() for p in model.parameters()]}
    dist.destroy_process_group()


def test_two_ranks_match_one_batch(tmp_path):
    out = mp.Manager().dict()
    mp.spawn(_worker, args=(2, (tmp_path / "store").as_uri(), out), nprocs=2)
    # same weights on both ranks
    for a, b in zip(out[0]["w"], out[1]["w"]):
        assert torch.equal(a, b)
    # averaged gradients equal the gradient of the whole batch on one process (equal frame counts per rank)
    model = small_model()
    with torch.no_grad():
        for p, w in zip(model.parameters(), out[0]["w"]):
            p.copy_(w)
    torch.manual_seed(0)
    x, y = torch.randn(4, 6, 8), torch.randn(4, 6, 8)
    losses.reconstruction(*model(x), y, torch.ones(4, 6, dtype=torch.bool)).backward()
    for p, g0, g1 in zip(model.parameters(), out[0]["g"], out[1]["g"]):
        assert torch.equal(g0, g1)
        assert torch.allclose(p.grad, g0, atol=1e-6)
