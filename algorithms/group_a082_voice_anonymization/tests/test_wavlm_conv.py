"""Second-scale checks of the WavLM conversion model with small dimensions and random data."""
import math

import torch

from a082.wavlm_conv import kmeans, losses
from a082.wavlm_conv.model import NoamLR, WavLMConverter, lengths_to_pad
from a082.wavlm_conv.train import frames_for


def small_model():
    return WavLMConverter(feat_dim=32, d=16, enc_layers=1, dec_layers=1, heads=2, filt=24, dropout=0.0)


def test_forward_shapes_and_padding():
    m = small_model().eval()
    x = torch.randn(3, 20, 32)
    lengths = torch.tensor([20, 12, 5])
    pad = lengths_to_pad(lengths, 20)
    y, y_post = m(x, pad)
    assert y.shape == y_post.shape == (3, 20, 32)
    assert y_post[2, 5:].abs().sum() == 0
    # a padded item gives the same valid output as the same item alone
    y_alone, _ = m(x[1:2, :12])
    assert torch.allclose(y[1, :12], y_alone[0], atol=1e-4)


def test_reconstruction_ignores_padding():
    y = torch.zeros(1, 4, 2); tgt = torch.zeros(1, 4, 2); tgt[0, 3] = 100.0
    valid = torch.tensor([[True, True, True, False]])
    assert losses.reconstruction(y, y, tgt, valid).item() == 0.0


def test_model_overfits_one_batch():
    torch.manual_seed(0)
    m = small_model()
    x, tgt = torch.randn(2, 10, 32), torch.randn(2, 10, 32)
    valid = torch.ones(2, 10, dtype=torch.bool)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        y, yp = m(x)
        loss = losses.reconstruction(y, yp, tgt, valid)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3 * first


def test_capped_adversarial():
    logits = torch.zeros(4, 10); logits[:, 0] = -50.0
    v = losses.capped_adversarial(logits, torch.zeros(4, dtype=torch.long), 10)
    assert abs(v.item() - math.log(10)) < 1e-6


def test_ctc_learns_units():
    torch.manual_seed(0)
    cent = kmeans.spherical_kmeans(torch.randn(500, 16), k=8, iters=10)
    assert cent.shape == (8, 16) and torch.allclose(cent.norm(dim=-1), torch.ones(8), atol=1e-5)
    ctc = losses.CosineCTC(cent, sigma=10.0)
    units = [torch.tensor([1, 4, 2]), torch.tensor([7, 3])]
    y = torch.randn(2, 12, 16, requires_grad=True)
    lengths = torch.tensor([12, 9])
    opt = torch.optim.Adam([y] + list(ctc.parameters()), lr=0.1)
    for _ in range(200):
        loss = ctc(y, lengths, units)
        opt.zero_grad(); loss.backward(); opt.step()
    assert losses.greedy_unit_error_rate(ctc.logits(y), lengths, units) == 0.0


def test_dedup():
    assert losses.dedup(torch.tensor([3, 3, 7, 7, 7, 3])).tolist() == [3, 7, 3]


def test_frames_for_matches_wavlm_frontend():
    assert frames_for(torch.tensor([16000, 160000, 400])).tolist() == [49, 499, 1]


def test_noam_peak():
    opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1.0)
    s = NoamLR(opt, d=512, warmup=4000)
    for _ in range(4000):
        opt.step(); s.step()
    assert abs(opt.param_groups[0]["lr"] - 512 ** -0.5 * 4000 ** -0.5) < 1e-6
