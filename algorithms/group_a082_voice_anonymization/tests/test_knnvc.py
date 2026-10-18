"""Second-scale checks of the kNN-VC code on synthetic tensors (no checkpoints needed)."""
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from a082 import kaldi, seeding
from a082.knnvc.core import knn_pooled, knn_regress, stretch, stretched_frames, uniform_noise
from a082.knnvc.anonymize import KnnSystem


def brute_knn(q, s, k):
    d = 1 - F.cosine_similarity(q[:, None, :], s[None, :, :], dim=-1)  # [T, N]
    idx = d.argsort(dim=-1)[:, :k]
    return s[idx].mean(dim=1)


def test_knn_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    q, s = torch.randn(37, 16, generator=g), torch.randn(500, 16, generator=g)
    assert torch.allclose(knn_regress(q, s, k=4, chunk=8), brute_knn(q, s, 4), atol=1e-5)


def test_pooled_single_set_equals_plain():
    g = torch.Generator().manual_seed(1)
    q, s = torch.randn(10, 8, generator=g), torch.randn(50, 8, generator=g)
    assert torch.allclose(knn_pooled(q, [s]), knn_regress(q, s))


def test_pooled_weights():
    g = torch.Generator().manual_seed(2)
    q, a, b = torch.randn(10, 8, generator=g), torch.randn(50, 8, generator=g), torch.randn(60, 8, generator=g)
    w = torch.tensor([0.3, 0.7])
    assert torch.allclose(knn_pooled(q, [a, b], w), 0.3 * knn_regress(q, a) + 0.7 * knn_regress(q, b), atol=1e-6)


def test_stretch_lengths():
    assert stretched_frames(16000 * 10, 0.8) == 400          # 8 s at 50 frames/s
    assert stretched_frames(16000 * 10, 1.2) == 600
    assert stretched_frames(16000 * 0.3, 1.0) == 25          # floor at 0.5 s
    x = torch.randn(500, 4)
    y = stretch(x, 400)
    assert y.shape == (400, 4)
    assert torch.allclose(y[0], x[0], atol=0.5) and stretch(x, 500) is x


def test_noise_bounds_and_determinism():
    e1 = uniform_noise((1000, 8), 32.0, seeding.torch_gen("u1", "noise"), "cpu")
    e2 = uniform_noise((1000, 8), 32.0, seeding.torch_gen("u1", "noise"), "cpu")
    assert torch.equal(e1, e2) and e1.abs().max() < 32.0 and e1.abs().max() > 30.0


def test_seeding_is_stable():
    assert seeding.seed_int("1272-128104-0000", "target") == seeding.seed_int("1272-128104-0000", "target")
    assert seeding.seed_int("a", "target") != seeding.seed_int("a", "length")
    u = [seeding.uniform01(f"utt{i}", "admix") for i in range(2000)]
    assert 0 <= min(u) and max(u) < 1 and abs(np.mean(u) - 0.5) < 0.03


class _FakePool:
    speakers = [f"id{i:05d}" for i in range(40)]

    def __len__(self):
        return len(self.speakers)


def test_plan_shares_target_across_systems():
    base = dict(k=4, n_targets=1, length_range=None, noise=None)
    s1 = KnnSystem(base, _FakePool(), None, None, "cpu")
    s2 = KnnSystem({**base, "length_range": [0.8, 1.2]}, _FakePool(), None, None, "cpu")
    p1, p2 = s1.plan("1462-170142-0000"), s2.plan("1462-170142-0000")
    assert p1["targets"] == p2["targets"] and p1["ratio"] is None and 0.8 <= p2["ratio"] <= 1.2
    sm = KnnSystem({**base, "n_targets": 2, "dirichlet": True}, _FakePool(), None, None, "cpu")
    pm = sm.plan("x")
    assert len(set(pm["targets"])) == 2 and abs(pm["weights"].sum() - 1) < 1e-9


def test_kaldi_round_trip(tmp_path):
    src, out = tmp_path / "data" / "set", tmp_path / "data" / "set_x"
    src.mkdir(parents=True)
    kaldi.write_kaldi({"s1-1-0001": "a.wav", "s1-1-0002": "b.wav"}, src / "wav.scp")
    kaldi.write_kaldi({"s1-1-0001": "s1", "s1-1-0002": "s1"}, src / "utt2spk")
    kaldi.prepare_output_dir(src, out)
    assert (out / "utt2spk").exists() and not (out / "wav.scp").exists()
    sf.write(kaldi.wav_path(out, "s1-1-0001"), np.zeros(1600, np.float32), 16000, subtype="PCM_16")
    try:
        kaldi.finalize_output_dir(src, out)
        raise AssertionError("missing output not detected")
    except RuntimeError:
        pass
    sf.write(kaldi.wav_path(out, "s1-1-0002"), np.zeros(3200, np.float32), 16000, subtype="PCM_16")
    kaldi.finalize_output_dir(src, out)
    assert kaldi.read_kaldi(out / "utt2dur") == {"s1-1-0001": "0.100", "s1-1-0002": "0.200"}
