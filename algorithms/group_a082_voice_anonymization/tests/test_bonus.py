"""Bonus-experiment code on synthetic data."""
import numpy as np
import torch

from a082 import admix, kaldi
from a082.bonus import attack, sysclf
from tests.test_wavlm_conv_pipeline import make_vpc, tiny_wavlm


def test_eer_known_cases():
    assert attack.eer(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    rng = np.random.default_rng(0)
    e = attack.eer(rng.normal(1, 1, 20000), rng.normal(-1, 1, 20000))
    assert abs(e - 15.87) < 0.6  # Phi(-1) for unit-variance Gaussians 2 apart


def test_subset_dir(tmp_path):
    src = tmp_path / "s"; src.mkdir()
    kaldi.write_kaldi({"a-1-1": "x", "a-1-2": "y", "b-1-1": "z"}, src / "wav.scp")
    kaldi.write_kaldi({"a-1-1": "a", "a-1-2": "a", "b-1-1": "b"}, src / "utt2spk")
    kaldi.write_kaldi({"a": "f", "b": "m"}, src / "spk2gender")
    n = kaldi.subset_dir(src, tmp_path / "o", {"a-1-2", "b-1-1"})
    assert n == 2
    assert kaldi.read_kaldi(tmp_path / "o" / "spk2utt") == {"a": "a-1-2", "b": "b-1-1"}
    assert kaldi.read_kaldi(tmp_path / "o" / "spk2gender") == {"a": "f", "b": "m"}


def test_speaker_level_admixture_is_constant_per_speaker():
    spk = [f"s{i}" for i in range(200)]
    assert abs(np.mean([admix.uses_tts(s, 0.4) for s in spk]) - 0.4) < 0.1
    assert admix.suffix_for(0.325) == "_mix0p325"


def test_logreg_separates():
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(200, 8, generator=g) - 1, torch.randn(200, 8, generator=g) + 1])
    y = torch.cat([torch.zeros(200), torch.ones(200)])
    m = sysclf.fit_logreg(x, y)
    assert ((sysclf.predict(m, x) > 0.5).float() == y).float().mean() > 0.95


def test_split_writes_labels_and_subsets(tmp_path, monkeypatch):
    monkeypatch.setattr(sysclf, "load_wavlm", tiny_wavlm)
    root = tmp_path / "vpc"
    make_vpc(root)
    data = root / "data"
    # pretend the mixture is the original data (we only check the plumbing)
    for name in ("train-clean-360",):
        d = data / name
        kaldi.link_output_dir(d, data / f"{name}_mix0p4", {u: p for u, p in kaldi.read_kaldi(d / "wav.scp").items()})
    model = {"mean": torch.zeros(128), "std": torch.ones(128), "w": torch.randn(128), "b": torch.zeros(1), "crop": 1.0}
    torch.save(model, tmp_path / "m.pt")

    class A:
        vpc_root = root; device = "cpu"; model = str(tmp_path / "m.pt"); mix_suffix = "_mix0p4"; p = 0.4
        sets = "train-clean-360"; labels_dir = str(tmp_path / "labels"); level = "utt"; name = "pred"; no_true = False
    sysclf.cmd_split(A)
    labels = sysclf.read_labels(tmp_path / "labels" / "mix0p4" / "train-clean-360.txt")
    assert len(labels) == 8
    n = sum(len(kaldi.read_kaldi(data / f"train-clean-360_mix0p4_true_{c}" / "wav.scp")) for c in ("knn", "tts"))
    assert n == 8


def test_analyse_with_synthetic_scores(tmp_path):
    root = tmp_path / "vpc"
    rng = np.random.default_rng(1)
    for enr, tri in attack.TRIAL_SETS:
        d = root / "data" / tri; d.mkdir(parents=True, exist_ok=True)
        lines, lab = [], []
        for i in range(60):
            utt = f"{tri}-u{i}"
            lines.append(f"spk{i % 6} {utt} {'target' if i % 3 == 0 else 'nontarget'}")
            lab.append(f"{utt} {'knn' if i % 2 else 'tts'} {'knn' if i % 2 else 'tts'} 0.5")
        (d / "trials").write_text("\n".join(lines) + "\n")
        (tmp_path / "labels").mkdir(exist_ok=True)
        (tmp_path / "labels" / f"{tri}.txt").write_text("\n".join(lab) + "\n")
        for model in ("single", "knn", "tts", "true_knn", "true_tts"):
            sd = root / "exp" / model / f"{enr}_mix0p4-{tri}_mix0p4"; sd.mkdir(parents=True)
            rows = []
            for ln in lines:
                e, t, l = ln.split()
                rows.append(f"{e} {t} {rng.normal(1.0 if l == 'target' else 0.0, 0.3):.4f}")
            (sd / "scores").write_text("\n".join(rows) + "\n")
    rep = attack.analyse(root, "_mix0p4", tmp_path / "labels",
                         {m: root / "exp" / m for m in ("single", "knn", "tts")})
    assert set(rep["average"]) == {"single", "knn", "tts", "routed"}
    assert rep["average"]["single"]["all"] < 20
    # without a predicted-tts model (improvement A): no routed; the oracle pair still routes
    rep = attack.analyse(root, "_mix0p4", tmp_path / "labels",
                         {m: root / "exp" / m for m in ("single", "knn", "true_knn", "true_tts")})
    assert set(rep["average"]) == {"single", "knn", "true_knn", "true_tts", "routed_true"}


def test_missing_iemocap_is_skipped_without_creating_folders(tmp_path):
    from a082 import runner
    root = tmp_path / "vpc"
    make_vpc(root)
    d = root / "data" / "IEMOCAP_dev"
    d.mkdir(parents=True)
    kaldi.write_kaldi({"Ses01F_impro01_F000": "data/IEMOCAP/wav/Session1/x.wav"}, d / "wav.scp")
    kaldi.write_kaldi({"Ses01F_impro01_F000": "Ses01F"}, d / "utt2spk")
    calls = []
    runner.anonymize_sets(root, ["libri_dev_enrolls", "IEMOCAP_dev"], "_t", lambda u, w: (calls.append(u), w)[1])
    assert len(calls) == 2
    assert not (root / "data" / "IEMOCAP_dev_t").exists()
    assert kaldi.audio_available(root / "data" / "libri_dev_enrolls", root)
    assert runner.resolve_sets("libri") == kaldi.LIBRI_SETS and kaldi.ALL_SETS[-4:] == ["IEMOCAP_dev", "IEMOCAP_test", "CREMA_dev", "CREMA_test"]
