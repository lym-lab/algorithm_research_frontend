"""kNN-VC driver, admixture and result collection on a tiny fake VPC directory (no checkpoints)."""
import csv
from argparse import Namespace

import torch
import yaml

from a082 import admix, kaldi
from a082.eval import vpc
from a082.knnvc import anonymize, pool as pool_mod
from tests.test_wavlm_conv_pipeline import DIM, StubVocoder, make_vpc, tiny_wavlm


def test_knn_systems_and_admixture(tmp_path, monkeypatch):
    monkeypatch.setattr(anonymize, "load_wavlm", tiny_wavlm)
    monkeypatch.setattr(anonymize, "load_hifigan", lambda device="cpu", prematched=True, path=None: StubVocoder())
    monkeypatch.setattr(pool_mod, "load_wavlm", tiny_wavlm)
    root = tmp_path / "vpc"
    tgt_files = make_vpc(root)

    # pool of 3 speakers built by the real build step (tiny WavLM), threshold lowered for 1.5 s files
    manifest = {f"id{i}": tgt_files for i in range(3)}
    cache = tmp_path / "pool"
    pool_mod.build(manifest, cache, device="cpu")
    monkeypatch.setattr(pool_mod.Pool.__init__, "__defaults__", ("cuda", 10, 8, 256))

    for name, extra in (("knn1", {}), ("knn3", {"length_range": [0.8, 1.2], "noise": {"where": "matching", "delta": 32.0}})):
        cfg = {"suffix": f"_{name}", "k": 4, "n_targets": 1, "length_range": None, "noise": None, **extra}
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        anonymize.main(["--config", str(p), "--pool", str(cache), "--vpc-root", str(root),
                        "--sets", "libri_dev_enrolls", "--device", "cpu"])
        assert len(kaldi.read_kaldi(root / "data" / f"libri_dev_enrolls_{name}" / "wav.scp")) == 2

    # pretend knn3 is the TTS system: admixture links per utterance, p=1 takes everything from "tts"
    frac = admix.build(root, "_knn1", "_knn3", 1.0, ["libri_dev_enrolls"])
    assert frac["libri_dev_enrolls"] == 1.0
    linked = kaldi.read_kaldi(root / "data" / "libri_dev_enrolls_mix1" / "wav.scp")
    assert all((root / "data" / "libri_dev_enrolls_mix1" / "wav").exists() for _ in linked)


def test_collect_reads_framework_csvs(tmp_path):
    exp = tmp_path / "exp"
    (exp / "asv_anon_knn1").mkdir(parents=True); (exp / "asr").mkdir(); (exp / "ser").mkdir()
    with open(exp / "asv_anon_knn1" / "results_knn1.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["", "dataset", "split", "gender", "enrollment", "trial", "EER"])
        for i, (s, g, e) in enumerate([("dev", "f", 11.789), ("dev", "m", 5.141), ("test", "f", 9.307), ("test", "m", 5.570)]):
            w.writerow([i, "libri", s, g, "anon", "anon", e])
            w.writerow([i, "libri", s, g, "original", "anon", 40.0])
    with open(exp / "asr" / "results_knn1.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["", "dataset", "split", "asr", "WER"])
        w.writerow([0, "libri", "dev", "anon", 3.1]); w.writerow([1, "libri", "test", "anon", 3.22])
        w.writerow([2, "libri", "dev", "original", 1.8])
    (exp / "ser" / "results_knn1.csv").write_text(
        ",dataset,split,ser,UAR\n,,,,mean\n0,IEMOCAP,dev,anon,56.733\n1,IEMOCAP,test,anon,56.674\n")
    m = vpc.collect(tmp_path, "_knn1", "1")
    assert m["EER_avg"] == 7.952 and m["WER_avg"] == 3.16 and abs(m["UAR_avg"] - 56.7035) < 1e-3
    assert vpc.judge("1", m) == {"EER": "ok", "UAR": "ok", "WER": "ok"}


def test_overwrite_json_is_libri_only():
    import json
    o = json.loads(vpc.overwrite_json("_knn1", "pre", True))
    assert [d["name"] for d in o["datasets"]] == ["libri_dev", "libri_test"]
    assert o["eval_steps"] == {"privacy": ["asv"], "utility": ["asr"]}
