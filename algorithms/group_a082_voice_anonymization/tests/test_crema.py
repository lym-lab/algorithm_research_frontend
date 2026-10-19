"""CREMA-D fallback: data preparation, evaluation overrides and result parsing (synthetic files)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from a082 import kaldi
from a082.eval import vpc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_crema  # noqa: E402


def make_crema(root: Path, sr=16000):
    wav = root / "AudioWAV"
    wav.mkdir(parents=True)
    rows = ["ActorID,Age,Sex,Race,Ethnicity"]
    for i, actor in enumerate(range(1001, 1013)):
        rows.append(f"{actor},30,{'Male' if i % 2 else 'Female'},x,y")
        for emo in ("ANG", "HAP", "NEU", "SAD", "DIS", "FEA"):
            sf.write(wav / f"{actor}_IEO_{emo}_HI.wav", np.zeros(sr // 10, dtype="float32"), sr, subtype="PCM_16")
    (root / "VideoDemographics.csv").write_text("\n".join(rows) + "\n")


def test_prepare_crema(tmp_path):
    make_crema(tmp_path / "crema", sr=22050)       # not 16 kHz: converted into data/CREMA/wav
    counts = prepare_crema.prepare(tmp_path / "vpc", tmp_path / "crema")
    assert counts == {"CREMA_dev": 24, "CREMA_test": 24}  # 12 actors x 4 kept classes, split 6 / 6
    d = tmp_path / "vpc" / "data" / "CREMA_dev"
    assert set(kaldi.read_kaldi(d / "utt2emo").values()) == {"ang", "hap", "neu", "sad"}
    gender = kaldi.read_kaldi(d / "spk2gender")
    assert sorted(gender.values()) == ["f"] * 3 + ["m"] * 3
    assert set(kaldi.read_kaldi(d / "spk2fold").values()) <= {"1", "2", "3", "4", "5"}
    first = Path(next(iter(kaldi.read_kaldi(d / "wav.scp").values())))
    assert sf.info(first).samplerate == 16000 and kaldi.audio_available(d, tmp_path / "vpc")


def test_prepare_crema_checks_label_names(tmp_path):
    make_crema(tmp_path / "crema")
    ref = tmp_path / "utt2emo"
    ref.write_text("Ses01F_impro01_F000 neu\nSes01F_impro01_F001 exc\n")
    with pytest.raises(SystemExit):
        prepare_crema.prepare(tmp_path / "vpc", tmp_path / "crema", ref)


def test_overwrite_json_crema():
    pre = json.loads(vpc.overwrite_json("_knn1", "pre", emotion="crema"))
    assert [d["name"] for d in pre["datasets"]] == ["libri_dev", "libri_test", "CREMA_dev", "CREMA_test"]
    assert pre["utility"]["ser"]["dataset_name"] == ["CREMA_dev", "CREMA_test"]
    assert "eval_steps" not in pre
    post = json.loads(vpc.overwrite_json("_knn1", "post", emotion="crema"))
    assert [d["name"] for d in post["datasets"]] == ["libri_dev", "libri_test"]
    assert json.loads(vpc.overwrite_json("_knn1", "pre")) == {"anon_data_suffix": "_knn1"}


def test_collect_crema_uar_is_not_compared(tmp_path):
    (tmp_path / "exp" / "ser").mkdir(parents=True)
    (tmp_path / "exp" / "ser" / "results_knn1.csv").write_text(
        ",dataset,split,ser,UAR\n,,,,mean\n0,CREMA,dev,anon,40.1\n1,CREMA,test,anon,41.3\n")
    m = vpc.collect(tmp_path, "_knn1", "1")
    assert m["UAR_corpus"] == "CREMA" and m["UAR_avg"] == 40.7
    assert vpc.judge("1", m)["UAR"].startswith("cross-corpus")
