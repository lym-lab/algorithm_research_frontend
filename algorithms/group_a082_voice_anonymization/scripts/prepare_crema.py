"""Fallback emotion data when IEMOCAP is not available: CREMA-D as data/CREMA_dev and data/CREMA_test.

    python scripts/prepare_crema.py --vpc-root <VPC repo> --crema <CREMA-D checkout> [--label-ref data/IEMOCAP_dev/utt2emo]

CREMA-D (ODbL): 7,442 clips of 91 actors, file names <actor>_<sentence>_<emotion>_<level>.wav. Only the four
IEMOCAP classes are kept (ANG, HAP, NEU, SAD -> ang, hap, neu, sad), so the organisers' IEMOCAP emotion
models can score them; UAR is then cross-corpus and not comparable with the paper's numbers.

Actors are split into dev and test alternately (by sex when VideoDemographics.csv is present, so both halves
are balanced), and each actor is assigned to one of the five IEMOCAP fold models in turn (spk2fold), which
is how the framework's SER evaluation picks the model per speaker. Files that are not 16 kHz 16-bit mono
are converted into data/CREMA/wav.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a082 import kaldi  # noqa: E402

EMOTIONS = {"ANG": "ang", "HAP": "hap", "NEU": "neu", "SAD": "sad"}
SETS = ("CREMA_dev", "CREMA_test")
N_FOLDS = 5


def split_actors(actors: list[str], sex: dict[str, str]) -> dict[str, str]:
    """actor -> 'CREMA_dev' / 'CREMA_test', alternating within each sex (or over all actors)."""
    groups: dict[str, list[str]] = {}
    for a in sorted(actors):
        groups.setdefault(sex.get(a, "?"), []).append(a)
    return {a: SETS[i % 2] for g in groups.values() for i, a in enumerate(g)}


def read_sex(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["ActorID"]: {"Male": "m", "Female": "f"}.get(r["Sex"], "?") for r in csv.DictReader(f)}


def to_16k(src: Path, dst: Path) -> Path:
    info = sf.info(src)
    if info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16":
        return src.absolute()
    import torch
    import torchaudio
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    x = torch.from_numpy(wav.T.copy()).mean(0, keepdim=True)
    x = torchaudio.functional.resample(x, sr, 16000)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x[0].numpy(), 16000, subtype="PCM_16")
    return dst.absolute()


def prepare(vpc_root: Path, crema: Path, label_ref: Path | None = None) -> dict[str, int]:
    wav_dir = crema / "AudioWAV" if (crema / "AudioWAV").is_dir() else crema
    clips = {}
    for f in sorted(wav_dir.glob("*.wav")):
        parts = f.stem.split("_")
        if len(parts) == 4 and parts[2] in EMOTIONS:
            clips[f.stem] = (parts[0], EMOTIONS[parts[2]], f)
    if not clips:
        raise SystemExit(f"no CREMA-D clips of the four classes under {wav_dir}")
    if label_ref is not None and label_ref.exists():
        ref = set(kaldi.read_kaldi(label_ref).values())
        if ref != set(EMOTIONS.values()):
            raise SystemExit(f"label mismatch: IEMOCAP uses {sorted(ref)}, we write {sorted(EMOTIONS.values())}")

    sex = read_sex(crema / "VideoDemographics.csv")
    split = split_actors(sorted({a for a, _, _ in clips.values()}), sex)
    counts = {}
    for name in SETS:
        out = vpc_root / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        utts = {u: c for u, c in clips.items() if split[c[0]] == name}
        actors = sorted({a for a, _, _ in utts.values()})
        conv = vpc_root / "data" / "CREMA" / "wav"
        kaldi.write_kaldi({u: str(to_16k(f, conv / f.name)) for u, (_, _, f) in utts.items()}, out / "wav.scp")
        kaldi.write_kaldi({u: a for u, (a, _, _) in utts.items()}, out / "utt2spk")
        kaldi.write_kaldi({u: e for u, (_, e, _) in utts.items()}, out / "utt2emo")
        spk2utt: dict[str, list[str]] = {}
        for u, (a, _, _) in sorted(utts.items()):
            spk2utt.setdefault(a, []).append(u)
        kaldi.write_kaldi({a: " ".join(us) for a, us in spk2utt.items()}, out / "spk2utt")
        kaldi.write_kaldi({a: str(i % N_FOLDS + 1) for i, a in enumerate(actors)}, out / "spk2fold")
        if sex:
            kaldi.write_kaldi({a: sex.get(a, "?") for a in actors}, out / "spk2gender")
        counts[name] = len(utts)
        print(f"{name}: {len(utts)} clips, {len(actors)} actors")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--crema", required=True, type=Path, help="CREMA-D checkout (with AudioWAV/) or the wav folder")
    ap.add_argument("--label-ref", type=Path, help="IEMOCAP utt2emo from data.zip, to check the label names")
    a = ap.parse_args()
    prepare(a.vpc_root, a.crema, a.label_ref)


if __name__ == "__main__":
    main()
