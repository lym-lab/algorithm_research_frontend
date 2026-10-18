"""Rebuild data/libri_dev and data/libri_test of the VPC 2024 framework from OpenSLR LibriSpeech
(dev-clean, test-clean), for when the organisers' sftp password is not available.

    python scripts/rebuild_libri.py --vpc-root <VPC repo> --librispeech <dir containing dev-clean/ and test-clean/>

The subset directories in data.zip (libri_dev_enrolls, libri_dev_trials_f, ...) already exist after
`01_download_data_model.sh` unpacks data.zip; their wav.scp entries point to
data/libri_dev/wav/<utt>/<utt>.wav, with the original LibriSpeech utterance ids. This script writes those
files as 16-bit wav and a data/libri_<x>/wav.scp, which makes the download script skip the sftp step.
It then checks every listed file exists and, where the subset has utt2dur, that durations agree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

SETS = {"libri_dev": "dev-clean", "libri_test": "test-clean"}
SUBSETS = ["_enrolls", "_trials_f", "_trials_m"]


def read_kaldi(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            k, _, v = line.strip().partition(" ")
            out[k] = v.strip()
    return out


def flac_path(libri: Path, part: str, utt: str) -> Path:
    spk, chapter, _ = utt.split("-")
    return libri / part / spk / chapter / f"{utt}.flac"


def rebuild(vpc_root: Path, libri: Path) -> int:
    errors = 0
    for name, part in SETS.items():
        entries: dict[str, str] = {}
        durations: dict[str, float] = {}
        for sub in SUBSETS:
            d = vpc_root / "data" / f"{name}{sub}"
            entries.update(read_kaldi(d / "wav.scp"))
            if (d / "utt2dur").exists():
                durations.update({u: float(v) for u, v in read_kaldi(d / "utt2dur").items()})
        for utt, rel in sorted(entries.items()):
            dst = vpc_root / rel
            if not dst.exists():
                data, sr = sf.read(flac_path(libri, part, utt), dtype="int16")
                assert sr == 16000, (utt, sr)
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst, data, sr, subtype="PCM_16")
            info = sf.info(dst)
            if utt in durations and abs(info.frames / info.samplerate - durations[utt]) > 0.01:
                print(f"duration mismatch {utt}: {info.frames / info.samplerate:.3f} vs {durations[utt]}")
                errors += 1
        top = vpc_root / "data" / name
        top.mkdir(parents=True, exist_ok=True)
        with open(top / "wav.scp", "w", encoding="utf-8", newline="\n") as f:
            for utt in sorted(entries):
                f.write(f"{utt} {entries[utt]}\n")
        print(f"{name}: {len(entries)} utterances from {part}, {sum(u in durations for u in entries)} durations checked")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--librispeech", required=True, type=Path)
    args = ap.parse_args()
    sys.exit(1 if rebuild(args.vpc_root, args.librispeech) else 0)


if __name__ == "__main__":
    main()
