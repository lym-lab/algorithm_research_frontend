"""IEMOCAP sentence audio from the Hugging Face copy AbstractTTS/IEMOCAP, laid out as IEMOCAP_full_release.

    python scripts/prepare_iemocap_hf.py --vpc-root <VPC repo> --out <dir> [--parquet-dir <dir>]

The copy holds all 10,039 utterances as the original 16 kHz 16-bit mono WAV bytes (one parquet column),
named <utt>.wav. They are written unchanged to <out>/Session<n>/sentences/wav/<dialog>/<utt>.wav, the
layout the framework's data/IEMOCAP_{dev,test}/wav.scp point into; <out> is then linked as
<vpc-root>/data/IEMOCAP/wav. Labels, speakers and folds still come from the framework's data.zip.

Checks: every file is PCM 16 kHz mono 16-bit, and every utterance listed in IEMOCAP_dev and IEMOCAP_test
(5,531) exists after linking.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

import soundfile as sf

REPO = "AbstractTTS/IEMOCAP"
REVISION = "9f1696a135a65ce997d898d4121c952269a822ca"  # 2024-08-11, checked on 2026-10-19
N_UTTERANCES = 10039
SETS = ("IEMOCAP_dev", "IEMOCAP_test")


def target(out: Path, utt: str) -> Path:
    """Ses01F_impro01_F000 -> Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"""
    return out / f"Session{int(utt[3:5])}" / "sentences" / "wav" / utt.rsplit("_", 1)[0] / f"{utt}.wav"


def check_wav(data: bytes, utt: str) -> None:
    info = sf.info(io.BytesIO(data))
    if (info.format, info.subtype, info.samplerate, info.channels) != ("WAV", "PCM_16", 16000, 1):
        raise SystemExit(f"{utt}: {info.format}/{info.subtype} {info.samplerate} Hz {info.channels} ch, "
                         "expected WAV/PCM_16 16000 Hz mono")


def write_all(rows: Iterable[tuple[str, bytes]], out: Path) -> int:
    n = 0
    for name, data in rows:
        utt = Path(name).name.removesuffix(".wav")
        check_wav(data, utt)
        dst = target(out, utt)
        if not (dst.exists() and dst.stat().st_size == len(data)):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        n += 1
    return n


def parquet_rows(files: list[Path]) -> Iterator[tuple[str, bytes]]:
    import pyarrow.parquet as pq
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(columns=["file", "audio"], batch_size=256):
            for name, audio in zip(batch.column("file").to_pylist(), batch.column("audio").to_pylist()):
                yield name or audio["path"], audio["bytes"]


def download(dst: Path) -> list[Path]:
    import time
    from huggingface_hub import snapshot_download
    for attempt in range(8):  # slow mirrors break connections; each attempt resumes the partial files
        try:
            snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=dst,
                              allow_patterns=["data/*.parquet"])
            break
        except Exception as e:  # noqa: BLE001 - requests and hub errors alike
            if attempt == 7:
                raise
            print(f"download interrupted ({type(e).__name__}), retry {attempt + 1}", file=sys.stderr)
            time.sleep(10)
    return sorted((dst / "data").glob("*.parquet"))


def link_and_check(vpc_root: Path, out: Path) -> int:
    link = vpc_root / "data" / "IEMOCAP" / "wav"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(out.resolve(), link, target_is_directory=True)
    missing, total = [], 0
    for name in SETS:
        for line in (vpc_root / "data" / name / "wav.scp").read_text().splitlines():
            utt, path = line.split(maxsplit=1)
            total += 1
            if not (vpc_root / path).exists():
                missing.append(utt)
    if missing:
        raise SystemExit(f"{len(missing)} of {total} IEMOCAP evaluation utterances missing, e.g. {missing[:5]}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="where IEMOCAP_full_release-style audio is written")
    ap.add_argument("--parquet-dir", type=Path, help="download cache (default <out>/../iemocap_hf)")
    a = ap.parse_args()
    files = download(a.parquet_dir or a.out.parent / "iemocap_hf")
    n = write_all(parquet_rows(files), a.out)
    if n != N_UTTERANCES:
        print(f"WARNING: {n} utterances, expected {N_UTTERANCES}")
    total = link_and_check(a.vpc_root, a.out)
    print(f"{n} utterances written under {a.out}; all {total} IEMOCAP_dev/test utterances present")


if __name__ == "__main__":
    main()
