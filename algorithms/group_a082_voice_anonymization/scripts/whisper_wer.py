"""Whisper medium.en's own WER on libri-dev-asr / libri-test-asr, from the cascade's cached transcripts.
The paper reports 3.38 % (dev) and 3.29 % (test) without saying how text was normalised, so both are given:
upper-case with punctuation removed (the VPC references' style), and Whisper's EnglishTextNormalizer.

    python scripts/whisper_wer.py --vpc-root R --transcripts work/transcripts
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import jiwer

from a082 import kaldi

SPLITS = {"dev": ["libri_dev_trials_f", "libri_dev_trials_m"], "test": ["libri_test_trials_f", "libri_test_trials_m"]}


def simple_norm(s: str) -> str:
    s = re.sub(r"[^A-Z0-9' ]", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--transcripts", required=True, type=Path)
    args = ap.parse_args()
    from whisper.normalizers import EnglishTextNormalizer
    en = EnglishTextNormalizer()
    for split, sets in SPLITS.items():
        ref, hyp = {}, {}
        for name in sets:
            ref.update(kaldi.read_kaldi(args.vpc_root / "data" / name / "text"))
            for f in args.transcripts.glob(f"{name}.*of*.txt"):
                hyp.update(kaldi.read_kaldi(f))
        utts = sorted(u for u in ref if u in hyp)
        missing = len(ref) - len(utts)
        w1 = jiwer.wer([simple_norm(ref[u]) for u in utts], [simple_norm(hyp[u]) for u in utts]) * 100
        w2 = jiwer.wer([en(ref[u]) for u in utts], [en(hyp[u]) for u in utts]) * 100
        print(f"{split}: {len(utts)} utterances ({missing} missing); WER upper/no-punct {w1:.2f} %, "
              f"Whisper normaliser {w2:.2f} % (paper: {'3.38' if split == 'dev' else '3.29'} %)")


if __name__ == "__main__":
    main()
