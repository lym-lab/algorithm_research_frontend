"""Admixture systems (Table 1 rows "1 + 4*", Fig. 4): per utterance, take the cascade (system 4)
output with probability p, otherwise the kNN-VC (system 1) output.

    python -m a082.admix --vpc-root <VPC repo> --knn-suffix _knn1 --tts-suffix _tts --p 0.2 0.325 0.4 [--sets all]

Nothing is re-anonymized: the output directories link to the two systems' files. Each utterance has one
draw xi_u in [0, 1) shared by every p, so the set of TTS utterances only grows with p.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import kaldi, seeding


def suffix_for(p: float) -> str:
    return "_mix" + f"{p:g}".replace(".", "p")


def uses_tts(utt: str, p: float) -> bool:
    return seeding.uniform01(utt, "admix") < p


def build(vpc_root: Path, knn_suffix: str, tts_suffix: str, p: float, sets: list[str]) -> dict[str, float]:
    """Create `<set><suffix_for(p)>` for each set; returns the realised TTS fraction per set."""
    data = vpc_root / "data"
    frac = {}
    for name in sets:
        utts = list(kaldi.read_kaldi(data / name / "wav.scp"))
        choose = {u: kaldi.wav_path(data / f"{name}{tts_suffix if uses_tts(u, p) else knn_suffix}", u) for u in utts}
        kaldi.link_output_dir(data / name, data / f"{name}{suffix_for(p)}", choose)
        frac[name] = sum(uses_tts(u, p) for u in utts) / len(utts)
    return frac


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--knn-suffix", default="_knn1")
    ap.add_argument("--tts-suffix", default="_tts")
    ap.add_argument("--p", type=float, nargs="+", required=True)
    ap.add_argument("--sets", default="all")
    args = ap.parse_args(argv)
    sets = kaldi.ALL_SETS if args.sets == "all" else args.sets.split(",")
    for p in args.p:
        for name, f in build(args.vpc_root, args.knn_suffix, args.tts_suffix, p, sets).items():
            print(f"p={p:g} {name}{suffix_for(p)}: TTS fraction {f:.3f}")


if __name__ == "__main__":
    main()
