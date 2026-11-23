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


def uses_tts(key: str, p: float) -> bool:
    """`key` is the utterance id (the paper's per-utterance admixture) or, for the speaker-level
    variant of the bonus experiments, the speaker id."""
    return seeding.uniform01(key, "admix") < p


def build(vpc_root: Path, knn_suffix: str, tts_suffix: str, p: float, sets: list[str],
          level: str = "utt", out_suffix: str | None = None) -> dict[str, float]:
    """Create `<set><suffix>` for each set; returns the realised TTS fraction per set.
    level="spk": every utterance of a speaker comes from the same system (bonus "improvement" candidate)."""
    data = vpc_root / "data"
    out_suffix = out_suffix or (suffix_for(p) + ("spk" if level == "spk" else ""))
    frac = {}
    for name in sets:
        if not all((data / f"{name}{sfx}" / "wav.scp").exists() for sfx in (knn_suffix, tts_suffix)):
            print(f"skipping {name}: no finished {knn_suffix} / {tts_suffix} output")
            continue
        utts = list(kaldi.read_kaldi(data / name / "wav.scp"))
        u2s = kaldi.read_kaldi(data / name / "utt2spk")
        key = (lambda u: u2s[u]) if level == "spk" else (lambda u: u)
        choose = {u: kaldi.wav_path(data / f"{name}{tts_suffix if uses_tts(key(u), p) else knn_suffix}", u) for u in utts}
        kaldi.link_output_dir(data / name, data / f"{name}{out_suffix}", choose)
        frac[name] = sum(uses_tts(key(u), p) for u in utts) / len(utts)
    return frac


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--knn-suffix", default="_knn1")
    ap.add_argument("--tts-suffix", default="_tts")
    ap.add_argument("--p", type=float, nargs="+", required=True)
    ap.add_argument("--sets", default="all")
    ap.add_argument("--level", choices=["utt", "spk"], default="utt")
    ap.add_argument("--out-suffix", help="default _mix<p> (utt) or _mix<p>spk")
    args = ap.parse_args(argv)
    sets = kaldi.ALL_SETS if args.sets == "all" else args.sets.split(",")
    for p in args.p:
        for name, f in build(args.vpc_root, args.knn_suffix, args.tts_suffix, p, sets, args.level, args.out_suffix).items():
            print(f"p={p:g} {name}: TTS fraction {f:.3f}")


if __name__ == "__main__":
    main()
