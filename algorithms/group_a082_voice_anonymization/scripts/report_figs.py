"""Figures 3-2 and 3-3 of the report from finished evaluations and a folder of sample outputs.

    python scripts/report_figs.py --vpc-root <dir with exp/> --samples <dir> --out results/

--vpc-root: a VPC root, or any folder holding a copy of its exp/ (results*.csv are enough).
--samples: one subfolder per suffix (_orig_src for the source audio, _knn1, _tts, ...), the same
utterance ids in each, as written by the cloud helper that collected them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from a082 import audio, viz
from a082.eval import vpc

ROWS = [("0", "_orig", "0"), ("1", "_knn1", "1"), ("1M", "_knn1m", None), ("2", "_knn2", "2"),
        ("3", "_knn3d18", "3"), ("4", "_tts", "4"), ("p=0.2", "_mix0p2", None), ("p=0.4", "_mix0p4", None),
        ("A", "_mixA0p4", None), ("B", "_mix0p4spk", None),
        # ID 5-8; collect() skips any row whose evaluation is missing, so wc6-8 stay out for now
        ("5", "_wc5", "5"), ("6", "_wc6", "6"), ("7", "_wc7", "7"), ("8", "_wc8", "8")]
CASE = [("original", "_orig_src"), ("ID 1  kNN-VC", "_knn1"), ("ID 3  + length + noise (δ=18)", "_knn3d18"),
        ("ID 4  Whisper → Piper", "_tts"), ("improvement A  tts → kNN-VC", "_ttsknn")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--utt", help="utterance id for figure 3-3 (default: the first one present everywhere)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for label, sfx, row in ROWS:
        m = vpc.collect(args.vpc_root, sfx, row)
        if None not in (m["EER_avg"], m["UAR_avg"], m["WER_avg"]):
            rows[label] = (m["EER_avg"], m["UAR_avg"], m["WER_avg"])
    offsets = {"p=0.4": (-34, 4), "A": (4, -11), "1": (-12, 3), "2": (4, 4), "5": (4, 4)}
    viz.privacy_utility_figure(rows, offsets=offsets).savefig(args.out / "fig3-2_privacy_utility.png", dpi=150)

    if args.samples:
        dirs = [(t, args.samples / s) for t, s in CASE if (args.samples / s).is_dir()]
        common = set.intersection(*({p.stem for p in d.glob("*.wav")} for _, d in dirs))
        utt = args.utt or sorted(common)[0]
        sig = {title: audio.load_audio(d / f"{utt}.wav") for title, d in dirs}
        viz.systems_figure(sig, seconds=6.0).savefig(args.out / "fig3-3_spectrograms.png", dpi=150)
        print(f"figure 3-3: utterance {utt}")


if __name__ == "__main__":
    main()

