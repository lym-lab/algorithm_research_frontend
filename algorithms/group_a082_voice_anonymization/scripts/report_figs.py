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
        ("3", "_knn3d18", "3"), ("4", "_tts", "4"), ("p=0.2", "_mix0p2", None), ("p=0.325", "_mix0p325", None),
        ("p=0.4", "_mix0p4", None), ("A", "_mixA0p4", None), ("B", "_mix0p4spk", None),
        # ID 5-8 (wc7, wc8: the corrected 2026-11-22 runs); a row without an evaluation is skipped
        ("5", "_wc5", "5"), ("6", "_wc6", "6"), ("7", "_wc7", "7"), ("8", "_wc8", "8")]
# label positions in points; per panel (1 = UAR, 2 = WER) where the WavLM rows crowd around WER 2.5-2.8
OFFSETS = {"p=0.4": (-34, 4), "A": (4, -11), "1": {1: (-12, 3), 2: (-13, -6)}, "2": (4, 4), "5": (4, 4),
           "6": {1: (4, 4), 2: (-12, 5)}, "8": {1: (4, 3), 2: (6, -5)}, "7": (5, -3), "1M": {1: (4, 3), 2: (-17, -4)},
           "B": {1: (4, -11), 2: (5, -3)}}
# magnified inset of the WER panel for rows 0-2, 1M and 5-8 (WER 1.7-3.0, EER 4-14)
WER_ZOOM = (1.7, 3.0, 4.0, 14.0)
ZOOM_OFFSETS = {"1": (-12, -3), "2": (5, -1), "6": (-11, 3), "8": (5, -5), "1M": (-18, -3), "7": (5, -3),
                "5": (5, -3), "0": (5, -3)}


def result_files(paths: list[Path]) -> dict[str, dict]:
    """suffix -> the JSON an evaluation wrote (result_*.txt), the fallback when exp/ lacks one of its CSVs."""
    import json
    found = {}
    for p in paths:
        for f in ([p] if p.is_file() else sorted(p.glob("result*.txt"))):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(r, dict) and r.get("suffix"):
                found.setdefault(r["suffix"], r)
    return found
CASE = [("original", "_orig_src"), ("ID 1  kNN-VC", "_knn1"), ("ID 3  + length + noise (δ=18)", "_knn3d18"),
        ("ID 4  Whisper → Piper", "_tts"), ("improvement A  tts → kNN-VC", "_ttsknn")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--utt", help="utterance id for figure 3-3 (default: the first one present everywhere)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--results", type=Path, action="append", default=[],
                    help="result_*.txt file or folder (repeatable): averages used where exp/ lacks a CSV")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    fallback = result_files(args.results)
    rows = {}
    for label, sfx, row in ROWS:
        m = vpc.collect(args.vpc_root, sfx, row)
        for key in ("EER_avg", "UAR_avg", "WER_avg"):
            if m[key] is None and fallback.get(sfx, {}).get(key) is not None:
                m[key] = fallback[sfx][key]
                print(f"{sfx}: {key} from the result file")
        if None not in (m["EER_avg"], m["UAR_avg"], m["WER_avg"]):
            rows[label] = (m["EER_avg"], m["UAR_avg"], m["WER_avg"])
    viz.privacy_utility_figure(rows, offsets=OFFSETS, wer_zoom=WER_ZOOM, zoom_offsets=ZOOM_OFFSETS).savefig(
        args.out / "fig3-2_privacy_utility.png", dpi=150)
    print("figure 3-2:", ", ".join(f"{k}={v}" for k, v in rows.items()))

    if args.samples:
        dirs = [(t, args.samples / s) for t, s in CASE if (args.samples / s).is_dir()]
        common = set.intersection(*({p.stem for p in d.glob("*.wav")} for _, d in dirs))
        utt = args.utt or sorted(common)[0]
        sig = {title: audio.load_audio(d / f"{utt}.wav") for title, d in dirs}
        viz.systems_figure(sig, seconds=6.0).savefig(args.out / "fig3-3_spectrograms.png", dpi=150)
        print(f"figure 3-3: utterance {utt}")


if __name__ == "__main__":
    main()

