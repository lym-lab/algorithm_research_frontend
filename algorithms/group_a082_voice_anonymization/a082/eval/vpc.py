"""Run the VPC 2024 evaluation for one anonymized suffix and collect the numbers of Table 1.

    python -m a082.eval.vpc identity --vpc-root R                 # data/<set>_orig -> original audio (row 0)
    python -m a082.eval.vpc run      --vpc-root R --suffix _knn1 [--stages pre post] [--no-iemocap]
    python -m a082.eval.vpc collect  --vpc-root R --suffix _knn1 [--row 1] [--json out.json]
    python -m a082.eval.vpc report   --vpc-root R --out results/summary.md

The framework's scripts are used unchanged (evaluation plan §6 forbids changing them); `run` is the
"option 2" route of its README: eval_pre with the pretrained models, then eval_post, which trains the
attacker ASV on the anonymized train-clean-360. --no-iemocap restricts both to LibriSpeech through the
framework's own --overwrite mechanism, for use before the IEMOCAP licence arrives.

Where the numbers come from (CSV files written by the framework):
  EER (row 0, original-original):  exp/asv_orig/results<suffix>.csv, enrollment = trial = original
  EER (anonymized rows, AA):       exp/asv_anon<suffix>/results<suffix>.csv, enrollment = trial = anon
  WER:                             exp/asr/results<suffix>.csv, asr = anon (row 0: original)
  UAR:                             exp/ser/results<suffix>.csv, ser = anon (row 0: original)
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path

from .. import kaldi
from . import targets

LIBRI_ONLY = {
    "datasets": [
        {"name": "libri_dev", "data": "libri_dev", "enrolls": ["_enrolls"], "trials": ["_trials_f", "_trials_m"]},
        {"name": "libri_test", "data": "libri_test", "enrolls": ["_enrolls"], "trials": ["_trials_f", "_trials_m"]},
    ],
}


def make_identity(vpc_root: Path, sets: list[str]) -> None:
    """data/<set>_orig with links to the original files, so row 0 runs through the same pipeline."""
    data = vpc_root / "data"
    for name in sets:
        scp = kaldi.read_kaldi(data / name / "wav.scp")
        choose = {}
        for u, entry in scp.items():
            if entry.endswith("|"):
                raise ValueError(f"{name}: piped wav.scp entries cannot be linked; row 0 needs only the eval sets")
            p = Path(entry)
            choose[u] = p if p.is_absolute() else vpc_root / p
        kaldi.link_output_dir(data / name, data / f"{name}_orig", choose)


def overwrite_json(suffix: str, stage: str, no_iemocap: bool) -> str:
    o: dict = {"anon_data_suffix": suffix}
    if no_iemocap:
        o.update(LIBRI_ONLY)
        if stage == "pre":
            o["eval_steps"] = {"privacy": ["asv"], "utility": ["asr"]}
            o["utility"] = {"asr": {"dataset_name": ["libri_dev", "libri_test"]}}
    return json.dumps(o)


def run(vpc_root: Path, suffix: str, stages=("pre", "post"), no_iemocap=False) -> None:
    for stage in stages:
        cmd = (f"source env.sh && python run_evaluation.py --config configs/eval_{stage}.yaml "
               f"--overwrite {shlex.quote(overwrite_json(suffix, stage, no_iemocap))} --force_compute True")
        print(f"[{vpc_root}] {cmd}", flush=True)
        subprocess.run(["bash", "-c", cmd], cwd=vpc_root, check=True)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect(vpc_root: Path, suffix: str, row: str | None = None) -> dict:
    """Return {'EER': {dev_f, dev_m, test_f, test_m}, 'UAR': {dev, test}, 'WER': {dev, test}, *_avg}."""
    exp = vpc_root / "exp"
    original = row == "0"
    out: dict = {"suffix": suffix, "EER": {}, "UAR": {}, "WER": {}}

    asv_csv = exp / "asv_orig" / f"results{suffix}.csv" if original else exp / f"asv_anon{suffix}" / f"results{suffix}.csv"
    want = "original" if original else "anon"
    if asv_csv.exists():
        for r in _rows(asv_csv):
            if r["enrollment"] == want and r["trial"] == want and r["dataset"] == "libri":
                out["EER"][f"{r['split']}_{r['gender']}"] = float(r["EER"])

    asr_csv = exp / "asr" / f"results{suffix}.csv"
    if asr_csv.exists():
        for r in _rows(asr_csv):
            if r["dataset"] == "libri" and r["asr"] == want:
                out["WER"][r["split"]] = float(r["WER"])

    ser_csv = exp / "ser" / f"results{suffix}.csv"
    if ser_csv.exists():
        # written from a two-level header (UAR, mean); the first two lines are header rows
        lines = ser_csv.read_text().splitlines()
        header = lines[0].split(",")
        for line in lines[2:]:
            cells = line.split(",")
            rec = dict(zip(header, cells))
            if rec.get("dataset") == "IEMOCAP" and rec.get("ser") == want:
                out["UAR"][rec["split"]] = float(rec["UAR"])

    def avg(d, keys):
        return round(sum(d[k] for k in keys) / len(keys), 3) if all(k in d for k in keys) else None

    out["EER_avg"] = avg(out["EER"], ["dev_f", "dev_m", "test_f", "test_m"])
    out["UAR_avg"] = avg(out["UAR"], ["dev", "test"])
    out["WER_avg"] = avg(out["WER"], ["dev", "test"])
    return out


def judge(row: str, m: dict, base5: dict | None = None) -> dict:
    """Compare our averages with Table 1 under the §5 tolerances."""
    _, eer, uar, wer = targets.TABLE1[row]
    tol = targets.TOLERANCE.get(row, {})
    verdict = {}
    for key, paper in (("EER", eer), ("UAR", uar), ("WER", wer)):
        ours = m.get(f"{key}_avg")
        if ours is None:
            verdict[key] = "missing"
        elif key == "EER" and "eer_min" in tol:
            verdict[key] = "ok" if ours >= tol["eer_min"] else f"below {tol['eer_min']}"
        elif key in tol:
            verdict[key] = "ok" if abs(ours - paper) <= tol[key] else f"off by {ours - paper:+.2f}"
        else:
            verdict[key] = f"{ours - paper:+.2f}"
    q = targets.qualitative(row, m, base5)
    if q:
        verdict["qualitative"] = q
    return verdict


def report(vpc_root: Path) -> str:
    lines = ["| row | suffix | EER avg (paper) | UAR avg (paper) | WER avg (paper) | verdict |",
             "|---|---|---|---|---|---|"]
    base5 = None
    for row, (suffix, eer, uar, wer) in targets.TABLE1.items():
        m = collect(vpc_root, suffix, row)
        if row == "5":
            base5 = m
        v = judge(row, m, base5)
        fmt = lambda x, p: f"{x if x is not None else '—'} ({p})"
        lines.append(f"| {row} | `{suffix}` | {fmt(m['EER_avg'], eer)} | {fmt(m['UAR_avg'], uar)} | "
                     f"{fmt(m['WER_avg'], wer)} | {'; '.join(f'{k}: {s}' for k, s in v.items())} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("identity", "run", "collect", "report"):
        a = sub.add_parser(name)
        a.add_argument("--vpc-root", required=True, type=Path)
        if name in ("run", "collect"):
            a.add_argument("--suffix", required=True)
    sub.choices["identity"].add_argument("--sets", default="eval")
    sub.choices["run"].add_argument("--stages", nargs="+", default=["pre", "post"])
    sub.choices["run"].add_argument("--no-iemocap", action="store_true")
    sub.choices["collect"].add_argument("--row")
    sub.choices["collect"].add_argument("--json", type=Path)
    sub.choices["report"].add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "identity":
        sets = kaldi.EVAL_SETS if args.sets == "eval" else args.sets.split(",")
        make_identity(args.vpc_root, sets)
    elif args.cmd == "run":
        run(args.vpc_root, args.suffix, args.stages, args.no_iemocap)
    elif args.cmd == "collect":
        m = collect(args.vpc_root, args.suffix, args.row)
        if args.row:
            m["verdict"] = judge(args.row, m)
        text = json.dumps(m, indent=1)
        print(text)
        if args.json:
            args.json.write_text(text)
    else:
        text = report(args.vpc_root)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
