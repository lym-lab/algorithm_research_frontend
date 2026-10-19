"""Run the VPC 2024 evaluation for one anonymized suffix and collect the numbers of Table 1.

    python -m a082.eval.vpc identity --vpc-root R                 # data/<set>_orig -> original audio (row 0)
    python -m a082.eval.vpc run      --vpc-root R --suffix _knn1 [--stages pre post] [--no-iemocap | --emotion crema]
                                     [--workspace auto | DIR]   # several runs at once, see a082/eval/workspace.py
    python -m a082.eval.vpc collect  --vpc-root R --suffix _knn1 [--row 1] [--json out.json]
    python -m a082.eval.vpc report   --vpc-root R --out results/summary.md

The framework's scripts are used unchanged (evaluation plan §6 forbids changing them); `run` is the
"option 2" route of its README: eval_pre with the pretrained models, then eval_post, which trains the
attacker ASV on the anonymized train-clean-360. --no-iemocap restricts both to LibriSpeech through the
framework's own --overwrite mechanism, for use while IEMOCAP is not prepared. --emotion crema scores
the organisers' IEMOCAP emotion models on CREMA-D instead (scripts/prepare_crema.py): UAR is then
cross-corpus and is not compared with the paper.

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
from . import targets, workspace as wsp

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
        if not kaldi.audio_available(data / name, vpc_root):
            print(f"skipping {name}: audio not available")
            continue
        scp = kaldi.read_kaldi(data / name / "wav.scp")
        choose = {}
        for u, entry in scp.items():
            if entry.endswith("|"):
                raise ValueError(f"{name}: piped wav.scp entries cannot be linked; row 0 needs only the eval sets")
            p = Path(entry)
            choose[u] = p if p.is_absolute() else vpc_root / p
        kaldi.link_output_dir(data / name, data / f"{name}_orig", choose)


EMOTION_CORPORA = ("iemocap", "crema", "none")


def overwrite_json(suffix: str, stage: str, no_iemocap: bool = False, emotion: str | None = None) -> str:
    """--overwrite for run_evaluation.py. `emotion`: iemocap (the framework's default), crema, or none
    (LibriSpeech only; same as no_iemocap)."""
    emotion = "none" if no_iemocap else (emotion or "iemocap")
    if emotion not in EMOTION_CORPORA:
        raise ValueError(f"emotion corpus {emotion!r}; choose from {EMOTION_CORPORA}")
    o: dict = {"anon_data_suffix": suffix}
    libri = ["libri_dev", "libri_test"]
    if emotion == "none" or (emotion == "crema" and stage != "pre"):
        o.update(LIBRI_ONLY)
        if stage == "pre":
            o["eval_steps"] = {"privacy": ["asv"], "utility": ["asr"]}
            o["utility"] = {"asr": {"dataset_name": libri}}
    elif emotion == "crema":
        crema = ["CREMA_dev", "CREMA_test"]
        o["datasets"] = LIBRI_ONLY["datasets"] + [{"name": n, "data": n} for n in crema]
        o["utility"] = {"asr": {"dataset_name": libri}, "ser": {"dataset_name": crema}}
    return json.dumps(o)


def framework(vpc_root: Path, cmds: list[str], ws: Path | str | None = None, suffix: str = "",
              fresh: tuple[str, ...] = ()) -> None:
    """Run framework commands (each after `source env.sh`) in the framework tree, or, with `ws`, in a
    separate working copy whose results are moved back afterwards (`auto`: <vpc-root>/../evalws/<suffix>).
    The working copy is kept when a command fails."""
    vpc_root = Path(vpc_root)
    cwd = vpc_root
    if ws is not None:
        # next to <vpc-root> as given, not as resolved: with the framework on the system disk and <work>/vpc2024
        # a link to it, the working copies (attacker models, features) belong on the data disk
        ws = vpc_root.absolute().parent / "evalws" / (suffix.lstrip("_") or "run") if str(ws) == "auto" else Path(ws)
        cwd = wsp.create(vpc_root, ws, suffix, fresh)
    for c in cmds:
        cmd = f"source env.sh && {c}"
        print(f"[{cwd}] {cmd}", flush=True)
        subprocess.run(["bash", "-c", cmd], cwd=cwd, check=True)
    if ws is not None:
        print(f"moved {len(wsp.merge(cwd, vpc_root))} result files into {vpc_root / 'exp'}", flush=True)


def run(vpc_root: Path, suffix: str, stages=("pre", "post"), no_iemocap=False, emotion: str | None = None,
        ws: Path | str | None = None) -> None:
    cmds = [f"python run_evaluation.py --config configs/eval_{stage}.yaml "
            f"--overwrite {shlex.quote(overwrite_json(suffix, stage, no_iemocap, emotion))} --force_compute True"
            for stage in stages]
    framework(vpc_root, cmds, ws, suffix, fresh=(f"asv_anon{suffix}",))


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
            if rec.get("dataset") in ("IEMOCAP", "CREMA") and rec.get("ser") == want:
                if out.get("UAR_corpus") not in (None, rec["dataset"]):
                    continue  # both present: keep the first (IEMOCAP is written first by our runs)
                out["UAR_corpus"] = rec["dataset"]
                out["UAR"][rec["split"]] = float(rec["UAR"])

    def avg(d, keys):
        return round(sum(d[k] for k in keys) / len(keys), 3) if all(k in d for k in keys) else None

    out["EER_avg"] = avg(out["EER"], ["dev_f", "dev_m", "test_f", "test_m"])
    out["UAR_avg"] = avg(out["UAR"], ["dev", "test"])
    out["WER_avg"] = avg(out["WER"], ["dev", "test"])
    return out


def judge(row: str, m: dict, ours: dict | None = None) -> dict:
    """Compare our averages with Table 1 under the §5 tolerances."""
    _, eer, uar, wer = targets.TABLE1[row]
    tol = targets.TOLERANCE.get(row, {})
    verdict = {}
    for key, paper in (("EER", eer), ("UAR", uar), ("WER", wer)):
        ours = m.get(f"{key}_avg")
        if ours is None:
            verdict[key] = "missing"
        elif key == "UAR" and m.get("UAR_corpus") == "CREMA":
            verdict[key] = "cross-corpus (CREMA-D), not compared"
        elif key == "EER" and "eer_min" in tol:
            verdict[key] = "ok" if ours >= tol["eer_min"] else f"below {tol['eer_min']}"
        elif key in tol:
            verdict[key] = "ok" if abs(ours - paper) <= tol[key] else f"off by {ours - paper:+.2f}"
        else:
            verdict[key] = f"{ours - paper:+.2f}"
    q = targets.qualitative(row, m, ours or {})
    if q:
        verdict["qualitative"] = q
    return verdict


def report_rows(vpc_root: Path) -> list[dict]:
    """Every Table 1 row with the paper's averages, ours (None where not measured) and the verdict."""
    rows, measured_rows = [], {}
    for row, (suffix, eer, uar, wer) in targets.TABLE1.items():
        m = collect(vpc_root, suffix, row)
        measured_rows[row] = m
        ours = {k: m[f"{k}_avg"] for k in ("EER", "UAR", "WER")}
        measured = any(v is not None for v in ours.values())
        rows.append({"row": row, "suffix": suffix, "name": targets.NAMES_ZH[row],
                     "paper": {"EER": eer, "UAR": uar, "WER": wer},
                     "ours": ours if measured else None,
                     "subsets": {k: m[k] for k in ("EER", "UAR", "WER")},
                     "verdict": judge(row, m, measured_rows) if measured else None})
    return rows


def sweep_points(vpc_root: Path) -> dict:
    """Fig. 4: admixture probability -> metrics, for the points that have results."""
    pts = {p: collect(vpc_root, sfx) for p, sfx in targets.SWEEP.items()}
    return {p: m for p, m in pts.items() if m["UAR_avg"] is not None or m["EER"]}


def fig4(vpc_root: Path, out: Path) -> None:
    from .. import viz
    pts = {p: (m["EER"]["dev_f"], m["UAR_avg"]) for p, m in sweep_points(vpc_root).items()
           if m["EER"].get("dev_f") is not None and m["UAR_avg"] is not None}
    anchors = {}
    for row, label in (("1", "kNN-VC (1)"), ("4", "cascade (4)")):
        m = collect(vpc_root, targets.TABLE1[row][0], row)
        if m["EER"].get("dev_f") is not None and m["UAR_avg"] is not None:
            anchors[label] = (m["EER"]["dev_f"], m["UAR_avg"])
    out.parent.mkdir(parents=True, exist_ok=True)
    viz.tradeoff_figure(pts, anchors).savefig(out, dpi=150)


def checks(vpc_root: Path) -> list[tuple[str, str]]:
    ours = {r: collect(vpc_root, sfx, r) for r, (sfx, *_) in targets.TABLE1.items()}
    return targets.internal_checks(ours, sweep_points(vpc_root))


def report(vpc_root: Path) -> str:
    lines = ["| row | suffix | EER avg (paper) | UAR avg (paper) | WER avg (paper) | verdict |",
             "|---|---|---|---|---|---|"]
    for r in report_rows(vpc_root):
        o = r["ours"] or {}
        fmt = lambda k: f"{o.get(k) if o.get(k) is not None else '—'} ({r['paper'][k]})"
        v = "; ".join(f"{k}: {s}" for k, s in (r["verdict"] or {}).items()) or "not measured"
        lines.append(f"| {r['row']} | `{r['suffix']}` | {fmt('EER')} | {fmt('UAR')} | {fmt('WER')} | {v} |")
    internal = checks(vpc_root)
    if internal:
        lines += ["", "| internal check | verdict |", "|---|---|"] + [f"| {n} | {v} |" for n, v in internal]
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
    sub.choices["run"].add_argument("--emotion", choices=EMOTION_CORPORA, help="emotion corpus for UAR (default iemocap)")
    sub.choices["run"].add_argument("--workspace", help="'auto' or a directory: run in a separate working copy, "
                                    "so that several evaluations can run at once (one GPU each)")
    sub.choices["collect"].add_argument("--row")
    sub.choices["collect"].add_argument("--json", type=Path)
    sub.choices["report"].add_argument("--out", type=Path)
    sub.choices["report"].add_argument("--json", type=Path, help="rows for the web page's results view")
    sub.choices["report"].add_argument("--fig4", type=Path, help="PNG of the admixture sweep (paper Fig. 4)")
    args = ap.parse_args(argv)

    if args.cmd == "identity":
        sets = kaldi.EVAL_SETS if args.sets == "eval" else args.sets.split(",")
        make_identity(args.vpc_root, sets)
    elif args.cmd == "run":
        run(args.vpc_root, args.suffix, args.stages, args.no_iemocap, args.emotion, args.workspace)
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
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report_rows(args.vpc_root), ensure_ascii=False, indent=1), encoding="utf-8")
        if args.fig4:
            fig4(args.vpc_root, args.fig4)


if __name__ == "__main__":
    main()
