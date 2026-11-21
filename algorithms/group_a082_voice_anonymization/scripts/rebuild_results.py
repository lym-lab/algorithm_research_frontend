"""Rebuild Table 1 and the summary from finished evaluation JSONs.

Pure standard library: no torch, no a082 import, so it runs anywhere.

Each evaluation writes a small JSON like

    {"suffix": "_knn1", "EER": {...}, "UAR": {...}, "WER": {...},
     "EER_avg": 12.241, "UAR_avg": 55.136, "WER_avg": 2.699, ...}

This script scans the given paths (directories or single files), picks up every
such JSON object, and emits:

    table1.md     main table: ID 0-8 plus the three submitted mixtures
    table1.json   the same, machine readable
    summary.md    paper comparison + internal consistency checks
    extras.md     controls and bonus systems that are not rows of Table 1

Usage
-----
    python scripts/rebuild_results.py \
        --scan <logs dir> \
        --scan <first_checks.txt> \
        --scan <result_wc5.txt> \
        --out results
"""
from __future__ import annotations

import argparse
import re
import json
from pathlib import Path

# (label, suffix, paper EER, paper UAR, paper WER)  -- paper = None for extra rows
MAIN_ROWS = [
    ("0", "_orig", 5.16, 70.07, 1.83),
    ("1", "_knn1", 7.95, 56.70, 3.16),
    ("2", "_knn2", 8.08, 56.29, 3.33),
    ("3", "_knn3d18", 20.58, 43.26, 11.00),
    ("4", "_tts", 48.25, 30.35, 3.75),
    ("mix p=0.2", "_mix0p2", 21.91, 51.71, 3.31),
    ("mix p=0.325", "_mix0p325", 33.91, 49.04, 3.43),
    ("mix p=0.4", "_mix0p4", 40.81, 47.09, 3.33),
    ("5", "_wc5", 8.54, 54.75, 3.01),
    ("6", "_wc6", 11.60, 50.62, 4.23),
    ("7", "_wc7", 13.73, 43.56, 10.16),
    ("8", "_wc8", 13.88, 43.50, 10.83),
]

EXTRA_ROWS = [
    ("1M  multi-target", "_knn1m"),
    ("3 control  delta=16", "_knn3"),
    ("1 variant  v1 full", "_knn1v1full"),
    ("1 variant  v2", "_knn1v2"),
    ("mix p=0.1", "_mix0p1"),
    ("mix p=0.15", "_mix0p15"),
    ("mix p=0.25", "_mix0p25"),
    ("mix p=0.3", "_mix0p3"),
    ("mix p=0.35", "_mix0p35"),
    ("mix p=0.5", "_mix0p5"),
    ("mix p=0.4 per-speaker", "_mix0p4spk"),
    ("improvement A  tts->kNN", "_mixA0p4"),
    ("ID1 + Opus 12k", "_knn1_opus12k"),
    ("ID4 + Opus 12k", "_tts_opus12k"),
    ("mix p=0.4 + Opus 12k", "_mix0p4_opus12k"),
    ("ID0 + Opus 12k", "_orig_opus12k"),
    ("7 control  wrong LR schedule (superseded run)", "_wc7_lrwarm"),
    ("8 control  wrong LR schedule, fp32 (superseded run)", "_wc8_lrwarm"),
]

TOL = 2.0   # "ok" if within this many points of the paper


def scan(paths: list[Path]) -> dict[str, dict]:
    """Collect every pretty-printed JSON object that carries an 'EER_avg' key."""
    found: dict[str, dict] = {}
    decoder = json.JSONDecoder()
    for p in paths:
        files = [p] if p.is_file() else sorted(p.rglob("*.txt"))
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            pos = 0
            while True:
                idx = text.find('"EER_avg"', pos)
                if idx < 0:
                    break
                # the block is pretty-printed and starts on its own line with '{'
                starts = [m.start() for m in re.finditer(r"^\{", text[:idx], re.M)]
                obj = None
                if starts:
                    try:
                        obj, _ = decoder.raw_decode(text, starts[-1])
                    except ValueError:
                        obj = None
                if isinstance(obj, dict) and obj.get("suffix") and "EER_avg" in obj:
                    found.setdefault(obj["suffix"], obj)
                pos = idx + len('"EER_avg"')
    return found


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def verdict(actual, paper):
    if actual is None or paper is None:
        return "not measured"
    d = actual - paper
    return "ok" if abs(d) <= TOL else f"off by {d:+.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="append", required=True,
                    help="file or directory to scan; repeatable")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args(argv)

    data = scan([Path(s) for s in args.scan])
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- main
    lines = ["| row | suffix | EER (ours) | EER (paper) | UAR (ours) | UAR (paper) | "
             "WER (ours) | WER (paper) | verdict |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    rows_json = []
    for label, sfx, pe, pu, pw in MAIN_ROWS:
        m = data.get(sfx, {})
        eer, uar, wer = m.get("EER_avg"), m.get("UAR_avg"), m.get("WER_avg")
        if eer is None:
            v = "not measured"
        else:
            off = [x for x in (verdict(eer, pe), verdict(uar, pu), verdict(wer, pw)) if x != "ok"]
            v = "; ".join(off) if off else "ok"
        lines.append(f"| {label} | `{sfx}` | {fmt(eer)} | {fmt(pe,2)} | {fmt(uar)} | "
                     f"{fmt(pu,2)} | {fmt(wer)} | {fmt(pw,2)} | {v} |")
        rows_json.append({"row": label, "suffix": sfx, "eer": eer, "uar": uar, "wer": wer,
                          "paper_eer": pe, "paper_uar": pu, "paper_wer": pw})
    (args.out / "table1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "table1.json").write_text(
        json.dumps({"rows": rows_json, "sources": [str(s) for s in args.scan]}, indent=1),
        encoding="utf-8")

    # --------------------------------------------------------------- extras
    ex = ["| system | suffix | EER | UAR | WER |", "|---|---|---:|---:|---:|"]
    for label, sfx in EXTRA_ROWS:
        m = data.get(sfx)
        if not m:
            continue
        ex.append(f"| {label} | `{sfx}` | {fmt(m.get('EER_avg'))} | "
                  f"{fmt(m.get('UAR_avg'))} | {fmt(m.get('WER_avg'))} |")
    (args.out / "extras.md").write_text("\n".join(ex) + "\n", encoding="utf-8")

    # -------------------------------------------------------------- summary
    sm = ["# Rebuilt results", "",
          f"Sources: {', '.join(str(s) for s in args.scan)}", "",
          "## Table 1", ""] + lines + ["", "## Internal checks", ""]
    base, tts = data.get("_knn1"), data.get("_tts")
    if base and tts:
        sm.append("| mixture | ours EER | linear EER | ours UAR | linear UAR | ours WER | linear WER |")
        sm.append("|---|---:|---:|---:|---:|---:|---:|")
        for p in (0.2, 0.325, 0.4):
            sfx = "_mix0p" + str(p).replace("0.", "0").replace(".", "p")
            sfx = "_mix0p2" if p == 0.2 else ("_mix0p325" if p == 0.325 else "_mix0p4")
            m = data.get(sfx)
            if not m:
                continue
            lin = tuple(p * tts[k] + (1 - p) * base[k] for k in ("EER_avg", "UAR_avg", "WER_avg"))
            sm.append(f"| p={p} | {fmt(m['EER_avg'])} | {fmt(lin[0])} | {fmt(m['UAR_avg'])} | "
                      f"{fmt(lin[1])} | {fmt(m['WER_avg'])} | {fmt(lin[2])} |")
        sm.append("")
        sm.append("A super-linear EER (ours above the linear interpolation) is the paper's "
                  "headline claim for the admixture; UAR should stay close to linear.")
    measured = sum(1 for _, s, *_ in MAIN_ROWS if s in data)
    sm.append("")
    sm.append(f"Measured rows: **{measured}/{len(MAIN_ROWS)}**; "
              f"suffixes not found: {', '.join(s for _, s, *_ in MAIN_ROWS if s not in data) or 'none'}")
    (args.out / "summary.md").write_text("\n".join(sm) + "\n", encoding="utf-8")

    print(f"scanned {len(data)} suffixes: {', '.join(sorted(data))}")
    print(f"wrote {args.out}/table1.md, table1.json, summary.md, extras.md")
    print(f"measured rows: {measured}/{len(MAIN_ROWS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

