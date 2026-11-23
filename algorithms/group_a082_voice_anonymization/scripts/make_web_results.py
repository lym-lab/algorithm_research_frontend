"""Turn the rebuilt table into the JSON that webapp/server.py serves.

    python scripts/make_web_results.py --table ../results/table1.json \
        --attack <work/bonus/attack_mix0p4.json> --sysclf <work/bonus/sysclf_eval.json> \
        --out ../webapp

Writes
    results.json   list of rows for GET /api/results  (row, name, paper, ours, verdict)
    bonus.json     {"mix_separated": {...}, "sysclf_accuracy": ...}

`ours` is null when the evaluation does not exist yet, which is what the page
renders as "待测". Tolerances are copied from a082/eval/targets.py so that this
script stays standard-library only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SUFFIX_TO_ROW = {
    "_orig": "0", "_knn1": "1", "_knn2": "2", "_knn3d18": "3", "_knn3": "3",
    "_tts": "4", "_mix0p2": "mix0.2", "_mix0p325": "mix0.325", "_mix0p4": "mix0.4",
    "_wc5": "5", "_wc6": "6", "_wc7": "7", "_wc8": "8",
}
NAMES = {
    "0": "原始语音", "1": "kNN-VC", "2": "+ 长度扰动", "3": "+ 长度扰动 + 特征加噪",
    "4": "Whisper → Piper 级联", "mix0.2": "混合 p = 0.2", "mix0.325": "混合 p = 0.325",
    "mix0.4": "混合 p = 0.4", "5": "WavLM 转换（基础）", "6": "+ 说话人对抗",
    "7": "+ 离散目标", "8": "+ 两者",
}
# a082/eval/targets.py, rows without an entry fall back to the defaults below
TOLERANCE = {
    "0": {"EER": 0.05, "UAR": 0.05, "WER": 0.05},
    "1": {"EER": 1.5, "UAR": 2.0, "WER": 0.3},
    "2": {"EER": 1.5, "UAR": 2.0, "WER": 0.3},
    "3": {"EER": 4.0, "UAR": 3.0, "WER": 2.0},
    "4": {"eer_min": 45.0, "UAR": 2.0, "WER": 0.4},
    "mix0.2": {"EER": 3.0, "UAR": 2.0, "WER": 0.3},
    "mix0.325": {"EER": 3.0, "UAR": 2.0, "WER": 0.3},
    "mix0.4": {"EER": 3.0, "UAR": 2.0, "WER": 0.3},
    "5": {"EER": 3.0, "UAR": 3.0, "WER": 1.0},
}
DEFAULT_TOL = {"EER": 3.0, "UAR": 3.0, "WER": 2.0}


def load_json_loose(path: Path):
    """Parse a file that may have log lines around the JSON object."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"^\{", text, re.M))):
        try:
            obj, _ = dec.raw_decode(text, m.start())
            return obj
        except ValueError:
            continue
    raise ValueError(f"no JSON object found in {path}")


def verdict(row: str, ours: dict, paper: dict) -> dict | None:
    """{'EER': 'ok' | 'off by ...', ...}; None when nothing is measured."""
    if not ours or ours.get("EER") is None:
        return None
    tol = TOLERANCE.get(row, DEFAULT_TOL)
    out = {}
    for k in ("EER", "UAR", "WER"):
        a, p = ours.get(k), paper.get(k)
        if a is None or p is None:
            continue
        if k == "EER" and "eer_min" in tol:      # paper value near the 50 % ceiling
            out[k] = "ok" if a >= tol["eer_min"] else f"off by {a - p:+.2f}"
        else:
            d = a - p
            out[k] = "ok" if abs(d) <= tol.get(k, DEFAULT_TOL[k]) else f"off by {d:+.2f}"
    return out or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True, help="results/table1.json")
    ap.add_argument("--attack", type=Path, help="bonus/attack_mix0p4.json (routing analysis)")
    ap.add_argument("--sysclf", type=Path, help="bonus/sysclf_eval.json")
    ap.add_argument("--out", type=Path, required=True, help="webapp directory")
    args = ap.parse_args(argv)

    table = json.loads(args.table.read_text(encoding="utf-8"))
    rows, seen = [], set()
    for r in table["rows"]:
        row = SUFFIX_TO_ROW.get(r["suffix"])
        if row is None or row in seen:
            continue
        seen.add(row)
        paper = {"EER": r["paper_eer"], "UAR": r["paper_uar"], "WER": r["paper_wer"]}
        ours = None
        if r["eer"] is not None:
            ours = {"EER": round(r["eer"], 3), "UAR": round(r["uar"], 3), "WER": round(r["wer"], 3)}
        rows.append({"row": row, "suffix": r["suffix"], "name": NAMES[row],
                     "paper": paper, "ours": ours, "verdict": verdict(row, ours or {}, paper)})
    order = {k: i for i, k in enumerate(["0", "1", "2", "3", "4", "mix0.2", "mix0.325", "mix0.4",
                                         "5", "6", "7", "8"])}
    rows.sort(key=lambda x: order[x["row"]])

    bonus = {}
    if args.attack and args.attack.exists():
        atk = load_json_loose(args.attack)
        avg = atk.get("average", {})
        routed = (avg.get("routed") or {}).get("all")
        single = (avg.get("single") or {}).get("all")
        if routed is not None:
            bonus["mix_separated"] = {"0.4": round(routed, 3)}
        if single is not None:
            bonus["mix_single"] = {"0.4": round(single, 3)}
    if args.sysclf and args.sysclf.exists():
        clf = load_json_loose(args.sysclf)
        vals = [v for d in clf.values() if isinstance(d, dict) for k, v in d.items() if k in ("knn", "tts")]
        if vals:
            bonus["sysclf_accuracy"] = round(sum(vals) / len(vals), 4)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "bonus.json").write_text(json.dumps(bonus, ensure_ascii=False, indent=1), encoding="utf-8")

    measured = sum(1 for r in rows if r["ours"])
    print(f"rows {len(rows)} (measured {measured}); bonus: {bonus}")
    print(f"wrote {args.out/'results.json'} and {args.out/'bonus.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

