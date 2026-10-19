"""Table 1 of the paper (p. 65) and our acceptance tolerances.

Averages: EER over the four LibriSpeech subsets (dev-f, dev-m, test-f, test-m); UAR and WER over dev and
test. Tolerances are absolute percentage points. `eer_min` replaces the EER tolerance where the
paper's value is close to the 50 % ceiling. Rows 6-8 are judged qualitatively (see `qualitative`).
"""

TABLE1 = {
    # id: (suffix, avg EER, avg UAR, avg WER)
    "0": ("_orig", 5.16, 70.07, 1.83),
    "1": ("_knn1", 7.95, 56.70, 3.16),
    "2": ("_knn2", 8.08, 56.29, 3.33),
    "3": ("_knn3", 20.58, 43.26, 11.00),
    "4": ("_tts", 48.25, 30.35, 3.75),
    "mix0.2": ("_mix0p2", 21.91, 51.71, 3.31),
    "mix0.325": ("_mix0p325", 33.91, 49.04, 3.43),
    "mix0.4": ("_mix0p4", 40.81, 47.09, 3.33),
    "5": ("_wc5", 8.54, 54.75, 3.01),
    "6": ("_wc6", 11.60, 50.62, 4.23),
    "7": ("_wc7", 13.73, 43.56, 10.16),
    "8": ("_wc8", 13.88, 43.50, 10.83),
}

NAMES_ZH = {
    "0": "原始语音", "1": "kNN-VC", "2": "+ 长度扰动", "3": "+ 长度扰动 + 特征加噪", "4": "Whisper → Piper 级联",
    "mix0.2": "混合 p = 0.2", "mix0.325": "混合 p = 0.325", "mix0.4": "混合 p = 0.4",
    "5": "WavLM 转换（基础）", "6": "+ 说话人对抗", "7": "+ 离散目标", "8": "+ 两者",
}

# Subset values available for row 0 (from the evaluation plan, tables 9-11, identical to Table 1).
ROW0_SUBSETS = {
    "EER": {"dev_f": 10.511, "dev_m": 0.931, "test_f": 8.761, "test_m": 0.418},
    "UAR": {"dev": 69.0796, "test": 71.0618},
    "WER": {"dev": 1.807, "test": 1.844},
}

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


def qualitative(row: str, m: dict, ours: dict) -> str | None:
    """Checks for rows 6-8: rows 6 and 7 against our row 5, row 8 against our row 7 (`ours`: row -> metrics).

    None if not applicable or not all averages are measured yet.
    """
    ref = {"6": "5", "7": "5", "8": "7"}.get(row)
    base = ours.get(ref) if ref else None
    if base is None or None in (m["EER_avg"], m["WER_avg"], base["EER_avg"], base["WER_avg"]):
        return None
    eer, wer = m["EER_avg"], m["WER_avg"]
    checks = {"6": [eer > base["EER_avg"], wer > base["WER_avg"]],
              "7": [eer > base["EER_avg"], wer >= 6.0],
              "8": [eer >= base["EER_avg"], abs(wer - base["WER_avg"]) <= WER_CLOSE]}[row]
    return "qualitative ok" if all(checks) else "qualitative NOT met"


WER_CLOSE = 1.0        # row 8 vs row 7: "WER close to row 7" (paper: 10.83 vs 10.16)
SAME_AS_1 = 0.5        # row 2 vs our row 1: UAR and WER within this (paper: "similar scores")
MIX_ABOVE_LINEAR = 3.0  # admixture EER must exceed the linear interpolation of rows 1 and 4 by this much

# Fig. 4 sweep: admixture probability -> suffix. The three Table 1 points are included.
SWEEP = {0.1: "_mix0p1", 0.15: "_mix0p15", 0.2: "_mix0p2", 0.25: "_mix0p25", 0.3: "_mix0p3",
         0.325: "_mix0p325", 0.35: "_mix0p35", 0.4: "_mix0p4", 0.5: "_mix0p5"}
FIG4_P05_UAR = 44.77    # the only Fig. 4 value given in the text (p = 0.5)


def internal_checks(ours: dict, sweep: dict) -> list[tuple[str, str]]:
    """Checks among our own rows (not against the paper's numbers): [(name, verdict)].

    `ours`: Table 1 row -> metrics from `collect`; `sweep`: p -> metrics of the Fig. 4 points.
    """
    out = []
    got = lambda r, k: (ours.get(r) or {}).get(f"{k}_avg")
    if None not in (got("1", "UAR"), got("2", "UAR"), got("1", "WER"), got("2", "WER")):
        d = max(abs(got("2", k) - got("1", k)) for k in ("UAR", "WER"))
        out.append(("row 2 vs row 1: UAR, WER within %.1f" % SAME_AS_1,
                    "ok" if d <= SAME_AS_1 else f"NOT met (largest difference {d:.2f})"))
    for row in ("mix0.2", "mix0.325", "mix0.4"):
        p = float(row[3:])
        vals = {k: (got(row, k), got("1", k), got("4", k)) for k in ("EER", "UAR", "WER")}
        if any(None in v for v in vals.values()):
            continue
        lin = {k: (1 - p) * a + p * b for k, (_, a, b) in vals.items()}
        tol = TOLERANCE[row]
        res = [vals["EER"][0] >= lin["EER"] + MIX_ABOVE_LINEAR,
               abs(vals["UAR"][0] - lin["UAR"]) <= tol["UAR"], abs(vals["WER"][0] - lin["WER"]) <= tol["WER"]]
        detail = ", ".join(f"{k} {vals[k][0]:.2f} vs linear {lin[k]:.2f}" for k in ("EER", "UAR", "WER"))
        out.append((f"{row} vs linear mix of rows 1, 4", ("ok" if all(res) else "NOT met") + f" ({detail})"))
    pts = sorted((p, m["EER"].get("dev_f"), m.get("UAR_avg")) for p, m in sweep.items()
                 if m["EER"].get("dev_f") is not None and m.get("UAR_avg") is not None)
    if len(pts) >= 3:
        mono = all(b[1] >= a[1] for a, b in zip(pts, pts[1:])) and all(b[2] <= a[2] for a, b in zip(pts, pts[1:]))
        out.append(("Fig. 4: dev-f EER rises and UAR falls with p", "ok" if mono else "NOT met"))
    if 0.5 in sweep and sweep[0.5].get("UAR_avg") is not None and sweep[0.5].get("UAR_corpus") != "CREMA":
        d = sweep[0.5]["UAR_avg"] - FIG4_P05_UAR
        out.append(("Fig. 4: UAR at p = 0.5 within 2 of 44.77", "ok" if abs(d) <= 2.0 else f"off by {d:+.2f}"))
    return out
