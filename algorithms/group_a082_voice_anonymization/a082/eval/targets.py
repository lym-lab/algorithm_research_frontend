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


def qualitative(row: str, m: dict, base: dict | None) -> str | None:
    """Checks for rows 6-8 relative to our row 5 (`base`); None if not applicable."""
    if row not in ("6", "7", "8") or base is None:
        return None
    eer, wer = m["EER_avg"], m["WER_avg"]
    checks = {"6": [eer > base["EER_avg"], wer > base["WER_avg"]],
              "7": [eer > base["EER_avg"], wer >= 6.0],
              "8": [eer >= 7.0]}[row]
    return "qualitative ok" if all(checks) else "qualitative NOT met"
