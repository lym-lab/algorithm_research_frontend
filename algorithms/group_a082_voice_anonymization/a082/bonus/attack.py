"""Bonus: a separating attacker against the admixture systems, and the analysis by trial type.

    # 1. attacker ASVs trained on parts of the admixture's training set (built by sysclf split)
    python -m a082.bonus.attack train --vpc-root R --mix-suffix _mix0p4 --subset pred_knn
    python -m a082.bonus.attack train --vpc-root R --mix-suffix _mix0p4 --subset true_knn      # oracle split
    # 2. EER overall and by the system of the trial utterance, for the VPC attacker and the separating one
    python -m a082.bonus.attack analyse --vpc-root R --mix-suffix _mix0p4 --labels work/bonus/labels/mix0p4 \
           --models single=exp/asv_anon_mix0p4 knn=exp/asv_anon_mix0p4__pred_knn tts=exp/asv_anon_mix0p4__pred_tts

`train` reuses the framework's eval_post unchanged, through its --overwrite mechanism: only the training
directory and the output directories change, so the attacker's recipe (ECAPA, 10 epochs, ...) is the
official one. Each run evaluates on the full admixture trials and writes the per-trial scores
(<model dir>/<enrolls>-<trials>/scores) that `analyse` reads.

`analyse` reports, per LibriSpeech trial list:
- EER of each model on all trials, on trials whose test utterance came from kNN-VC, and from the cascade;
- the "routed" attacker: each trial is scored by the model trained on the predicted system of its test
  utterance (knn or tts); scores are standardised per model before pooling (our heuristic: cosine scores
  of two separately trained models are not on the same scale).
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np

from .sysclf import read_labels

TRIAL_SETS = [("libri_dev_enrolls", "libri_dev_trials_f"), ("libri_dev_enrolls", "libri_dev_trials_m"),
              ("libri_test_enrolls", "libri_test_trials_f"), ("libri_test_enrolls", "libri_test_trials_m")]


def eer(target: np.ndarray, nontarget: np.ndarray) -> float:
    """Equal error rate in %, by sweeping the threshold over all scores."""
    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(len(target)), np.zeros(len(nontarget))])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    fnr = 1 - tp / max(len(target), 1)
    fpr = fp / max(len(nontarget), 1)
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2 * 100)


def overwrite_for(mix_suffix: str, subset: str) -> dict:
    tag = f"{mix_suffix}__{subset}"
    model_dir = f"exp/asv_anon{tag}"
    return {
        "anon_data_suffix": mix_suffix,
        "train_data_name": f"train-clean-360{mix_suffix}_{subset}",
        "results_summary_path": f"exp/results_summary/eval_anon{tag}/results_anon.txt",
        "privacy": {"asv": {"training": {"model_dir": model_dir},
                            "evaluation": {"model_dir": model_dir, "results_dir": model_dir}}},
    }


def cmd_train(args) -> None:
    o = json.dumps(overwrite_for(args.mix_suffix, args.subset))
    if args.no_iemocap:
        from ..eval.vpc import LIBRI_ONLY
        o = json.dumps({**json.loads(o), **LIBRI_ONLY})
    from ..eval import vpc
    tag = f"{args.mix_suffix}__{args.subset}"
    vpc.framework(args.vpc_root, [f"python run_evaluation.py --config configs/eval_post.yaml "
                                  f"--overwrite {shlex.quote(o)} --force_compute True"],
                  args.workspace, tag, fresh=(f"asv_anon{tag}",))


def read_scores(path: Path) -> dict[tuple[str, str], float]:
    out = {}
    for line in Path(path).read_text().splitlines():
        e, t, s = line.split()
        out[(e, t)] = float(s)
    return out


def read_trials(path: Path) -> dict[tuple[str, str], int]:
    out = {}
    for line in Path(path).read_text().splitlines():
        e, t, lab = line.split()
        out[(e, t)] = int(lab == "target")
    return out


def analyse(vpc_root: Path, mix_suffix: str, labels_dir: Path, models: dict[str, Path]) -> dict:
    data = vpc_root / "data"
    report: dict = {}
    for enrolls, trials_name in TRIAL_SETS:
        trials = read_trials(data / trials_name / "trials")
        labels = read_labels(labels_dir / f"{trials_name}.txt")
        pair = f"{enrolls}{mix_suffix}-{trials_name}{mix_suffix}"
        scores = {name: read_scores(vpc_root / d / pair / "scores") for name, d in models.items()}
        res = {}

        def split_eer(sc: dict, keep) -> float | None:
            t = np.array([s for k, s in sc.items() if trials[k] and keep(k)])
            n = np.array([s for k, s in sc.items() if not trials[k] and keep(k)])
            return eer(t, n) if len(t) and len(n) else None

        for name, sc in scores.items():
            res[name] = {"all": split_eer(sc, lambda k: True),
                         "test_knn": split_eer(sc, lambda k: labels[k[1]][0] == "knn"),
                         "test_tts": split_eer(sc, lambda k: labels[k[1]][0] == "tts")}
        if "knn" in scores and "tts" in scores:
            z = {}
            for name in ("knn", "tts"):
                v = np.array(list(scores[name].values()))
                z[name] = {k: (s - v.mean()) / (v.std() + 1e-9) for k, s in scores[name].items()}
            routed = {k: z[labels[k[1]][1]][k] for k in trials}
            res["routed"] = {"all": split_eer(routed, lambda k: True),
                             "test_knn": split_eer(routed, lambda k: labels[k[1]][0] == "knn"),
                             "test_tts": split_eer(routed, lambda k: labels[k[1]][0] == "tts")}
        report[trials_name] = res
    names = next(iter(report.values())).keys()
    report["average"] = {n: {c: _mean([report[t][n][c] for _, t in TRIAL_SETS])
                             for c in ("all", "test_knn", "test_tts")} for n in names}
    return report


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def cmd_analyse(args) -> None:
    models = dict(m.split("=", 1) for m in args.models)
    rep = analyse(args.vpc_root, args.mix_suffix, Path(args.labels), {k: Path(v) for k, v in models.items()})
    text = json.dumps(rep, indent=1)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--vpc-root", required=True, type=Path); t.add_argument("--mix-suffix", required=True)
    t.add_argument("--subset", required=True, help="pred_knn | pred_tts | true_knn | true_tts")
    t.add_argument("--no-iemocap", action="store_true")
    t.add_argument("--workspace", help="'auto' or a directory: run in a separate working copy (a082/eval/workspace.py)")
    a = sub.add_parser("analyse")
    a.add_argument("--vpc-root", required=True, type=Path); a.add_argument("--mix-suffix", required=True)
    a.add_argument("--labels", required=True); a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--json")
    args = ap.parse_args(argv)
    {"train": cmd_train, "analyse": cmd_analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
