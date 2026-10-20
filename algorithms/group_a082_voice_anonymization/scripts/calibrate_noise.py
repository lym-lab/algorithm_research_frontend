"""Where the feature noise of system 3 goes and how large it is.

    python scripts/calibrate_noise.py --vpc-root R --pool <cache> [--n 50] [--combos matching:18,matching:20] [--device cuda]

1. Prints the per-dimension standard deviation of WavLM layer-6 features on the calibration utterances,
   to judge how large "infinity norm < 32" is relative to the feature scale.
2. Anonymizes the same n dev utterances (from libri_dev_trials_f + _m, i.e. libri-dev-asr) with system 3
   for noise position in {matching, query} x delta in {16, 32}.
3. Runs the framework's ASR evaluation on each (unchanged scripts, --overwrite restricts it to the
   calibration set). Pick the combination whose WER is closest to the paper's dev WER for system 3
   (11.993); with IEMOCAP available, also compare UAR (dev 44.126).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import torch
import yaml

from a082 import audio, kaldi, runner, seeding
from a082.eval import vpc
from a082.knnvc.anonymize import KnnSystem
from a082.knnvc.pool import Pool
from a082.models import load_hifigan, load_wavlm, wavlm_features

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
COMBOS = [("matching", 16.0), ("matching", 32.0), ("query", 16.0), ("query", 32.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--combos", help="where:delta,... (default: matching/query x 16/32)")
    args = ap.parse_args()
    combos = COMBOS if not args.combos else [(c.split(":")[0], float(c.split(":")[1])) for c in args.combos.split(",")]
    data = args.vpc_root / "data"

    # calibration set: n utterances of libri-dev-asr, chosen by hash
    utts = []
    for sub in ("libri_dev_trials_f", "libri_dev_trials_m"):
        utts += list(kaldi.read_kaldi(data / sub / "wav.scp"))
    keep = set(sorted(utts, key=lambda u: seeding.seed_int(u, "g9"))[: args.n])
    cname = "calib_g9" if args.n == 50 else f"calib_g9_n{args.n}"   # the framework caches results per set name
    calib = data / cname
    merged = data / "calib_g9_all"
    merged.mkdir(exist_ok=True)
    for name in ("wav.scp", "utt2spk", "text", "utt2dur", "spk2gender"):
        d = {}
        for sub in ("libri_dev_trials_f", "libri_dev_trials_m"):
            d.update(kaldi.read_kaldi(data / sub / name))
        kaldi.write_kaldi(d, merged / name)
    kaldi.subset_dir(merged, calib, keep)

    wavlm, hifigan = load_wavlm(args.device), load_hifigan(args.device)
    scp = kaldi.read_kaldi(calib / "wav.scp")
    feats = torch.cat([wavlm_features(wavlm, audio.resample(*audio.load_scp_entry(scp[u], args.vpc_root))).float()
                       for u in sorted(keep)])
    std = feats.std(0)
    print(f"layer-6 feature std per dim: median {std.median():.3f}, max {std.max():.3f}; "
          f"|x| max {feats.abs().max():.2f}; L2 norm per frame median {feats.norm(dim=1).median():.2f}")

    pool = Pool(args.pool, device=args.device)
    base = yaml.safe_load((CONFIGS / "knn3.yaml").read_text())
    results = {}
    for where, delta in combos:
        suffix = f"_g9{where[0]}{int(delta)}n{args.n}"
        system = KnnSystem({**base, "suffix": suffix, "noise": {"where": where, "delta": delta}},
                           pool, wavlm, hifigan, args.device)
        runner.anonymize_sets(args.vpc_root, [cname], suffix, lambda u, w: system.convert(u, w))
        o = {"anon_data_suffix": suffix,
             "datasets": [{"name": cname, "data": cname}],
             "eval_steps": {"privacy": [], "utility": ["asr"]},
             "utility": {"asr": {"dataset_name": [cname]}}}

        cmd = (f"source env.sh && python run_evaluation.py --config configs/eval_pre.yaml "
               f"--overwrite {shlex.quote(json.dumps(o))} --force_compute True")
        subprocess.run(["bash", "-c", cmd], cwd=args.vpc_root, check=True)
        rows = vpc._rows(args.vpc_root / "exp" / "asr" / f"results{suffix}.csv")
        wer = [float(r["WER"]) for r in rows if r["asr"] == "anon"][0]
        results[f"{where}/{delta:g}"] = wer
        print(f"noise on {where}, delta {delta:g}: WER {wer:.2f} (paper dev WER for system 3: 11.993)")
    best = min(results, key=lambda k: abs(results[k] - 11.993))
    print(json.dumps({"wer": results, "closest": best}, indent=1))


if __name__ == "__main__":
    main()
