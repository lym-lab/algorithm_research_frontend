"""Bonus, step "experiment": can an attacker tell which system produced an anonymized utterance?

The admixture systems get their high EER by mixing kNN-VC and cascade (TTS) outputs in the attacker's
training data (paper §2.4, "data poisoning"); the paper's own Limitations note that an informed adversary
can undo this. A semi-informed attacker knows the anonymization pipeline, so can produce labelled examples
of both systems and train a classifier. This module trains that classifier and uses it to split the
anonymized training set before the attacker trains the ASV (see attack.py).

    # train on utterances the attacker anonymized with both systems (here: train-clean-360 outputs)
    python -m a082.bonus.sysclf train --vpc-root R --knn-suffix _knn1 --tts-suffix _tts --out work/bonus/sysclf.pt
    # accuracy on the evaluation sets, per system
    python -m a082.bonus.sysclf eval  --vpc-root R --model work/bonus/sysclf.pt --sets eval
    # label every utterance of an admixture and write Kaldi subsets of its training set:
    #   train-clean-360<mix>_{pred,true}_{knn,tts}
    python -m a082.bonus.sysclf split --vpc-root R --model work/bonus/sysclf.pt --mix-suffix _mix0p4 --p 0.4

Features: mean and standard deviation over time of WavLM layer-6 frames of the first `crop` seconds
(4 s by default, to keep labelling the 104k training utterances cheap). Classifier: logistic regression.
Validation is speaker-disjoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import admix, audio, kaldi, runner, seeding
from ..models import load_wavlm, wavlm_features

LABELS = ("knn", "tts")


@torch.inference_mode()
def embed(wavlm, wav: torch.Tensor, crop_s: float = 4.0) -> torch.Tensor:
    x = wavlm_features(wavlm, wav[:, : int(crop_s * audio.SR)]).float()
    return torch.cat([x.mean(0), x.std(0)]).cpu()


def embed_dir(wavlm, d: Path, utts: list[str], crop_s: float, root: Path) -> torch.Tensor:
    scp = kaldi.read_kaldi(d / "wav.scp")
    out = []
    for u in tqdm(utts, desc=d.name):
        wav, sr = audio.load_scp_entry(scp[u], root=root)
        out.append(embed(wavlm, audio.resample(wav, sr), crop_s))
    return torch.stack(out)


def fit_logreg(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3, steps: int = 300):
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5)
    z = (x - mean) / std
    w = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z @ w + b, y.float()) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"mean": mean, "std": std, "w": w.detach(), "b": b.detach()}


def predict(model: dict, x: torch.Tensor) -> torch.Tensor:
    """P(tts) per row."""
    return torch.sigmoid(((x - model["mean"]) / model["std"]) @ model["w"] + model["b"])


def _sample(utts: list[str], n: int, key: str) -> list[str]:
    return sorted(utts, key=lambda u: seeding.seed_int(u, key))[:n]


def cmd_train(args) -> None:
    data = args.vpc_root / "data"
    wavlm = load_wavlm(args.device)
    src = data / kaldi.TRAIN_SET
    utt2spk = kaldi.read_kaldi(src / "utt2spk")
    spks = sorted(set(utt2spk.values()))
    valid_spk = {s for s in spks if seeding.uniform01(s, "sysclf-valid") < 0.2}
    # different utterances for the two classes, so the classifier cannot match content
    pool = _sample(list(utt2spk), 2 * args.n, "sysclf-pick")
    knn_u, tts_u = pool[0::2], pool[1::2]
    xk = embed_dir(wavlm, data / f"{kaldi.TRAIN_SET}{args.knn_suffix}", knn_u, args.crop, args.vpc_root)
    xt = embed_dir(wavlm, data / f"{kaldi.TRAIN_SET}{args.tts_suffix}", tts_u, args.crop, args.vpc_root)
    x = torch.cat([xk, xt])
    y = torch.cat([torch.zeros(len(knn_u)), torch.ones(len(tts_u))])
    is_valid = torch.tensor([utt2spk[u] in valid_spk for u in knn_u + tts_u])
    model = fit_logreg(x[~is_valid], y[~is_valid])
    acc = ((predict(model, x[is_valid]) > 0.5).float() == y[is_valid]).float().mean().item()
    model["valid_acc"], model["crop"] = acc, args.crop
    model["suffixes"] = {"knn": args.knn_suffix, "tts": args.tts_suffix}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.out)
    print(f"speaker-disjoint validation accuracy {acc:.4f} on {int(is_valid.sum())} utterances -> {args.out}")


def cmd_eval(args) -> None:
    data = args.vpc_root / "data"
    model = torch.load(args.model)
    wavlm = load_wavlm(args.device)
    report = {}
    for name in runner.available_sets(args.vpc_root, runner.resolve_sets(args.sets)):
        utts = list(kaldi.read_kaldi(data / name / "wav.scp"))
        res = {}
        for label, truth in (("knn", 0), ("tts", 1)):
            x = embed_dir(wavlm, data / f"{name}{model['suffixes'][label]}", utts, model["crop"], args.vpc_root)
            res[label] = ((predict(model, x) > 0.5).float() == truth).float().mean().item()
        report[name] = res
        print(f"{name}: accuracy on kNN-VC {res['knn']:.4f}, on cascade {res['tts']:.4f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1))


def cmd_split(args) -> None:
    data = args.vpc_root / "data"
    model = torch.load(args.model)
    wavlm = load_wavlm(args.device)
    # --name other than "pred": a second split of the same admixture (e.g. by a classifier the attacker retrained
    # for improvement A) next to the first one, in labels/<mix>_<name> and train-clean-360<mix>_<name>_{knn,tts}
    tag = "" if args.name == "pred" else f"_{args.name}"
    out_labels = Path(args.labels_dir) / (args.mix_suffix.lstrip("_") + tag)
    out_labels.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in runner.available_sets(args.vpc_root, runner.resolve_sets(args.sets), args.mix_suffix):
        d = data / f"{name}{args.mix_suffix}"
        utts = sorted(kaldi.read_kaldi(d / "wav.scp"))
        prob = predict(model, embed_dir(wavlm, d, utts, model["crop"], args.vpc_root)).tolist()
        u2s = kaldi.read_kaldi(d / "utt2spk")
        truth = {u: "tts" if admix.uses_tts(u2s[u] if args.level == "spk" else u, args.p) else "knn" for u in utts}
        pred = {u: "tts" if q > 0.5 else "knn" for u, q in zip(utts, prob)}
        with open(out_labels / f"{name}.txt", "w", encoding="utf-8", newline="\n") as f:
            for u, q in zip(utts, prob):
                f.write(f"{u} {truth[u]} {pred[u]} {q:.4f}\n")
        acc = sum(pred[u] == truth[u] for u in utts) / len(utts)
        summary[name] = {"accuracy": acc, "n": len(utts)}
        print(f"{name}{args.mix_suffix}: accuracy {acc:.4f}")
        if name == kaldi.TRAIN_SET:
            for kind, lab in ((args.name, pred),) + ((("true", truth),) if not args.no_true else ()):
                for cls in LABELS:
                    keep = {u for u in utts if lab[u] == cls}
                    n = kaldi.subset_dir(d, data / f"{name}{args.mix_suffix}_{kind}_{cls}", keep)
                    summary[f"{kind}_{cls}"] = n
    (out_labels / "summary.json").write_text(json.dumps(summary, indent=1))


def read_labels(path: Path) -> dict[str, tuple[str, str, float]]:
    """utt -> (true system, predicted system, P(tts)) from a split labels file."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        u, t, p, q = line.split()
        out[u] = (t, p, float(q))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "split"):
        a = sub.add_parser(name)
        a.add_argument("--vpc-root", required=True, type=Path)
        a.add_argument("--device", default="cuda")
    t = sub.choices["train"]
    t.add_argument("--knn-suffix", default="_knn1"); t.add_argument("--tts-suffix", default="_tts")
    t.add_argument("--n", type=int, default=2000, help="utterances per class")
    t.add_argument("--crop", type=float, default=4.0); t.add_argument("--out", default="work/bonus/sysclf.pt")
    e = sub.choices["eval"]
    e.add_argument("--model", default="work/bonus/sysclf.pt"); e.add_argument("--sets", default="eval"); e.add_argument("--json")
    s = sub.choices["split"]
    s.add_argument("--model", default="work/bonus/sysclf.pt"); s.add_argument("--mix-suffix", required=True)
    s.add_argument("--p", type=float, required=True); s.add_argument("--sets", default="all")
    s.add_argument("--labels-dir", default="work/bonus/labels")
    s.add_argument("--level", choices=["utt", "spk"], default="utt", help="how the admixture was drawn")
    s.add_argument("--name", default="pred", help="name of the predicted subsets (default pred)")
    s.add_argument("--no-true", action="store_true", help="do not (re)write the true-label subsets")
    args = ap.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval, "split": cmd_split}[args.cmd](args)


if __name__ == "__main__":
    main()
