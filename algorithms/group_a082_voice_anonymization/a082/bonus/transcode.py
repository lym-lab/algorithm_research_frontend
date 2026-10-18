"""Bonus, real-world condition: anonymized speech is usually shared through messaging apps or phone calls,
which re-encode it. Does the privacy/utility trade-off survive? (The paper evaluates clean 16 kHz files only.)

    python -m a082.bonus.transcode make     --vpc-root R --suffix _knn1 --codec opus12k [--sets eval] [--workers 8]
    python -m a082.bonus.transcode evaluate --vpc-root R --suffix _knn1 --codec opus12k [--no-iemocap]

Writes data/<set><suffix>_<codec>/ with each anonymized file encoded and decoded back to 16 kHz PCM, so the
framework can evaluate it like any other suffix. Codecs (ffmpeg must have the encoder):
  opus12k   Opus 12 kbit/s, VoIP mode   (WhatsApp / WeChat voice messages use Opus at low rates)
  amrnb     AMR-NB 12.2 kbit/s, 8 kHz   (GSM / 3G voice calls; needs libopencore_amrnb)
  mp3_32k   MP3 32 kbit/s mono
`evaluate` scores the transcoded copy with the attacker ASV already trained on the clean anonymized
training set (exp/asv_anon<suffix>; the realistic case: the attacker trains on what the anonymizer
publishes, then meets re-encoded recordings), plus eval_pre's ASR/SER on the transcoded audio.
Applying the same codec to the original data (suffix _orig from `a082.eval.vpc identity`) gives the
reference: how much of any change is the codec's own effect on ASV/ASR.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .. import kaldi, runner

CODECS = {
    "opus12k": (["-c:a", "libopus", "-b:a", "12k", "-application", "voip", "-ar", "16000"], "ogg"),
    "amrnb": (["-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-ar", "8000"], "amr"),
    "mp3_32k": (["-c:a", "libmp3lame", "-b:a", "32k", "-ar", "16000"], "mp3"),
}


def roundtrip(src: Path, dst: Path, codec: str, ffmpeg: str = "ffmpeg") -> None:
    args, ext = CODECS[codec]
    with tempfile.TemporaryDirectory() as tmp:
        enc = Path(tmp) / f"x.{ext}"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-ac", "1", *args, str(enc)], check=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(enc), "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(dst)], check=True)


def check_encoder(codec: str, ffmpeg: str = "ffmpeg") -> None:
    enc = CODECS[codec][0][1]
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if enc not in out:
        raise SystemExit(f"this ffmpeg has no {enc} encoder; codec {codec} unavailable")


def run(vpc_root: Path, suffix: str, codec: str, sets: list[str], workers: int = 8, ffmpeg: str = "ffmpeg") -> None:
    check_encoder(codec, ffmpeg)
    data = vpc_root / "data"
    for name in sets:
        src, out = data / f"{name}{suffix}", data / f"{name}{suffix}_{codec}"
        if not (src / "wav.scp").exists():
            print(f"skipping {name}: {src.name} not found")
            continue
        kaldi.prepare_output_dir(src, out)
        scp = kaldi.read_kaldi(src / "wav.scp")
        todo = [u for u in scp if not kaldi.wav_path(out, u).exists()]
        with ThreadPoolExecutor(workers) as pool:
            list(tqdm(pool.map(lambda u: roundtrip(Path(scp[u]), kaldi.wav_path(out, u), codec, ffmpeg), todo),
                      total=len(todo), desc=out.name))
        kaldi.finalize_output_dir(data / name, out)


def evaluate(vpc_root: Path, suffix: str, codec: str, no_iemocap: bool = False) -> None:
    from ..eval import vpc
    sc = f"{suffix}_{codec}"
    data = vpc_root / "data"
    # the framework insists that data/train-clean-360<suffix> exists; point it at the clean anonymized
    # training set (training itself is skipped: retrain = false and the model exists)
    if suffix == "_orig":  # reference: original-original through the codec; there is no attacker model
        vpc.run(vpc_root, sc, stages=("pre",), no_iemocap=no_iemocap)
        return
    trained = f"exp/asv_anon{suffix}"
    if not (vpc_root / trained).exists():
        raise SystemExit(f"{trained} missing: run eval_post for {suffix} first")
    link = data / f"{kaldi.TRAIN_SET}{sc}"
    if not (link.is_symlink() or link.exists()):
        link.symlink_to((data / f"{kaldi.TRAIN_SET}{suffix}").resolve(), target_is_directory=True)
    post = {"anon_data_suffix": sc, "results_summary_path": f"exp/results_summary/eval_anon{sc}/results_anon.txt",
            "privacy": {"asv": {"training": {"model_dir": trained, "retrain": False},
                                "evaluation": {"model_dir": trained, "results_dir": f"exp/asv_anon{sc}"}}}}
    if no_iemocap:
        post.update(vpc.LIBRI_ONLY)
    vpc.run(vpc_root, sc, stages=("pre",), no_iemocap=no_iemocap)
    cmd = (f"source env.sh && python run_evaluation.py --config configs/eval_post.yaml "
           f"--overwrite {shlex.quote(json.dumps(post))}")  # no --force_compute: it would delete the trained model
    print(cmd, flush=True)
    subprocess.run(["bash", "-c", cmd], cwd=vpc_root, check=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "evaluate"):
        a = sub.add_parser(name)
        a.add_argument("--vpc-root", required=True, type=Path)
        a.add_argument("--suffix", required=True)
        a.add_argument("--codec", choices=sorted(CODECS), required=True)
    m = sub.choices["make"]
    m.add_argument("--sets", default="eval")
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    sub.choices["evaluate"].add_argument("--no-iemocap", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        run(args.vpc_root, args.suffix, args.codec, runner.resolve_sets(args.sets), args.workers, args.ffmpeg)
    else:
        evaluate(args.vpc_root, args.suffix, args.codec, args.no_iemocap)


if __name__ == "__main__":
    main()
