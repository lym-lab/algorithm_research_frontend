"""Small-sample WER probe for WavLM-converter checkpoints, on CPU next to running GPU jobs.

Converts a fixed subset of libri_dev_asr with each checkpoint (the source WavLM features are computed
once) and transcribes the outputs with the framework's ASR model (exp/asr, EncoderASR), the model
eval_post uses for WER. Used to see how the utility of a variant changes with its training steps
(wc6, 2026-10-22). The original audio is scored too, as the subset's own floor.

    cd <code> && nice python scripts/wc_probe.py --vpc-root R --out W/probe --ckpt wc5=W/wc/wc5/last.pt wc6_10k=...
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def pick(ids: list[str], n: int) -> list[str]:
    """Every k-th utterance of the sorted list: spread over all speakers and chapters."""
    ids = sorted(ids)
    k = max(len(ids) // n, 1)
    return ids[::k][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vpc-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ckpt", nargs="+", required=True, help="tag=path")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--set", default="libri_dev_asr")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from a082 import audio, kaldi
    from a082.models import load_hifigan, load_wavlm, vocode, wavlm_features
    from a082.wavlm_conv.model import WavLMConverter

    R = args.vpc_root.resolve()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    scp = kaldi.read_kaldi(R / "data" / args.set / "wav.scp")
    text = kaldi.read_kaldi(R / "data" / args.set / "text")
    utts = pick(list(scp), args.n)

    feats_file = out / "feats.pt"
    if feats_file.exists():
        feats = torch.load(feats_file)
    else:
        wavlm = load_wavlm("cpu"); t0 = time.time()
        feats = {u: wavlm_features(wavlm, audio.load_audio(R / scp[u])).half() for u in utts}
        torch.save(feats, feats_file); del wavlm
        print(f"features of {len(utts)} utterances in {time.time() - t0:.0f} s", flush=True)

    sets = {"orig": {u: str(R / scp[u]) for u in utts}}
    hifigan = None
    for spec in args.ckpt:
        tag, path = spec.split("=", 1)
        wav_dir = out / tag
        if not all((wav_dir / f"{u}.wav").exists() for u in utts):
            hifigan = hifigan or load_hifigan("cpu", prematched=True)
            ck = torch.load(path, map_location="cpu")
            model = WavLMConverter(**ck["cfg"].get("model", {})).eval()
            model.load_state_dict(ck["model"]); step = ck.get("step"); del ck
            t0 = time.time()
            with torch.inference_mode():
                for u in utts:
                    if (wav_dir / f"{u}.wav").exists():
                        continue
                    _, y = model(feats[u].float()[None])
                    audio.save_wav16(wav_dir / f"{u}.wav", audio.loudness_normalize(vocode(hifigan, y[0])))
            print(f"{tag} (step {step}): converted in {time.time() - t0:.0f} s", flush=True)
        sets[tag] = {u: str(wav_dir / f"{u}.wav") for u in utts}

    # the framework's ASR, imported from its own tree (it expects R as the working directory)
    os.chdir(R); sys.path.insert(0, str(R))
    from evaluation.utility.asr.speechbrain_asr.inference import ASRDataset, InferenceSpeechBrainASR
    from torch.utils.data import DataLoader
    asr = InferenceSpeechBrainASR(model_path=R / "exp" / "asr", asr_hparams="hyperparams.yaml",
                                  model_type="EncoderASR", device="cpu")
    refs = {u: text[u] for u in utts}
    n_words = sum(len(t.split()) for t in refs.values())
    rows = []
    for tag, wavs in sets.items():
        res = out / "asr" / tag
        if (res / "text").exists():
            hyp = kaldi.read_kaldi(res / "text")
        else:
            res.mkdir(parents=True, exist_ok=True)
            kaldi.write_kaldi(wavs, res / "wav.scp")
            ds = ASRDataset(wav_scp_file=res / "wav.scp", asr_model=asr.asr_model)
            t0 = time.time()
            hyp = asr.transcribe_audios(DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn), res / "text")
            print(f"{tag}: transcribed in {time.time() - t0:.0f} s", flush=True)
        wer = asr.compute_wer(refs, hyp, res / "wer").summarize("error_rate")
        rows.append(f"{tag}\t{len(utts)}\t{n_words}\t{wer:.2f}")
        print(rows[-1], flush=True)
    (out / "results.tsv").write_text("tag\tutts\twords\tWER\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
