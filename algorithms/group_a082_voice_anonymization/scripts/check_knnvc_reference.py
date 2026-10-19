"""Check our kNN-VC against the official implementation (bshall/knn-vc via torch.hub) on real audio.

    python scripts/check_knnvc_reference.py --source <wav> --target <wav> [<wav> ...] [--device cuda] [--hub-dir DIR]

--hub-dir: an unpacked copy of github.com/bshall/knn-vc, for machines that cannot reach GitHub (torch.hub
then loads it with source="local"; the weights come from the torch hub checkpoint cache).

Same WavLM, same matching set, same query: our features after kNN must equal the official ones up to
float rounding, and so must the vocoded waveforms (both normalised to -16 LUFS). Prints the largest
differences and exits non-zero if they exceed the tolerance. (A near-tie between two neighbours can
resolve differently in the two distance formulas; a few isolated outliers are therefore not a bug.)
"""
from __future__ import annotations

import argparse
import sys

import torch

from a082 import audio
from a082.knnvc.core import knn_regress
from a082.models import load_hifigan, load_wavlm, vocode, wavlm_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--hub-dir", help="local copy of the bshall/knn-vc repository")
    args = ap.parse_args()

    kw = dict(prematched=True, pretrained=True, device=args.device)
    ref = (torch.hub.load(args.hub_dir, "knn_vc", source="local", **kw) if args.hub_dir
           else torch.hub.load("bshall/knn-vc", "knn_vc", trust_repo=True, **kw))
    matching_ref = ref.get_matching_set(args.target)                    # VAD 7, layer 6
    query_ref = ref.get_features(args.source)
    out_ref = ref.match(query_ref, matching_ref, topk=4)                 # -16 LUFS

    wavlm, hifigan = load_wavlm(args.device), load_hifigan(args.device, prematched=True)
    matching = torch.cat([wavlm_features(wavlm, audio.vad_trim(audio.load_audio(p))) for p in args.target]).cpu()
    query = wavlm_features(wavlm, audio.load_audio(args.source))
    y = knn_regress(query.float(), matching.to(args.device).float(), k=4)
    out = audio.loudness_normalize(vocode(hifigan, y))[0]

    d_match = (matching - matching_ref).abs().max().item()
    d_query = (query.cpu() - query_ref.cpu()).abs().max().item()
    n = min(out.shape[-1], out_ref.shape[-1])
    d_wav = (out[:n] - out_ref[:n]).abs().max().item()
    print(f"matching set {tuple(matching.shape)} max|diff| {d_match:.2e}; query max|diff| {d_query:.2e}; "
          f"output samples {out.shape[-1]} vs {out_ref.shape[-1]}, max|diff| {d_wav:.2e}")
    sys.exit(0 if max(d_match, d_query, d_wav) <= args.tol else 1)


if __name__ == "__main__":
    main()
