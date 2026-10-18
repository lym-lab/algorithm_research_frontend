"""Pretrained components (weights only; see README for what is ours and what is loaded).

- WavLM-Large, cut at layer 6 (kNN-VC release v0.1 copy of the microsoft/unilm checkpoint);
- HiFi-GAN vocoder from WavLM layer-6 features to 16 kHz audio, prematched variant
  (`prematch_g_02500000.pt`, trained on LibriSpeech train-clean-100).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from third_party.knnvc.hifigan.models import Generator as HiFiGAN
from third_party.knnvc.hifigan.utils import AttrDict
from third_party.knnvc.wavlm.WavLM import WavLM, WavLMConfig

WAVLM_URL = "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
HIFIGAN_URLS = {
    True: "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt",
    False: "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt",
}
FEATURE_LAYER = 6
HOP = 320  # WavLM frame = 20 ms at 16 kHz
_HIFIGAN_CFG = Path(__file__).resolve().parent.parent / "third_party" / "knnvc" / "hifigan" / "config_v1_wavlm.json"


def _state_dict(url: str, path: str | Path | None, device) -> dict:
    if path is not None:
        return torch.load(path, map_location=device)
    return torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)


def load_wavlm(device="cuda", path: str | Path | None = None) -> WavLM:
    ckpt = _state_dict(WAVLM_URL, path, device)
    model = WavLM(WavLMConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_hifigan(device="cuda", prematched: bool = True, path: str | Path | None = None) -> HiFiGAN:
    h = AttrDict(json.loads(_HIFIGAN_CFG.read_text()))
    gen = HiFiGAN(h).to(device)
    gen.load_state_dict(_state_dict(HIFIGAN_URLS[prematched], path, device)["generator"])
    gen.eval()
    gen.remove_weight_norm()
    return gen


@torch.inference_mode()
def wavlm_features(wavlm: WavLM, wav: torch.Tensor) -> torch.Tensor:
    """wav[1, T] at 16 kHz -> layer-6 features [frames, 1024].
    No input normalisation, exactly as kNN-VC extracts them (its vocoder was trained on these)."""
    x = wav.to(next(wavlm.parameters()).device)
    if x.dim() == 1:
        x = x[None]
    feats = wavlm.extract_features(x, output_layer=FEATURE_LAYER, ret_layer_results=False)[0]
    return feats.squeeze(0)


@torch.inference_mode()
def vocode(hifigan: HiFiGAN, feats: torch.Tensor) -> torch.Tensor:
    """features [frames, 1024] -> wav[1, frames*320] at 16 kHz."""
    dev = next(hifigan.parameters()).device
    return hifigan(feats[None].to(dev)).squeeze(1).cpu()
