"""Shape checks of the vendored networks with random weights (no downloads)."""
import json

import torch

from a082 import models
from third_party.knnvc.hifigan.models import Generator
from third_party.knnvc.hifigan.utils import AttrDict
from third_party.knnvc.wavlm.WavLM import WavLM, WavLMConfig


def test_hifigan_maps_frames_to_320_samples():
    h = AttrDict(json.loads(models._HIFIGAN_CFG.read_text()))
    g = Generator(h).eval()
    g.remove_weight_norm()
    wav = models.vocode(g, torch.randn(25, 1024))
    assert wav.shape == (1, 25 * models.HOP)


def test_wavlm_layer6_frames():
    cfg = WavLMConfig({"encoder_layers": 6, "encoder_embed_dim": 64, "encoder_ffn_embed_dim": 128,
                       "encoder_attention_heads": 4, "extractor_mode": "layer_norm"})
    m = WavLM(cfg).eval()
    f = models.wavlm_features(m, torch.randn(1, 16000))
    assert f.shape == (49, 64)  # 1 s -> 49 frames, as in kNN-VC
