"""Local-only adapters so the demo runs on this Windows machine.

The production code (a082/cascade, a082/service.py) expects two things that
are awkward here and would need no change on a Linux GPU server:

  1. ``openai-whisper`` with its ``.pt`` weights. We instead serve the *same*
     Whisper `medium.en` weights from the Hugging Face snapshot we already have,
     through ``transformers``. A stub ``whisper`` module is injected into
     ``sys.modules`` so ``a082.cascade.anonymize.Cascade`` uses it unchanged.
  2. ``speechbrain.pretrained.EncoderClassifier``, which builds symlinks inside
     ``savedir`` (needs SeCreateSymbolicLink on Windows) and wants
     ``mean_var_norm_emb.ckpt``. We run the same ECAPA graph by hand.

Nothing in this file is needed on the deployment server; it exists so the page
can be exercised locally without network access.

    python webapp/local_demo.py --config ...\\config.local.yaml --port 8013
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from a082.service import AnonService


# --------------------------------------------------------------- whisper stub
class TransformerWhisper:
    """Same Whisper weights, different runtime: ``transcribe`` mirrors openai-whisper."""

    def __init__(self, model_dir: str | Path, device: str = "cuda") -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.processor = WhisperProcessor.from_pretrained(str(model_dir))
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   fp16: bool = False, **_) -> dict:
        feats = self.processor.feature_extractor(
            np.asarray(audio, dtype=np.float32), sampling_rate=16000,
            return_tensors="pt").input_features.to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            ids = self.model.generate(feats, max_new_tokens=200)
        text = self.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        return {"text": text.strip()}


def install_whisper_stub(model_dir: str | Path) -> None:
    """Make ``import whisper; whisper.load_model(...)`` return TransformerWhisper."""
    module = types.ModuleType("whisper")
    module.load_model = lambda name, device="cpu", **_: TransformerWhisper(model_dir, device)
    sys.modules["whisper"] = module


# ----------------------------------------------------------------- ECAPA local
class _EcapaModel(torch.nn.Module):
    """spkrec-ecapa-voxceleb built from its checkpoints, no SpeechBrain fetch."""

    def __init__(self, model_dir: str | Path, device: str) -> None:
        super().__init__()
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        self.device = torch.device(device)
        self.fbank = Fbank(n_mels=80).to(self.device)
        self.input_norm = InputNormalization(norm_type="sentence", std_norm=False).to(self.device)
        self.embedding = ECAPA_TDNN(
            input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
            attention_channels=128, lin_neurons=192,
        ).to(self.device)
        state = torch.load(Path(model_dir) / "embedding_model.ckpt", map_location=self.device)
        self.embedding.load_state_dict(state)
        self.embedding.eval()
        # The 2 KB mean_var_norm_emb.ckpt stores the running mean under "glob_mean".
        # SpeechBrain 1.1 changed InputNormalization's buffers, so loading it into the
        # module fails; with std_norm=False the operation is just "subtract the mean",
        # which we do explicitly.
        self.glob_mean = None
        ck = Path(model_dir) / "mean_var_norm_emb.ckpt"
        if ck.exists():
            sd = torch.load(ck, map_location="cpu")
            if "glob_mean" in sd:
                self.glob_mean = sd["glob_mean"].float().to(self.device).reshape(-1)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(self.device).float()
        lens = torch.ones(wav.shape[0], device=self.device)
        feats = self.input_norm(self.fbank(wav), lens)
        emb = self.embedding(feats, lens)
        if self.glob_mean is not None:
            emb = emb - self.glob_mean.view(1, 1, -1) if emb.dim() == 3 else emb - self.glob_mean
        return F.normalize(emb.squeeze(1), p=2, dim=-1)


class LocalService(AnonService):
    """AnonService with an offline ECAPA similarity readout."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._ecapa = None

    def speaker_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if self._ecapa is None:
            self._ecapa = _EcapaModel(self.cfg["speaker_encoder"], self.device)
        ea = self._ecapa.encode(a.reshape(1, -1)).reshape(-1)
        eb = self._ecapa.encode(b.reshape(1, -1)).reshape(-1)
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))


