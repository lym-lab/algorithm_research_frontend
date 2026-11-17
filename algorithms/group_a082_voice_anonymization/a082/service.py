"""Inference service behind the web demo: one call anonymizes one recording with any system.

Models load on first use, so a deployment that never selects the cascade never loads Whisper.
Configuration (YAML, see webapp/config.example.yaml):

    device: cpu | cuda
    pool: <matching-set cache of the demo target speakers>   # LibriTTS-R speakers for public demos
    piper_model: <en_US-libritts_r-medium.onnx>
    whisper_model: medium.en                                # the paper's model; base.en is faster on CPU
    wavlm_conv: {wc5: <ckpt>, wc6: <ckpt>, ...}             # optional, rows 5-8
    speaker_encoder: speechbrain/spkrec-ecapa-voxceleb      # for the similarity readout
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from . import admix, audio

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

LABELS = {
    "knn1": "1  kNN-VC (random target)",
    "knn1m": "1M kNN-VC, two pooled targets",
    "knn2": "2  kNN-VC + length perturbation",
    "knn3": "3  kNN-VC + length perturbation + feature noise",
    "tts": "4  Whisper -> Piper (cascade)",
    "mix": "1+4 admixture (p = TTS probability)",
    "wc5": "5  WavLM conversion (base)",
    "wc6": "6  WavLM conversion + adversarial",
    "wc7": "7  WavLM conversion + discrete",
    "wc8": "8  WavLM conversion + both",
}


@dataclass
class Result:
    wav: torch.Tensor                      # [1, T] at 16 kHz
    system: str                            # the system actually used (admixture resolves to knn1 or tts)
    details: dict = field(default_factory=dict)


class AnonService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._wavlm = self._hifigan = self._pool = self._cascade = self._spk = None
        self._knn: dict = {}
        self._wc: dict = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnonService":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------ what is available
    def systems(self) -> dict[str, str]:
        out = {}
        if self.cfg.get("pool"):
            out.update({k: LABELS[k] for k in ("knn1", "knn1m", "knn2", "knn3")})
        if self.cfg.get("piper_model"):
            out["tts"] = LABELS["tts"]
            if self.cfg.get("pool"):
                out["mix"] = LABELS["mix"]
        for k, p in (self.cfg.get("wavlm_conv") or {}).items():
            if p and Path(p).exists():
                out[k] = LABELS[k]
        return out

    # ------------------------------------------------------------ lazy components
    def _base(self):
        if self._wavlm is None:
            from .models import load_hifigan, load_wavlm
            self._wavlm = load_wavlm(self.device)
            self._hifigan = load_hifigan(self.device, prematched=True)
        return self._wavlm, self._hifigan

    def _knn_system(self, key: str):
        if key not in self._knn:
            from .knnvc.anonymize import KnnSystem
            from .knnvc.pool import Pool
            if self._pool is None:
                self._pool = Pool(self.cfg["pool"], device=self.device, min_frames=self.cfg.get("pool_min_frames", 3000))
            wavlm, hifigan = self._base()
            cfg = yaml.safe_load((CONFIGS / f"{key}.yaml").read_text())
            self._knn[key] = KnnSystem(cfg, self._pool, wavlm, hifigan, self.device)
        return self._knn[key]

    def _cascade_system(self):
        if self._cascade is None:
            from .cascade.anonymize import Cascade
            self._cascade = Cascade(self.cfg["piper_model"], self.device, self.cfg.get("whisper_model", "medium.en"))
        return self._cascade

    def _wc_model(self, key: str):
        if key not in self._wc:
            from .wavlm_conv.model import WavLMConverter
            ck = torch.load(self.cfg["wavlm_conv"][key], map_location="cpu")
            m = WavLMConverter(**ck["cfg"].get("model", {})).to(self.device).eval()
            m.load_state_dict(ck["model"])
            self._wc[key] = (m, ck.get("target_speaker"))
        return self._wc[key]

    # ------------------------------------------------------------ anonymize
    @torch.inference_mode()
    def anonymize(self, wav: torch.Tensor, sr: int, system: str, seed: str | None = None,
                  p: float = 0.4) -> Result:
        """`seed` plays the role of the utterance id: the same seed gives the same random choices."""
        seed = seed or secrets.token_hex(4)
        wav = audio.resample(wav.reshape(1, -1).float(), sr)
        if system == "mix":
            chosen = "tts" if admix.uses_tts(seed, p) else "knn1"
            r = self.anonymize(wav, audio.SR, chosen, seed)
            r.details.update(admixture_p=p, drawn=chosen)
            return r
        if system.startswith("knn"):
            s = self._knn_system(system)
            plan = s.plan(seed)
            out = s.convert(seed, wav, plan)
            details = {"targets": plan["targets"], "weights": [round(float(w), 3) for w in plan["weights"]]}
            if plan["ratio"] is not None:
                details["length_factor"] = round(plan["ratio"], 3)
            return Result(out, system, details)
        if system == "tts":
            c = self._cascade_system()
            text = c.transcribe(wav)
            out = c.convert(seed, wav, text)
            return Result(out, system, {"transcript": text, "piper_speaker": c.speaker(seed)})
        if system.startswith("wc"):
            from .models import vocode, wavlm_features
            model, target = self._wc_model(system)
            wavlm, hifigan = self._base()
            _, y = model(wavlm_features(wavlm, wav).float()[None])
            return Result(audio.loudness_normalize(vocode(hifigan, y[0])), system, {"target": target})
        raise KeyError(system)

    # ------------------------------------------------------------ readouts
    @torch.inference_mode()
    def speaker_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity of ECAPA speaker embeddings (16 kHz inputs)."""
        if self._spk is None:
            from speechbrain.pretrained import EncoderClassifier
            src = self.cfg.get("speaker_encoder", "speechbrain/spkrec-ecapa-voxceleb")
            self._spk = EncoderClassifier.from_hparams(source=src, savedir=self.cfg.get("speaker_encoder_dir", "cache/spk_ecapa"),
                                                       run_opts={"device": self.device})
        # normalize=True applies the model's mean-var centring. Without it the cosines are
        # inflated (a genuinely different voice scores ~0.40), which makes the 0.25
        # "same speaker" threshold in the UI meaningless.
        try:
            ea = self._spk.encode_batch(a.reshape(1, -1), normalize=True).reshape(-1)
            eb = self._spk.encode_batch(b.reshape(1, -1), normalize=True).reshape(-1)
        except AttributeError:   # hyperparams without mean_var_norm: fall back to raw embeddings
            ea = self._spk.encode_batch(a.reshape(1, -1)).reshape(-1)
            eb = self._spk.encode_batch(b.reshape(1, -1)).reshape(-1)
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))
