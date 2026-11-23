"""A stand-in for AnonService with no models: the "anonymized" signal is the input slowed to 0.8x.
Used by the API test and to look at the web pages locally (`python tests/fake_service.py`)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a082.service import LABELS, Result  # noqa: E402


class FakeService:
    def __init__(self, keys=None):
        self.keys = keys or list(LABELS)

    def systems(self):
        return {k: LABELS[k] for k in self.keys}

    def anonymize(self, wav, sr, system, seed=None, p=0.4):
        out = torchaudio.functional.resample(wav.reshape(1, -1), 4, 5)  # 0.8x speed, lower pitch
        if system == "mix":
            return Result(out, "tts", {"admixture_p": p, "drawn": "tts", "transcript": "(fake)"})
        details = {"targets": ["id10001"]} if system.startswith("knn") else {"transcript": "(fake)", "piper_speaker": 17}
        return Result(out, system, details)

    def speaker_similarity(self, a, b):
        return 0.123


if __name__ == "__main__":
    import argparse

    import uvicorn

    from webapp.server import create_app

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(create_app({"similarity_threshold": 0.25}, FakeService()), host="127.0.0.1", port=args.port)
