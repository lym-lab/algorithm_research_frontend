"""System 4 of Table 1: Whisper medium.en transcription, then Piper (VITS) synthesis with a random
LibriTTS-R voice per utterance (paper §2.2).

    python -m a082.cascade.anonymize --vpc-root <VPC repo> --piper-model <en_US-libritts_r-medium.onnx>
           [--suffix _tts] [--transcripts <dir>] [--sets all] [--shard i/n] [--device cuda]

Transcripts are cached per set in <transcripts>/<set>.txt (Kaldi text format), so the TTS step can be
re-run without Whisper and Whisper's own WER can be measured.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .. import audio, kaldi, runner, seeding

log = logging.getLogger(__name__)
EMPTY_SECONDS = 0.5  # output for an empty transcript


class Cascade:
    def __init__(self, piper_model: str | Path, device="cuda", whisper_model: str = "medium.en",
                 piper_cuda: bool = False):
        import whisper
        from piper import PiperVoice
        self.whisper = whisper.load_model(whisper_model, device=device)
        self.voice = PiperVoice.load(piper_model, use_cuda=piper_cuda)
        self.n_speakers = self.voice.config.num_speakers
        self.device = device

    def transcribe(self, wav: torch.Tensor) -> str:
        """Default `transcribe` arguments (temperature fallback, condition_on_previous_text)."""
        res = self.whisper.transcribe(wav.reshape(-1).numpy().astype(np.float32), language="en",
                                      fp16=str(self.device).startswith("cuda"))
        return res["text"].strip()

    def speaker(self, utt: str) -> int:
        return int(seeding.rng(utt, "tts-speaker").integers(self.n_speakers))

    def synthesize(self, text: str, speaker_id: int) -> torch.Tensor:
        """Text -> wav[1, T] at 16 kHz. Voice-file defaults for noise and length scales."""
        from piper import SynthesisConfig
        chunks = [c.audio_float_array for c in self.voice.synthesize(text, SynthesisConfig(speaker_id=speaker_id))]
        if not chunks:
            return torch.zeros(1, int(EMPTY_SECONDS * audio.SR))
        wav = torch.from_numpy(np.concatenate(chunks))[None]
        return audio.resample(wav, self.voice.config.sample_rate, audio.SR)

    def convert(self, utt: str, wav: torch.Tensor, text: str | None = None) -> torch.Tensor:
        if text is None:
            text = self.transcribe(wav)
        if not text:
            log.warning("%s: empty transcript, writing %.1f s of silence", utt, EMPTY_SECONDS)
            return torch.zeros(1, int(EMPTY_SECONDS * audio.SR))
        return self.synthesize(text, self.speaker(utt))


class TranscriptCache:
    """Kaldi-format transcript file per set; appended as utterances are transcribed."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data = kaldi.read_kaldi(path) if path.exists() else {}

    def get(self, utt: str):
        return self.data.get(utt)

    def put(self, utt: str, text: str) -> None:
        self.data[utt] = text
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{utt} {text}\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--piper-model", required=True)
    ap.add_argument("--suffix", default="_tts")
    ap.add_argument("--transcripts", type=Path, default=Path("work/transcripts"))
    ap.add_argument("--whisper-model", default="medium.en")
    runner.add_common_args(ap)
    args = ap.parse_args(argv)
    shard = runner.parse_shard(args.shard)
    sets = runner.resolve_sets(args.sets)
    if args.finalize_only:
        runner.anonymize_sets(args.vpc_root, sets, args.suffix, None, finalize_only=True)
        return
    cascade = Cascade(args.piper_model, args.device, args.whisper_model)
    for name in sets:
        cache = TranscriptCache(args.transcripts / f"{name}.{shard[0]}of{shard[1]}.txt")

        def convert(utt, wav):
            text = cache.get(utt)
            if text is None:
                text = cascade.transcribe(wav)
                cache.put(utt, text)
            return cascade.convert(utt, wav, text)

        runner.anonymize_sets(args.vpc_root, [name], args.suffix, convert, shard=shard)


if __name__ == "__main__":
    main()
