"""Web demo: upload or record speech, pick an anonymization system, listen to the result and see what changed.

    python webapp/app.py --config webapp/config.yaml [--host 0.0.0.0] [--port 7860]
"""
from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a082.service import AnonService  # noqa: E402
from a082.viz import compare_figure  # noqa: E402

INTRO = """\
# Speaker anonymization — HLTCOE JHU submission to VoicePrivacy 2024 (reproduction)

Upload or record English speech, choose a system from Table 1 of the paper, and compare the original with
the anonymized output. *Speaker similarity* is the cosine similarity of ECAPA speaker embeddings between the
two recordings: the lower it is, the less the output sounds like the same person to a speaker-verification model.
The *seed* fixes the random choices (target speaker, length factor, TTS voice, admixture draw); reuse it to
get the same output again.
"""


def to_tensor(audio_in) -> tuple[torch.Tensor, int]:
    sr, data = audio_in
    x = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        x /= np.iinfo(data.dtype).max
    if x.ndim == 2:
        x = x.mean(axis=1)
    return torch.from_numpy(x)[None], int(sr)


def build_ui(service: AnonService) -> gr.Blocks:
    systems = service.systems()
    choices = [(label, key) for key, label in systems.items()]

    def run(audio_in, system, seed, p, want_similarity):
        if audio_in is None:
            raise gr.Error("Please upload or record some speech first.")
        wav, sr = to_tensor(audio_in)
        if wav.shape[-1] > 30 * sr:
            raise gr.Error("Please keep recordings under 30 seconds.")
        seed = (seed or "").strip() or secrets.token_hex(4)
        t0 = time.time()
        res = service.anonymize(wav, sr, system, seed=seed, p=p)
        elapsed = time.time() - t0
        from a082.audio import resample
        orig16 = resample(wav, sr)
        info = {"system used": systems.get(res.system, res.system), "seed": seed,
                "processing time (s)": round(elapsed, 2), **res.details}
        if want_similarity:
            info["speaker similarity (original vs anonymized)"] = round(service.speaker_similarity(orig16, res.wav), 3)
        fig = compare_figure(orig16, res.wav)
        return (16000, res.wav.reshape(-1).numpy()), fig, info, seed

    with gr.Blocks(title="A082 speaker anonymization") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Input speech")
                system = gr.Dropdown(choices=choices, value=choices[0][1] if choices else None, label="System")
                p = gr.Slider(0.0, 1.0, value=0.4, step=0.025, label="Admixture: probability of the cascade (TTS)")
                with gr.Row():
                    seed = gr.Textbox(label="Seed (optional)", placeholder="random")
                    want_sim = gr.Checkbox(value=True, label="Compute speaker similarity")
                go = gr.Button("Anonymize", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Anonymized speech", type="numpy")
                info = gr.JSON(label="Details")
        plot = gr.Plot(label="Waveform and spectrogram")
        go.click(run, [audio_in, system, seed, p, want_sim], [audio_out, plot, info, seed])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="webapp/config.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    service = AnonService.from_yaml(args.config)
    build_ui(service).queue(max_size=16).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
