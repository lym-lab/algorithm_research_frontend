import json
from pathlib import Path
import numpy as np
import torch
from .audio import read_audio, chunks, save_audio
from .model import WatermarkModel, SAMPLE_RATE, SEGMENT_SAMPLES


def load_model(checkpoint, device="cpu"):
    # Only load locally produced/trusted PyTorch checkpoints.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = WatermarkModel(state["config"]["bits"], state["config"].get("stft_normalized", False)).to(device).eval()
    model.load_state_dict(state["model"])
    return model, state["config"]


def parse_message(text, bits):
    text = "".join(text.split())
    if not text:
        return np.random.default_rng().integers(0, 2, bits).astype(np.float32)
    if len(text) != bits or set(text) - {"0", "1"}:
        raise ValueError(f"请输入恰好 {bits} 位 0/1，或留空随机生成")
    return np.array([int(v) for v in text], dtype=np.float32)


@torch.no_grad()
def embed_array(model, audio, message):
    blocks, length = chunks(audio)
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(blocks), 8):
        x = torch.from_numpy(blocks[start:start+8]).to(device)
        m = torch.as_tensor(message, device=device).expand(len(x), -1)
        marked, _, _, _ = model(x, m)
        values.append(marked.cpu().numpy())
    # Keep final padded block in downloadable WAV so independent extraction
    # observes exactly the same samples as the embedder generated.
    return np.concatenate(values).reshape(-1), length


@torch.no_grad()
def extract_array(model, audio):
    blocks, _ = chunks(audio)
    device = next(model.parameters()).device
    return np.concatenate([model.extract(torch.from_numpy(blocks[i:i+8]).to(device)).cpu().numpy()
                           for i in range(0, len(blocks), 8)])


def embed_file(checkpoint, source, destination, text="", device="cpu"):
    model, config = load_model(checkpoint, device)
    message = parse_message(text, model.bits)
    audio = read_audio(source)
    marked, length = embed_array(model, audio, message)
    save_audio(destination, marked)
    recovered = extract_array(model, read_audio(destination))
    info = {"bits": model.bits, "message": "".join(map(str, message.astype(int))),
            "sample_rate": SAMPLE_RATE, "original_samples": length,
            "output_samples": len(marked), "segment_samples": SEGMENT_SAMPLES,
            "ber_after_save": float(np.mean((recovered >= .5) != message)),
            "per_block_ber": np.mean((recovered >= .5) != message, axis=1).tolist(),
            "peak": float(np.abs(marked).max()), "pilot": config.get("pilot", False),
            "alignment_required": True, "wav_subtype": "FLOAT"}
    Path(destination).with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info
