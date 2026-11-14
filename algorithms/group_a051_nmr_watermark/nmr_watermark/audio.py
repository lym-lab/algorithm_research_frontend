import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .model import SAMPLE_RATE, SEGMENT_SAMPLES


def read_audio(path):
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("音频为空或包含非有限数值")
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return audio


def chunks(audio):
    length = len(audio)
    padded = np.pad(audio, (0, (-length) % SEGMENT_SAMPLES))
    return padded.reshape(-1, SEGMENT_SAMPLES), length


def save_audio(path, audio):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # IEEE float WAV avoids unreported clipping/PCM quantization in reproduction.
    sf.write(str(path), audio, SAMPLE_RATE, subtype="FLOAT")
