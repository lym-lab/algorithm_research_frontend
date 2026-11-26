# -*- coding: utf-8 -*-
"""
LRAW —— LightGBM-Based Audio Watermarking Robust to Recapturing and
Hybrid Attacks（IEEE TIFS 2025）课程大作业复现项目。

包结构：
  config    全局配置（论文参数集中管理）
  audio_io  音频读写
  vad       帧能量 VAD（静音剔除）
  transform DWT-GBT-SVD 系数提取 / 重建
  msequence m 序列生成与重建
  embed     水印嵌入（奇异值量化规则）
  extract   LightGBM 逐段水印比特提取
  recover   滑窗匹配 + m 序列完整恢复
  model     LightGBM 训练 / 加载 / 评估
  attacks   表 III 攻击库 + 数字重录信道模型
  metrics   SNR / BER / ODG
  pipeline  端到端流水线
"""
from .config import Config, DEFAULT_CONFIG
from .audio_io import read_audio, write_audio
from .vad import vad
from .transform import extract_features, reconstruct_with_modified_s3
from .msequence import generate_m_sequence, make_key
from .embed import embed
from .extract import extract_bits
from .recover import recover_watermark, align_length, align_probs, blind_recover
from .model import load_model, train_model, evaluate
from .attacks import apply_attack, hybrid_attack, ATTACK_REGISTRY
from .metrics import snr, ber, odg
from .pipeline import run_pipeline
from .synth import synth_music, synth_speech_like

__all__ = [
    "Config", "DEFAULT_CONFIG",
    "read_audio", "write_audio", "vad",
    "extract_features", "reconstruct_with_modified_s3",
    "generate_m_sequence", "make_key",
    "embed", "extract_bits", "recover_watermark", "blind_recover",
    "align_length", "align_probs",
    "load_model", "train_model", "evaluate",
    "apply_attack", "hybrid_attack", "ATTACK_REGISTRY",
    "snr", "ber", "odg", "run_pipeline", "synth_music", "synth_speech_like",
]

__version__ = "1.0.0"
