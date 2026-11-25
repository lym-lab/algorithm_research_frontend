# -*- coding: utf-8 -*-
"""水印提取（论文 IV-D 节）：LightGBM 把“每段嵌入的是 0 还是 1”
当作二分类问题，直接对 DWT-GBT-SVD 归一化奇异值做预测。
"""
from __future__ import annotations

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .vad import vad
from .transform import extract_features


def extract_bits(audio: np.ndarray,
                 model,
                 cfg: Config = DEFAULT_CONFIG,
                 seg_len: int | None = None,
                 expected_len: int | None = None,
                 tol: float = 0.03,
                 repeat: int = 1) -> dict:
    """从（攻击后的）音频中逐段预测水印比特。

    expected_len：接收端已知的原始样本数（= 载荷比特数 × 段长 × 重复度）。
      当攻击只造成“时长变化”时，接收端应按已知时长把音频重采样回原长，
      再做逐段特征——这是所有实际水印检测器的同步步骤：
        - 重录的 100 ppm 时钟漂移（0.01%）：不归一会累计 48 样本漂移，
          且重采样滤波器带亚样本相位偏移，BER 从 4% 被抬到 30%+；
        - 纯变速类 TSM（±10~20%）：不归一 BER 约 47%，归一到原长后 0.6%
          （时间轴映射严格线性，见 attacks._resample_scale 的注释）。
      tol 表示只把“变化幅度 < tol”当成同步误差并归一：
        - 默认 0.03 用于普通攻击（裁剪类 LSC 会删掉大段音频，不能归一，
          否则会把删段当变速，反而破坏索引对齐）；
        - 变速/重录类攻击在评估时显式传 tol=0.25（见 run_attack_eval）。

    repeat：嵌入侧把同一比特连续写进 k 段时的重复度。提取侧对每 k 段
      的“判为 1 的概率”取均值再判决（软判决融合），把 k 段的证据合并成
      一个比特。k=1 时行为与论文完全一致（每段 1 比特）。

    返回：
      bits       : 每个（重复组）的比特判决，长度 = 段数 // repeat
      probs      : 对应的“判为 1”概率（融合后，可用于软判决/对齐）
      n          : 比特数（= 段数 // repeat）
      n_seg      : 实际参与判决的段数
      seg_len, repeat
    """
    cfg = cfg if cfg is not None else DEFAULT_CONFIG
    if seg_len is None:
        seg_len = 400
    repeat = max(1, int(repeat))

    audio = np.asarray(audio, dtype=np.float64).ravel()
    if expected_len and expected_len > 0:
        dev = abs(audio.size - expected_len) / float(expected_len)
        if 1e-12 < dev < tol:                     # 时长变化 → 按已知时长归一
            from scipy import signal as _sg
            audio = _sg.resample_poly(audio, expected_len, audio.size)
            audio = audio[:expected_len]

    if cfg.use_vad:
        voiced = vad(audio, cfg.sr, frame_ms=cfg.vad_frame_ms,
                     mode=cfg.vad_threshold_mode, threshold=cfg.vad_threshold)
    else:
        # 固定网格：与嵌入侧完全一致（保证逐段特征与嵌入比特对齐）
        voiced = audio
    n_seg = voiced.size // seg_len
    if n_seg < repeat:
        return {"bits": np.array([], dtype=np.int64), "probs": np.array([]),
                "n": 0, "n_seg": 0, "seg_len": seg_len, "repeat": repeat}

    segs = voiced[: n_seg * seg_len].reshape(n_seg, seg_len)
    # 特征矩阵 feats：段数行 × 5 列（每段 5 个归一化奇异值 s1..s5），
    # 与训练数据集列结构一致，直接作为 LightGBM 二分类输入。
    feats = np.zeros((n_seg, 5), dtype=np.float64)
    for i in range(n_seg):
        f = extract_features(segs[i], cfg.wavelet, cfg.dwt_levels,
                             cfg.svd_rows, cfg.svd_cols)
        feats[i] = f.singular_values

    # 模型输出每段判为“嵌入 1”的概率；阈值 0.5 做硬判决得到比特序列。
    # probs 保留软概率，供变时长攻击时做长度归一化（align_length）。
    probs_seg = model.predict_proba(feats)[:, 1]

    # ---- 重复度融合：每 repeat 段的概率取均值，再按 0.5 判决 ----
    # 为什么在“概率域”平均而不是“比特域”投票：概率保留了置信度，
    # 弱证据段（概率 0.7）与强证据段（0.95）按均值合成更稳；若改成
    # 比特多数投票，两段 0.51 会压过一段 0.95，信息被丢掉。
    nb = n_seg // repeat
    probs = probs_seg[: nb * repeat].reshape(nb, repeat).mean(axis=1)
    bits = (probs >= 0.5).astype(np.int64)
    return {"bits": bits, "probs": probs, "n": nb,
            "n_seg": nb * repeat, "seg_len": seg_len, "repeat": repeat}
