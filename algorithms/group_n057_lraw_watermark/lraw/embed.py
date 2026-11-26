# -*- coding: utf-8 -*-
"""水印嵌入（论文 IV-C 节，公式 6）。

嵌入规则（a=5, b=2，论文 IV-C 节）：
    s3_tilde = (s3 + a*s4) / (a + 1),  wi = 0   （向 s4 拉近，向下）
    s3_tilde = (s2 + b*s3) / (b + 1),  wi = 1   （向 s2 拉近，向上）
保持 s4 < s3_tilde < s2，奇异值降序结构不被破坏——这是提取侧
LightGBM 学习“中间奇异值偏置”的依据。

流程：VAD 去静音 -> 定长分段（默认 400 样本/段，对应 5x20 SVD 几何）
-> 逐段 DWT-GBT-SVD -> 量化 s3 -> SVD 重建 -> IGBT -> IDWT -> 拼接。
"""
from __future__ import annotations

import numpy as np

from .config import Config, DEFAULT_CONFIG, PRIMITIVE_POLYNOMIAL
from .vad import vad
from .transform import extract_features, reconstruct_with_modified_s3
from .msequence import generate_m_sequence, make_key


def segment_voiced(voiced: np.ndarray, seg_len: int) -> tuple[np.ndarray, int]:
    """把有声信号切成 seg_len 的等长段，返回 (段矩阵, 段数 n)。"""
    n = voiced.size // seg_len
    if n == 0:
        raise ValueError("voiced samples too short for embedding")
    return voiced[: n * seg_len].reshape(n, seg_len), n


def embed(audio: np.ndarray,
          watermark_bits: np.ndarray | list | str | None = None,
          cfg: Config = DEFAULT_CONFIG,
          seg_len: int | None = None,
          fill: int | None = None,
          repeat: int = 1) -> dict:
    """嵌入水印。

    参数
    ----
    audio         : 宿主音频（float 一维）。
    watermark_bits: 水印比特。可为 0/1 数组/列表、或二进制字符串（如
                    "110010..."）。None 时随机生成 n 比特（用于数据集构造）。
    seg_len       : 每段样本数。默认 400（= 5x20 SVD 需要的几何）。
    fill          : 若为 0 或 1，则所有段都嵌入该比特（数据集构造用）。
    repeat        : 重复度 k。同一比特连续写进 k 段（默认 1，即论文口径
                    “每段 1 比特”）。k>1 时容量降为 1/k，但每比特有 k 段
                    证据可供提取端融合平均，用于抗强攻击（L4 改进项）。

    返回 dict：
      audio_wm        : 含水印音频（仅有声部分，论文 IV-C 节做法）
      bits            : 逻辑比特（长度 = 段数 // repeat），即真正的水印
      n               : 逻辑比特数
      n_seg           : 实际被嵌入的段数（= n * repeat）
      L, taps, m_seq, key, seg_len, repeat
      snr_voiced      : 有声区域内原信号与水印信号的 SNR（dB）
    """
    cfg = cfg if cfg is not None else DEFAULT_CONFIG
    if seg_len is None:
        seg_len = 400
    repeat = max(1, int(repeat))

    if cfg.use_vad:
        voiced = vad(audio, cfg.sr, frame_ms=cfg.vad_frame_ms,
                     mode=cfg.vad_threshold_mode, threshold=cfg.vad_threshold)
    else:
        # 固定网格：从样本 0 起等长分段，保证嵌入/提取两侧网格完全一致
        #（论文 3s 音频 330 段 = 132300/400，无 VAD 裁剪；VAD 仅作可选预处理）
        voiced = np.asarray(audio, dtype=np.float64).ravel()
    segs_all, n_seg_all = segment_voiced(voiced, seg_len)

    # 重复写入时只用整组段（n_seg = nb * k），避免尾部半组破坏对齐
    nb = n_seg_all // repeat
    if nb == 0:
        raise ValueError(f"too few segments: {n_seg_all} < repeat={repeat}")
    n_seg = nb * repeat
    segs = segs_all[:n_seg]
    n = nb                                   # n 的语义是“逻辑比特数”

    # ---- 准备水印比特 ----
    if fill in (0, 1):
        bits = np.full(n, fill, dtype=np.int64)
    elif watermark_bits is None:
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=n).astype(np.int64)
    elif isinstance(watermark_bits, str):
        bits = np.array([int(c) for c in watermark_bits.strip() if c in "01"],
                        dtype=np.int64)
    else:
        bits = np.asarray(watermark_bits, dtype=np.int64).ravel()
    # 论文约定：每段恰好嵌入 1 比特；在 repeat=k 时是“每组 k 段 1 比特”。
    # 用户给的比特串过长则截断、过短则用固定种子随机补齐，保证长度一致。
    if bits.size != n:
        if bits.size > n:
            bits = bits[:n]
        else:
            rng = np.random.default_rng(1)
            pad = rng.integers(0, 2, size=n - bits.size).astype(np.int64)
            bits = np.concatenate([bits, pad])

    # ---- m 序列（纠错同步序列）与嵌入密钥 ----
    # L 取 ceil(log2(n+1))；要求一个完整 m 序列周期 2^L-1 不短于 n。
    L = cfg.L if (1 << cfg.L) - 1 >= n else int(np.ceil(np.log2(n + 1)))
    taps = cfg.lfsr_taps if L == cfg.L else PRIMITIVE_POLYNOMIAL.get(L)
    if taps is None or (1 << L) - 1 < n:
        raise ValueError(f"unsupported n={n}: need m-sequence length >= n")
    m_seq = generate_m_sequence(L, taps)
    if m_seq.size != n:
        # m 序列周期 n_m = 2^L - 1 ≥ n，只取前 n 位，使 M 与 W 等长。
        m_seq = m_seq[:n]
    # 嵌入密钥 K = W ⊕ M（论文公式）：发送端保存 K；接收端恢复出 M 后
    # 由 W = K ⊕ M 反解水印，M 本身不直接写进音频。
    key = make_key(bits, m_seq)

    # 逐段使用的比特：第 i 组（i//k）的比特，铺满 n_seg 段
    bits_seg = np.repeat(bits, repeat)

    # ---- 逐段嵌入：对每一段单独做“量化第 3 奇异值 s3” ----
    a, b = cfg.a, cfg.b
    out_segs = np.empty_like(segs)
    for i in range(n_seg):
        seg = segs[i]
        # 先正向提取该段的 DWT-GBT-SVD 奇异值 s1≥s2≥s3≥s4≥s5
        feats = extract_features(seg, cfg.wavelet, cfg.dwt_levels,
                                 cfg.svd_rows, cfg.svd_cols)
        s = feats.S
        if bits_seg[i] == 0:
            # 公式6-w=0：s3̃=(s3+a·s4)/(a+1)，用 a=5 加权把 s3 向下拉向 s4，
            # 使 s3 偏小（偏向下位），对应“嵌入 0”。
            s3_tilde = (s[2] + a * s[3]) / (a + 1.0)
        else:
            # 公式6-w=1：s3̃=(s2+b·s3)/(b+1)，用 b=2 加权把 s3 向上抬向 s2，
            # 使 s3 偏大（偏向上位），对应“嵌入 1”。
            s3_tilde = (s[1] + b * s[2]) / (b + 1.0)
        # 改 s3 后逆 SVD/IGBT/IDWT 重建时域段；其余奇异值与细节系数保持不变
        seg_out, _ = reconstruct_with_modified_s3(
            seg, s3_tilde, cfg.wavelet, cfg.dwt_levels, cfg.svd_rows, cfg.svd_cols,
            feats=feats)
        out_segs[i] = seg_out

    audio_wm = out_segs.reshape(-1)

    # ---- 嵌入保真度 SNR（公式7，仅反映“水印造成的失真”，与后续攻击无关）----
    # SNR=10·lg(原始能量/嵌入扰动能量)；只在被嵌入的有声/网格区域内统计。
    orig_voiced = segs.reshape(-1)
    noise = audio_wm - orig_voiced
    snr_v = 10.0 * np.log10(
        (orig_voiced ** 2).sum() / (noise ** 2).sum()
    ) if (noise ** 2).sum() > 0 else float("inf")

    return {
        "audio_wm": audio_wm,
        "bits": bits,
        "n": n,
        "n_seg": n_seg,
        "repeat": repeat,
        "L": L,
        "taps": taps,
        "m_seq": m_seq,
        "key": key,
        "seg_len": seg_len,
        "snr_voiced": float(snr_v),
        "voiced_len": n_seg * seg_len,
    }
