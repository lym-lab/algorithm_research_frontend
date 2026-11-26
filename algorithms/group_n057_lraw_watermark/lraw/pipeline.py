# -*- coding: utf-8 -*-
"""端到端流水线：嵌入 ->（可选攻击）-> 提取 -> 恢复 -> 指标。

供 scripts/ 与 webui 复用的统一入口。
"""
from __future__ import annotations

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .embed import embed
from .extract import extract_bits
from .recover import recover_watermark
from .metrics import snr, ber
from .attacks import apply_attack, hybrid_attack


def run_pipeline(audio: np.ndarray,
                 watermark_bits=None,
                 attack: str | None = None,
                 attack_kw: dict | None = None,
                 cfg: Config = DEFAULT_CONFIG,
                 model=None,
                 hybrid: tuple | None = None,
                 seg_len: int | None = None) -> dict:
    """单次端到端实验。

    参数
    ----
    audio         : 宿主音频
    watermark_bits: 水印比特（None 时随机）
    attack        : 攻击名（如 "mp3c"/"recapture"）；None 表示无攻击
    attack_kw     : 攻击参数覆盖
    hybrid        : (base_attack, second_attack) 混合攻击
    model         : LightGBM 模型；None 时自动加载作者模型

    返回 dict：完整中间量与指标。
    """
    cfg = cfg if cfg is not None else DEFAULT_CONFIG
    if model is None:
        from .model import load_model
        model = load_model()

    # 阶段①发送端嵌入：返回含水印音频、真值 W、密钥 K、m 序列阶数 L/抽头 taps
    emb = embed(audio, watermark_bits, cfg, seg_len=seg_len)
    wm_audio = emb["audio_wm"]

    # 阶段②信道攻击：三选一——混合攻击 / 单种攻击 / 无攻击（干净传输）
    if hybrid is not None:
        attacked = hybrid_attack(wm_audio, cfg.sr, base=hybrid[0],
                                 second=hybrid[1], cfg=cfg,
                                 **(attack_kw or {}))
        attack_name = f"hybrid:{hybrid[0]}+{hybrid[1]}"
    elif attack is not None:
        attacked = apply_attack(attack, wm_audio, cfg.sr, cfg,
                                **(attack_kw or {}))
        attack_name = attack
    else:
        attacked = wm_audio
        attack_name = "none"

    # 阶段③接收端提取：LightGBM 对攻击后音频逐段判 0/1，得到含错序列 W~
    ext = extract_bits(attacked, model, cfg, seg_len=seg_len)
    # 阶段④恢复：m 序列滑窗重建（本实现为 oracle 口径，借真值 W 找锚点）
    rec = recover_watermark(emb["bits"], ext["bits"], emb["key"],
                            emb["L"], emb["taps"])

    result = {
        "attack": attack_name,
        "n": emb["n"],
        "snr_voiced": emb["snr_voiced"],
        "extracted_raw_ber": ber(emb["bits"], ext["bits"]),
        "recovered_ber": rec["ber_after"],
        "anchor": rec["anchor"],
        "bits": emb["bits"],
        "extracted_bits": ext["bits"],
        "recovered_bits": rec["recovered"],
        "wm_audio": wm_audio,
        "attacked_audio": attacked,
        "key": emb["key"],
        "L": emb["L"],
        "taps": emb["taps"],
        "seg_len": emb["seg_len"],
    }
    return result
