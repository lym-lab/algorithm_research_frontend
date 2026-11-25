# -*- coding: utf-8 -*-
"""水印恢复（论文 IV-E 节，Algorithm 1 + m 序列重建）。

滑窗匹配：
  在原始水印 W 与提取的（含错）水印 W~ 上同步滑动，寻找一段
  长度恰好为 L、且两者完全一致的比特窗口 W_anchor；
  该窗口对应的密钥段 K_anchor 与 W_anchor 异或，得到 m 序列的
  L 个连续比特，再依据 m 序列周期性重建完整 M，最后
  W_recovered = K xor M，实现“只需 L 个正确比特即可零误码恢复”。

口径警示（重要）：本函数接收真值 W 作为滑窗匹配参照，属于
“oracle / 先知上界”实现——只要命中锚点，重建在构造上恒为 BER≈0，
不能单独当作真实接收方的鲁棒性证据。真实接收方（仅持 K/L/taps、
不持 W）的盲恢复判断见 `lraw.recover.blind_recover`。
"""
from __future__ import annotations

import numpy as np

from .msequence import recover_m_sequence_from_window


def blind_recover(W_tilde: np.ndarray,
                  key: np.ndarray,
                  L: int,
                  taps: list | None = None,
                  probs: np.ndarray | None = None) -> dict:
    """盲恢复（真实接收方口径：只有 W~、K、L、taps，不持真值 W）。

    原理（论文命题 3 的接收端实现）：
      1) 观测 m 序列 M_obs = K ⊕ W~（嵌入时 K = W ⊕ M，故 K ⊕ W = M；
         逐位错码会让 M_obs 的对应位置翻错）；
      2) m 序列是周期 P = 2^L - 1 的循环序列，整条序列完全由相位 d 决定：
         M_d[i] = M_full[(i + d) mod P]；
      3) 枚举 d ∈ [0, n)，取与 M_obs 一致率最高者。若最优一致率明显高于
         次优（margin），说明相位被唯一确定 → 判定“恢复成功”；
      4) 由该相位重建 M，再 W_hat = K ⊕ M。

    两项工程优化（不改变论文的 m 序列恢复原理）：
      - **按实际可观测长度 m 打分，不再补零到 n**：裁剪后只剩 m 位时，补零会
        把“全对的前 m 位”稀释成 ≈50%，导致明明能恢复却失锁。这里只统计真实
        观测到的 m 位，再用已知 K 重建完整 n 位水印。
      - **软判决打分**：传入分类器概率 probs 时，用 c=2p-1 与候选 ±1 序列做
        相关（等价于按置信度加权的一致率），弱证据段权重自动降低；不传则退回
        硬判决。一致率仍按 [0,1] 口径返回，兼容旧调用。
      - 锁定阈值随 m 自适应：短片段噪声更容易“碰巧一致”，阈值按零假设
        0.5 + 5.5·(0.5/√m) 取 5.5σ；不再用固定 0.60 的一刀切门槛。
        长序列（n≈1323）只要相位显著性足够（≈58%）就能锁定，短序列自动更严。
      - 最少 20 位观测才判锁定：13 位在“已知 W”的 oracle 口径里够用，但盲恢复
        要在 1323 个相位里挑最大值，13 位全对也会被随机序列碰中（仿真实测约 15%
        虚警），因此盲口径要求 ≥20 位。

    与 recover_watermark（oracle 口径，借真值找锚点）的区别：这里不碰 W，
    因此它的锁定率就是“真实接收方能否恢复”的可报告指标。
    返回：agreement（一致率）、margin、phase_d、locked、W（恢复比特）、
          n_obs（实际参与打分的位数）、threshold（锁定阈值）、z（显著性）。
    """
    K = np.asarray(key, dtype=np.int64).ravel()
    Wt = np.asarray(W_tilde, dtype=np.int64).ravel()
    n = int(K.size)
    # 只取“真实观测到”的位数，绝不补零：裁剪后剩余 m 位本身就是完整证据
    m = int(min(Wt.size, n))
    if m <= 0:
        return {"agreement": 0.0, "second": 0.0, "margin": 0.0,
                "phase_d": 0, "locked": False, "W": K.copy(),
                "n_obs": 0, "threshold": 1.0, "z": 0.0}
    Wt = Wt[:m]

    # 软判决权重：c=+1 表示“很可能是 1”，c=-1 表示“很可能是 0”；
    # 概率越接近 0.5 权重越小，避免把弱证据当成硬事实。
    if probs is not None:
        p = np.asarray(probs, dtype=np.float64).ravel()
        if p.size < m:
            p = np.concatenate([p, np.full(m - p.size, 0.5)])
        c = 2.0 * p[:m] - 1.0
    else:
        c = 2.0 * Wt.astype(np.float64) - 1.0

    if taps is None:
        from .config import PRIMITIVE_POLYNOMIAL
        taps = PRIMITIVE_POLYNOMIAL[L]
    from .msequence import generate_m_sequence
    M_full = generate_m_sequence(L, taps)
    P = int(M_full.size)

    # 枚举 m 序列相位：处理提取序列相对 m 序列起点的相位偏移。
    # 注意：这里只搜 M 的相位，不额外搜内容时间轴起点（K 仍从 0 对齐），
    # 因此主要覆盖保留前缀的裁剪、等时长攻击与重采样对齐后的场景；
    # 若攻击从音频中间任意起点裁剪，需再做时间轴同步（见文档边界说明）。
    # 每个相位只用 m 个观测位打分；O(n·m)，n≈1323 时仍是毫秒级。
    idx = np.arange(m)
    scores = np.empty(n, dtype=np.float64)
    for d in range(n):
        # 候选水印 = K ⊕ 候选 M；软权重 c 对应的是“提取出的 W̃”，
        # 因此必须和候选 W 比，不能直接和 M 比（否则干净信道也会失锁）。
        cand_w = np.bitwise_xor(K[:m], M_full[(idx + d) % P])
        scores[d] = float(np.mean(c * (2.0 * cand_w - 1.0)))
    order = np.argsort(-scores)
    best_d = int(order[0])
    best_score = float(scores[best_d])
    second_score = float(scores[order[1]]) if n > 1 else -1.0

    # 转回一致率口径，兼容旧字段/旧测试
    best = (best_score + 1.0) / 2.0
    second = (second_score + 1.0) / 2.0
    margin = best - second

    # 自适应阈值：m 位随机序列的零假设一致率为 0.5、标准差 0.5/√m；
    # 取 5.5σ 控制“1323 个相位里挑最大值”的多重比较虚警。m<20 时证据不足，
    # 阈值会顶到 1.0（只接受全对窗口），且直接不判锁定。
    std = 0.5 / np.sqrt(m) if m > 0 else 0.5
    threshold = min(1.0, 0.5 + 5.5 * std)
    z = (best - 0.5) / std if std > 0 else 0.0
    locked = bool(m >= 20 and best >= threshold and margin > 0.03)

    # 由相位重建完整 n 位 M（即使只观测到 m 位，LFSR 周期也能补全整条），
    # 再 W_hat = K ⊕ M。这一步与 oracle 的“13 位锚点重建”在数学上同源。
    M_rec = M_full[(np.arange(n) + best_d) % P]
    W_hat = np.bitwise_xor(K, M_rec)
    return {"agreement": best, "second": second, "margin": margin,
            "phase_d": best_d, "locked": locked, "W": W_hat,
            "n_obs": m, "threshold": float(threshold), "z": float(z)}


def align_probs(probs: np.ndarray, n: int) -> np.ndarray:
    """把变时长后的“概率序列”线性插值回 n 位，保留置信度。

    旧版 `align_length` 先硬判决再插值，边界位会被 0/1 截断丢失置信度；
    盲恢复的软判决需要概率本身，故单独提供本函数。返回 [0,1] 概率。
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    m = probs.size
    if m == n:
        return probs.copy()
    if m == 0:
        return np.full(n, 0.5, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, m)
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, probs)


def align_length(probs: np.ndarray, n: int) -> np.ndarray:
    """把长度变化后的提取概率序列重采样回 n 位（TSM 等变时长攻击）。

    时间缩放攻击会整体改变音频长度，导致提取段数 n' != n，逐位比较
    无法对齐。此处把概率序列线性插值回 n 位再判决，属于论文 Algorithm 1
    的工程补充（论文未显式给出 TSM 的对齐步骤，但其报告的 TSM BER=0
    隐式依赖等价对齐）。返回 0/1 序列。
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    m = probs.size
    if m == n:
        return (probs >= 0.5).astype(np.int64)
    if m == 0:
        return np.zeros(n, dtype=np.int64)
    x_old = np.linspace(0.0, 1.0, m)
    x_new = np.linspace(0.0, 1.0, n)
    p = np.interp(x_new, x_old, probs)
    return (p >= 0.5).astype(np.int64)


def find_anchor_window(W: np.ndarray, W_tilde: np.ndarray, L: int):
    """扫描 W 与 W~ 完全一致的 L 位窗口，返回起始下标；找不到返回 None。

    实现：eq[i]=1 表示第 i 位判决正确；run 累计“当前连续正确位数”，
    一旦 run≥L 说明存在一段长度 L 的全对窗口，其起点为 i-L+1。
    """
    W = np.asarray(W, dtype=np.int64).ravel()
    Wt = np.asarray(W_tilde, dtype=np.int64).ravel()
    m = min(W.size, Wt.size)
    if m < L:
        return None
    eq = (W[:m] == Wt[:m]).astype(np.int64)
    run = 0
    for i in range(m):
        run = run + 1 if eq[i] else 0   # 正确则连续计数+1，错误则清零重新计数
        if run >= L:
            return i - L + 1
    return None


def recover_watermark(W: np.ndarray,
                      W_tilde: np.ndarray,
                      key: np.ndarray,
                      L: int,
                      taps: list | None = None) -> dict:
    """由含错的 W~ 恢复完整水印。

    返回：
      recovered : 恢复出的水印（与 W 等长）；失败时为 None
      anchor    : 找到的窗口起始下标；None 表示滑窗匹配失败
      ber_after : 恢复水印与原始 W 的 BER（成功时恒为 0）
    """
    # 第一步：在真值参照下找一段 L 位全对的锚点窗口；找不到直接判失败。
    anchor = find_anchor_window(W, W_tilde, L)
    if anchor is None:
        return {"recovered": None, "anchor": None, "ber_after": None}

    # 第二步：取锚点处的 L 位真值 W 与密钥 K，由 M=W⊕K 反解出 m 序列的
    # L 个连续比特（因为嵌入时 K=W⊕M，所以 W⊕K=M）。
    W_anchor = np.asarray(W, dtype=np.int64).ravel()[anchor: anchor + L]
    K = np.asarray(key, dtype=np.int64).ravel()
    K_anchor = K[anchor: anchor + L]
    M_anchor = np.bitwise_xor(W_anchor, K_anchor)

    # 第三步：m 序列是周期为 2^L-1 的循环序列，任意 L 个连续比特足以确定
    # 整条序列；据此由 M_anchor 递推重建一个完整周期 M_full。
    M_full = recover_m_sequence_from_window(M_anchor, L, taps)
    # M_full 是以 anchor 为起点排列的完整周期（M_full[k]=M[(anchor+k) mod n]）；
    # 用 np.roll(M_full, anchor) 循环移位回时间轴起点对齐，再截前 n 位
    # （嵌入侧只保留了 m 序列前 n 位）。
    n = K.size
    M_aligned = np.roll(M_full, anchor)[:n]
    # 第四步：W_recovered = K ⊕ M_aligned，由密钥与重建序列反解完整水印。
    recovered = np.bitwise_xor(K[:n], M_aligned)

    # 统计恢复结果与真值 W 的误码率（命中锚点时构造上恒为 0，仅供上界参考）
    ber_after = float(np.mean(recovered != np.asarray(W, dtype=np.int64).ravel()))
    return {"recovered": recovered, "anchor": anchor, "ber_after": ber_after}
