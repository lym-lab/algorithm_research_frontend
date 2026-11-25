# -*- coding: utf-8 -*-
"""m 序列（最大长度 LFSR 序列）生成与重建（论文 IV-B / IV-E 节）。

论文：用 L = ceil(log2(|W|+1)) 级 LFSR 生成与 W 等长的二进制 m 序列 M，
密钥 K = W xor M。提取侧通过“滑动窗口找到 L 个连续一致比特”，再利用
m 序列的周期性（任意连续 L 个比特可确定整条序列）重建完整 M，从而
恢复 W —— 这是 LRAW 抗大比例裁剪/同步错位的关键（命题 3）。

本模块自带自洽性检验：任何 L 位连续窗口都能无歧义重建整条序列。
"""
from __future__ import annotations

import numpy as np

from .config import Config, PRIMITIVE_POLYNOMIAL


def lfsr_next(state: np.ndarray, taps: list) -> int:
    """Fibonacci LFSR 一步：返回输出比特并推进状态。

    taps 形如 [13, 4, 3, 1]（本原多项式指数，不含最高次项与常数项）。
    反馈 = 各抽头位置比特的异或；输出 = 移位前最高位。
    """
    fb = 0
    for t in taps:
        fb ^= int(state[t - 1])
    out = int(state[-1])
    state[1:] = state[:-1]
    state[0] = fb
    return out


def generate_m_sequence(L: int, taps: list | None = None) -> np.ndarray:
    """生成长度为 n = 2^L - 1 的 m 序列（一个完整周期）。

    参数
    ----
    L   : LFSR 级数；n = 2^L - 1。
    taps: 本原多项式指数列表；None 时从 config.PRIMITIVE_POLYNOMIAL 取。
    """
    if taps is None:
        taps = PRIMITIVE_POLYNOMIAL[L]
    n = (1 << L) - 1
    # 初始状态取“非全零”任意值（全零态是死循环）
    state = np.zeros(L, dtype=np.int64)
    state[0] = 1
    bits = np.zeros(n, dtype=np.int64)
    for i in range(n):
        bits[i] = lfsr_next(state, taps)
    return bits


def verify_m_sequence(L: int, taps: list) -> bool:
    """校验 (L, taps) 生成的是真 m 序列（周期 = 2^L - 1 且各态互异）。"""
    n = (1 << L) - 1
    state = np.zeros(L, dtype=np.int64)
    state[0] = 1
    seen = set()
    for _ in range(n):
        seen.add(tuple(state.tolist()))
        lfsr_next(state, taps)
    return len(seen) == n


def _recurrence_next(bits: np.ndarray, L: int, taps: list) -> int:
    """给定最近的 L 个比特（bits 为当前状态，按时间序存放），返回下一比特。

    对 Fibonacci LFSR，若状态比特为 b[0..L-1]（b[L-1] 最新），
    则 b_new = xor( b[L-t] for t in taps )。
    """
    fb = 0
    for t in taps:
        fb ^= int(bits[L - t])
    return fb


def extend_m_sequence(window: np.ndarray, L: int, taps: list) -> np.ndarray:
    """从连续 L 个比特重建整条长度 n = 2^L - 1 的 m 序列。

    m 序列是周期 n 的循环序列：窗口给出序列上某个位置起的 L 个比特，
    用线性递推向前补足剩余 n - L 个比特，即得到完整周期（循环意义下）。
    """
    window = np.asarray(window, dtype=np.int64).ravel()
    if window.size != L:
        raise ValueError(f"window size {window.size} != L={L}")
    n = (1 << L) - 1
    seq = np.zeros(n, dtype=np.int64)
    seq[:L] = window
    for i in range(L, n):
        seq[i] = _recurrence_next(seq[i - L:i], L, taps)
    return seq


def recover_m_sequence_from_window(window_bits: np.ndarray,
                                   L: int,
                                   taps: list | None = None) -> np.ndarray:
    """对外接口：由任意 L 位连续窗口重建整条 m 序列（与 generate 自洽）。"""
    if taps is None:
        taps = PRIMITIVE_POLYNOMIAL[L]
    return extend_m_sequence(window_bits, L, taps)


def make_key(watermark_bits: np.ndarray, m_seq: np.ndarray) -> np.ndarray:
    """K = W xor M（论文公式：watermarking key）。"""
    return np.bitwise_xor(watermark_bits, m_seq)


if __name__ == "__main__":
    # 自检：对每个支持的 L 验证“生成 -> 任取窗口 -> 重建”闭环
    for L, taps in PRIMITIVE_POLYNOMIAL.items():
        assert verify_m_sequence(L, taps), f"L={L} not maximal"
        M = generate_m_sequence(L, taps)
        n = (1 << L) - 1
        # 随机取一个 L 位窗口（含跨越周期边界的循环窗口）
        for _ in range(5):
            start = int(np.random.randint(0, n))
            win = np.concatenate([M[start:], M[: max(0, L - (n - start))]])[:L]
            rec = recover_m_sequence_from_window(win, L, taps)
            # 重建序列应等于 M 循环左移 start 位
            assert np.array_equal(rec, np.roll(M, -start)), f"L={L} mismatch"
    print("m-sequence self-check OK")
