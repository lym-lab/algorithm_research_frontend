# -*- coding: utf-8 -*-
"""DWT-GBT-SVD 系数提取（论文 III-B 节 + 公式 5）。

流程（论文 4215 页 B 节）：
  1. 对音频段做两层 DWT，取第 2 层近似系数 appr（长度 = 段长/4）；
  2. 构造图拉普拉斯，取其正交特征向量 E，做 GBT：spar = appr·E^T，
     提升去相关；
  3. 把 spar 整形为 5x20 矩阵 Y；
  4. 对 Y 做 SVD：Y = U·S·V^T，得到降序排列的 5 个奇异值
     S = {s1, s2, s3, s4, s5}（DWT-GBT-SVD 系数）。

论文未指明图结构与 DWT 小波基：本项目默认用路径图（path graph，
其拉普拉斯特征向量近似 DCT 基）与 db4 小波，二者均可配置做消融。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt


def _path_graph_eigenbasis(n: int) -> np.ndarray:
    """路径图（相邻节点相连、边权为 1）拉普拉斯 L = D - A 的特征向量。

    返回 E（nxn）：第 i 行为第 i 个特征向量（按特征值升序），已正交归一。
    路径图拉普拉斯特征向量在数学上即 DCT-II 型基函数。
    """
    lap = np.zeros((n, n))
    for i in range(n):
        lap[i, i] = 1
        if i > 0:
            lap[i, i - 1] = -1
        if i < n - 1:
            lap[i, i + 1] = -1
    vals, vecs = np.linalg.eigh(lap)          # vecs 列为特征向量
    return vecs.T                             # E 以特征向量为行


def _sing_val_norm(s: np.ndarray, mode: str = "max") -> np.ndarray:
    """奇异值归一化（与训练数据集一致；默认按 s1（最大奇异值）归一化）。

    论文 4219 页：“The five singular values of each segment are then
    normalized”。作者开源 dataset.xlsx 中的特征列表现为 s1=1，
    故默认 mode='max'（除以最大奇异值，保留降序结构）。
    """
    s = np.asarray(s, dtype=np.float64)
    if mode == "max":
        if s[0] > 1e-12:
            return s / s[0]
        return s
    if mode == "sum":
        total = s.sum()
        return s / total if total > 0 else s
    if mode == "minmax":
        lo, hi = s[-1], s[0]
        if hi - lo > 1e-12:
            return (s - lo) / (hi - lo)
        return s
    if mode == "none":
        return s
    raise ValueError(f"unknown norm mode: {mode}")


@dataclass
class SegmentFeatures:
    """单个音频段提取的全部中间量（用于嵌入/可视化/调试）。"""
    appr: np.ndarray            # 两层 DWT 后的近似系数（长度 = 段长/4）
    E: np.ndarray               # GBT 正交基（n x n）
    spar: np.ndarray            # GBT 稀疏系数（长度 n）
    Y: np.ndarray               # 5x20 矩阵
    U: np.ndarray
    S: np.ndarray               # 5 个降序奇异值
    Vt: np.ndarray
    singular_values: np.ndarray = None   # 归一化后的特征（模型输入）
    segment_len: int = 0

    def __post_init__(self):
        if self.singular_values is None:
            self.singular_values = self.S


def extract_features(segment: np.ndarray,
                     wavelet: str = "db4",
                     dwt_levels: int = 2,
                     svd_rows: int = 5,
                     svd_cols: int = 20,
                     norm_mode: str = "max") -> SegmentFeatures:
    """对单个音频段提取 DWT-GBT-SVD 系数。

    要求 len(segment) / 2**dwt_levels == svd_rows * svd_cols（默认 400/4=100）。
    """
    seg = np.asarray(segment, dtype=np.float64).ravel()
    expected = svd_rows * svd_cols * (2 ** dwt_levels)
    if seg.size != expected:
        raise ValueError(
            f"segment length {seg.size} != expected {expected} "
            f"({svd_rows}x{svd_cols} * 2**{dwt_levels}); "
            "请检查分段长度是否为 400（训练数据）或 n 等分结果。"
        )

    # 步骤1：两层离散小波变换（DWT）。
    #   periodization（周期延拓）模式保证每层输出长度精确为上一层的 1/2，
    #   两层后近似系数 cA2 长度 = 400 / 2 / 2 = 100，正好等于 5×20。
    #   coeffs[0]=cA2（低频近似，承载主要能量），coeffs[1:]=各层细节系数（逆变换时原样保留）。
    coeffs = pywt.wavedec(seg, wavelet, level=dwt_levels, mode="periodization")
    appr = np.asarray(coeffs[0], dtype=np.float64)          # cA2，长度 = seg/4 = 100
    assert appr.size == svd_rows * svd_cols

    # 步骤2：图小波变换 GBT（Graph-based transform）去相关。
    #   E 是路径图拉普拉斯的正交特征向量矩阵（行向量、E·Eᵀ=I）；
    #   spar = appr · Eᵀ 把时域近似系数投影到图频域，得到更稀疏、更平稳的系数，
    #   便于后续 SVD 奇异值稳定可分。
    E = _path_graph_eigenbasis(appr.size)
    spar = appr @ E.T

    # 步骤3：把 100 个稀疏系数按行优先整形为 5 行 × 20 列的二维矩阵 Y，
    #   作为 SVD 的输入（论文 DWT-GBT-SVD 特征的几何载体）。
    Y = spar.reshape(svd_rows, svd_cols)

    # 步骤4：对 Y 做薄 SVD（full_matrices=False）：Y = U · diag(s) · Vᵀ。
    #   np.linalg.svd 的奇异值 s 自动按降序排列 s1≥s2≥s3≥s4≥s5；
    #   嵌入侧只修改第 3 个奇异值 s3，提取侧 LightGBM 用这 5 个归一化奇异值判 0/1。
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s = np.asarray(s, dtype=np.float64)

    feats = SegmentFeatures(
        appr=appr, E=E, spar=spar, Y=Y, U=U, S=s, Vt=Vt,
        singular_values=_sing_val_norm(s, norm_mode),
        segment_len=seg.size,
    )
    return feats


def reconstruct_with_modified_s3(segment: np.ndarray,
                                 s3_tilde: float,
                                 wavelet: str = "db4",
                                 dwt_levels: int = 2,
                                 svd_rows: int = 5,
                                 svd_cols: int = 20) -> tuple:
    """快捷入口：对原始段做特征提取 -> 替换 s3 -> 重建时域段。

    返回 (重建后的音频段, 特征对象)。与 extract_features 使用同一套参数，
    保证嵌入/提取两侧的特征完全对齐。
    """
    # 正向提取一次，拿到 U / Vt / E 等分解分量，用于“只改 s3 再重建”。
    feats = extract_features(segment, wavelet, dwt_levels, svd_rows, svd_cols)
    s_mod = feats.S.copy()
    s_mod[2] = s3_tilde                       # 仅替换第 3 个奇异值（索引 2），其余 s1/s2/s4/s5 不动
    # SVD 逆合成：Ỹ = U · diag(s̃) · Vᵀ，得到修改后的 5×20 矩阵
    Y_tilde = feats.U @ np.diag(s_mod) @ feats.Vt
    spar_tilde = Y_tilde.reshape(-1)           # 拍平回 100 个图频域系数
    # IGBT（逆图小波变换）：提取侧 spar = appr·Eᵀ；因 E 正交 E⁻¹=Eᵀ，
    # 故 appr̃ = spar̃·E，即可还原近似系数。
    appr_tilde = spar_tilde @ feats.E
    # 细节系数直接复用原始段分解结果（嵌入只改近似系数，不破坏高频细节）
    details = pywt.wavedec(segment, wavelet, level=dwt_levels, mode="periodization")[1:]
    # IDWT：用修改后的 cA2 加原始细节系数重建时域段
    seg_out = pywt.waverec([appr_tilde] + list(details), wavelet, mode="periodization")
    return seg_out, feats
