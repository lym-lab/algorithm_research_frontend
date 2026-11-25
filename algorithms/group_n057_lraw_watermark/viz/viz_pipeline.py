# -*- coding: utf-8 -*-
"""图 1：LRAW 方法架构流程图（嵌入 / 提取恢复 / 模型训练）。

输出 output/figs/fig_pipeline.png，可直接用于报告“方法理解”一节。
本版只保留有明确上下游关系的箭头：同一行内从左到右，训练模型用一条
竖直虚线指向提取阶段的 LightGBM；不再画跨行斜线或指向空处的箭头。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for _f in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

FIG = Path(__file__).resolve().parent.parent / "output" / "figs"


def box(ax, x, y, w, h, text, fc, ec="#4b5563", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.8",
        fc=fc, ec=ec, lw=1.0
    ))
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fs, fontweight="bold" if bold else "normal",
        color="#111827", linespacing=1.35
    )


def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.2, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        lw=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=1.5, shrinkB=1.5,
        connectionstyle=f"arc3,rad={rad}"
    ))


def main():
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    # 标题
    ax.text(50, 62.4, "LRAW 方法架构：嵌入、提取恢复与模型训练",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")
    ax.text(50, 59.9, "LightGBM-based Robust Audio Watermarking · IEEE TIFS 2025, Vol.20",
            ha="center", va="center", fontsize=9.5, color="#6b7280")

    # 配色
    C_HOST = "#dbeafe"      # 宿主 / 含水印音频
    C_PROC = "#f3f4f6"      # 通用处理
    C_FEAT = "#e0f2fe"      # 特征
    C_QUANT = "#fffbeb"     # 量化
    C_ATTACK = "#fee2e2"    # 攻击音频 / 提取域
    C_LGBM = "#fca5a5"      # LightGBM
    C_REC = "#dcfce7"       # 恢复
    C_KEY = "#ede9fe"       # 密钥
    C_DATA = "#ede9fe"      # 数据集
    C_TRAIN = "#ddd6fe"     # 训练

    # 分隔线
    ax.plot([2, 98], [37.0, 37.0], color="#d1d5db", lw=0.8)
    ax.plot([2, 98], [16.5, 16.5], color="#d1d5db", lw=0.8)

    # ===================== ① 水印嵌入 =====================
    ax.text(3, 55.4, "① 水印嵌入（Embedding）", fontsize=11.5, fontweight="bold", color="#1d4ed8")
    y1, h1 = 48.0, 4.6

    box(ax, 3, y1, 13, h1, "宿主音频 A", C_HOST, fs=10)
    arrow(ax, 16, y1 + h1 / 2, 19, y1 + h1 / 2)
    box(ax, 19, y1, 14, h1, "固定网格分段\n400 samples/段", C_PROC, fs=8.8)
    arrow(ax, 33, y1 + h1 / 2, 36, y1 + h1 / 2)
    box(ax, 36, y1, 17, h1, "DWT-GBT-SVD\nS={s1..s5} 降序", C_FEAT, fs=8.8)
    arrow(ax, 53, y1 + h1 / 2, 56, y1 + h1 / 2)
    box(ax, 56, y1, 19, h1, "s3 量化\n按 W 比特选择规则", C_QUANT, fs=8.8)
    arrow(ax, 75, y1 + h1 / 2, 78, y1 + h1 / 2)
    box(ax, 78, y1, 19, h1, "SVD 重建 + IGBT/IDWT\n拼接为含水印音频", C_HOST, fs=8.4)

    # 密钥生成（独立说明行，不参与上排箭头）
    box(ax, 3, 40.0, 74, 4.6,
        "密钥生成：W(n bits) ⊕ M(LFSR, L=13) → K；K 在发送端保存\n"
        "接收端由 K 与 M 异或还原 W",
        C_KEY, fs=8.2)

    # ===================== ② 提取与恢复 =====================
    ax.text(3, 34.8, "② 提取与恢复（Extraction & Recovery）", fontsize=11.5,
            fontweight="bold", color="#b91c1c")
    y2, h2 = 27.5, 4.6

    box(ax, 3, y2, 12, h2, "攻击后音频\n重录/CSP/混合", C_ATTACK, fs=8.4)
    arrow(ax, 15, y2 + h2 / 2, 18, y2 + h2 / 2)
    box(ax, 18, y2, 12, h2, "固定网格\n分段", C_PROC, fs=8.8)
    arrow(ax, 30, y2 + h2 / 2, 33, y2 + h2 / 2)
    box(ax, 33, y2, 13, h2, "DWT-GBT-SVD\n归一化", C_ATTACK, fs=8.6)
    arrow(ax, 46, y2 + h2 / 2, 49, y2 + h2 / 2)
    box(ax, 49, y2, 18, h2, "LightGBM\n二分类：0 / 1", C_LGBM, fs=9.0, bold=True)
    arrow(ax, 67, y2 + h2 / 2, 69, y2 + h2 / 2)
    box(ax, 69, y2, 12, h2, "分段水印 W′\n（含误码）", C_ATTACK, fs=8.4)
    arrow(ax, 81, y2 + h2 / 2, 83, y2 + h2 / 2)
    box(ax, 83, y2, 14, h2, "滑窗找 L 位\n+ K → 恢复 W", C_REC, fs=8.2, bold=True)

    # 密钥 K 从下方进入恢复框，箭头向上
    box(ax, 83, 19.2, 14, 4.2, "密钥 K\n（发送端保存）", C_KEY, fs=8.6)
    arrow(ax, 90, 23.4, 90, 27.5, color="#6d28d9", lw=1.2)
    ax.text(89.4, 25.3, "K", ha="right", va="center", fontsize=8.0, color="#6d28d9")

    # 训练模型指向提取 LightGBM：竖直虚线，不产生斜穿
    arrow(ax, 58.5, 11.0, 58.5, 27.5, color="#6d28d9", lw=1.2, dashed=True)
    ax.text(59.3, 18.0, "训练好的模型", ha="left", va="center", fontsize=8.0, color="#6d28d9")

    # ===================== ③ 数据集与训练 =====================
    ax.text(3, 14.4, "③ 数据集与模型训练（论文 IV-D）", fontsize=11.5,
            fontweight="bold", color="#6d28d9")
    y3, h3 = 6.2, 5.2

    box(ax, 2, y3, 27, h3,
        "60 首 FMA × 全0/全1 水印\n× 9 种攻击\n→ 396,000 条特征样本",
        C_DATA, fs=7.3)
    arrow(ax, 29, y3 + h3 / 2, 32, y3 + h3 / 2)
    box(ax, 32, y3, 19, h3,
        "5 维归一化奇异值\n标签：嵌入比特 0/1",
        C_DATA, fs=8.2)
    arrow(ax, 51, y3 + h3 / 2, 55, y3 + h3 / 2)
    box(ax, 55, y3, 21, h3,
        "LightGBM 训练\n贝叶斯超参 + F1",
        C_TRAIN, fs=8.4)

    # 底部注释
    ax.text(3, 1.4,
            "注：嵌入与提取共享同一套 DWT-GBT-SVD 与归一化参数；图中只保留单向数据流，密钥 K 在接收端参与 XOR。",
            ha="left", va="center", fontsize=8.2, color="#6b7280")

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
