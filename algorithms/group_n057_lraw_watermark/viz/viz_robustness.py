# -*- coding: utf-8 -*-
"""鲁棒性可视化（匹配新版 attack_matrix_results.csv 口径）。

口径（重要）：
  - ber_raw_mean_pct：LightGBM 逐段原始误码率，主指标；
  - oracle_hit_rate：借真值 W 的先知上界恢复命中率（0~1），不等于盲恢复；
    真实（不持真值 W）盲恢复结论不在本运行包内，需另行做诊断；
  - oracle_ber_hit_pct：仅命中样本、构造性≈0，不作为鲁棒性主指标。
本脚本据此画：
  fig_robustness_bar.png       各攻击 raw BER + oracle 命中率（次轴）
  fig_robustness_distance.png 数字重录距离 vs raw BER / 命中率
  fig_recovery_effect.png     raw BER 与恢复命中率关系（显示失效边界）

用法：先运行
    python scripts/run_attack_eval.py
    python viz/viz_robustness.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "output" / "figs"
CSV = ROOT / "output" / "attack_matrix_results_L4.csv"
if not CSV.exists():          # 回退：L4 最终矩阵没跑时用历史版本
    CSV = ROOT / "output" / "attack_matrix_results.csv"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    rows = load()
    if not rows:
        print("no results; run scripts/run_attack_eval.py first")
        return
    FIG.mkdir(parents=True, exist_ok=True)

    names = [r["attack"] for r in rows]
    raw = [_f(r["ber_raw_mean_pct"]) or 0.0 for r in rows]
    hit = [(_f(r.get("oracle_hit_rate")) or 0.0) * 100.0 for r in rows]

    # ---- (a) raw BER 柱 + oracle 命中率（次轴折线点）----
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = range(len(names))
    colors = ["#059669" if v < 20 else "#dc2626" for v in raw]
    ax.bar(x, raw, width=0.62, color=colors, alpha=0.85,
           label="raw BER (%) — main metric")
    ax.axhline(20, color="#dc2626", ls="--", lw=1, label="20% effective threshold")
    ax.set_ylabel("raw BER (%)")
    ax.set_ylim(0, 60)
    ax2 = ax.twinx()
    ax2.plot(x, hit, "o-", color="#1d4ed8", ms=5, lw=1.4,
             label="oracle recovery hit rate (%) — upper bound only")
    ax2.set_ylabel("oracle hit rate (%)")
    ax2.set_ylim(-5, 105)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Attack matrix: raw BER (main) vs oracle hit rate "
                 "(synthetic hosts, digital recapture model)")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_robustness_bar.png", dpi=160)
    print("[saved]", FIG / "fig_robustness_bar.png")

    # ---- (b) 重录距离 vs raw BER / 命中率 ----
    dist_rows = [r for r in rows if r["attack"].startswith("recapture_")]
    if dist_rows:
        dists = [float(r["attack"].split("_")[1].rstrip("m")) for r in dist_rows]
        order = sorted(range(len(dists)), key=lambda i: dists[i])
        d = [dists[i] for i in order]
        dr = [_f(dist_rows[i]["ber_raw_mean_pct"]) or 0.0 for i in order]
        dh = [(_f(dist_rows[i].get("oracle_hit_rate")) or 0.0) * 100.0
              for i in order]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(d, dr, "o-", color="#dc2626", lw=2, ms=6, label="raw BER (%)")
        ax.axhline(20, color="#999999", ls="--", lw=1)
        ax.set_xlabel("Recapture distance (m, simulated)")
        ax.set_ylabel("raw BER (%)", color="#dc2626")
        ax.set_ylim(0, 60)
        ax2 = ax.twinx()
        ax2.plot(d, dh, "s--", color="#1d4ed8", lw=1.6, ms=6,
                 label="oracle hit rate (%)")
        ax2.set_ylabel("oracle hit rate (%)", color="#1d4ed8")
        ax2.set_ylim(-5, 105)
        ax.set_title("Simulated recapture: raw BER rises with difficulty")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "fig_robustness_distance.png", dpi=160)
        print("[saved]", FIG / "fig_robustness_distance.png")

    # ---- (c) raw BER vs 恢复命中率：显示失效边界 ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(raw, hit, c="#2563eb", s=45, zorder=3)
    ax.axvline(20, color="#dc2626", ls="--", lw=1, label="raw BER=20%")
    ax.axhline(50, color="#999999", ls=":", lw=1)
    ax.set_xlabel("raw BER before recovery (%)")
    ax.set_ylabel("oracle recovery hit rate (%)")
    ax.set_title("Recovery feasible only when raw BER stays low and phase "
                 "alignable; high raw BER -> lock fails")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_recovery_effect.png", dpi=160)
    print("[saved]", FIG / "fig_recovery_effect.png")


if __name__ == "__main__":
    main()
