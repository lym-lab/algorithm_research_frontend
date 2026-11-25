# -*- coding: utf-8 -*-
"""鲁棒性评估：对一组宿主音频跑“攻击矩阵”，输出 BER/SNR 汇总表。

攻击矩阵（对齐论文表 III / VII / IX / XI / XIII / XV）：
  CSP : gna / mp3c / am / rs / lpf / mf / rq / cp
  TSM/PSM : 0.8 / 0.9 / 1.1 / 1.2
  LSC : 20% / 50% / 70% / 80% / 90%
  重录: 0.5 / 1 / 2 / 3 / 5 / 10 / 13 m（数字信道模型）
  混合: recapture(1m) + {ecsp, tsm, psm}

口径说明（重要，避免把“恢复后 BER=0”误读为鲁棒性证据）：
  - ber_raw_mean_pct：模型逐段分类的原始误码率，是“攻击是否被抗住”的真实
    指标，应作为主结论；
  - oracle_hit_rate：现有 recover_watermark 借用真值水印 W 找 L 位锚点的命中
    比例（先知上界，接收方实际拿不到真值 W）；
  - oracle_ber_hit_pct：仅在命中样本上统计（构造性重建≈0）；未命中样本不计入，
    失败不再被静默平均成 0；
  - snr_embed_db：嵌入保真度（式 7，原始宿主 vs 含水印音频），与攻击无关，
    不代表攻击后 SNR。
真实（不掌握真值 W 的）盲恢复不在本运行包内；本表只报告 oracle 上界与 raw BER。

用法：
    python scripts/run_attack_eval.py                       # 默认攻击矩阵
    python scripts/run_attack_eval.py --quick               # 快速子集
    python scripts/run_attack_eval.py --attacks mp3c rs lsc
"""
import argparse  # 命令行参数解析
import csv       # 结果表写 CSV
import json      # 结果表写 JSON
import sys       # 注入工程根到 import 路径
from pathlib import Path  # 路径处理

# 把工程根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # 统计均值

# 引入核心函数
from lraw import (Config, DEFAULT_CONFIG, read_audio, run_pipeline,
                  synth_music, synth_speech_like, apply_attack, hybrid_attack,
                  metrics, embed, extract_bits, recover_watermark, load_model,
                  align_length, snr, ber)

ROOT = Path(__file__).resolve().parent.parent  # 工程根目录
OUT = ROOT / "output"                           # 输出目录


def make_hosts(cfg, n_hosts: int, sr: int, real_only: bool = False,
               max_real: int = 0):
    """宿主池：真实 wav（data/*.wav）+ 可选合成类音乐/类语音。

    real_only=True 时只用真实宿主，用于“论文口径”的宿主对照实验；
    否则合成宿主照旧生成，真实宿主追加在后面（默认行为保持不变）。
    max_real=0 表示不限制真实宿主条数（此前硬编码上限 8 条，装不下
    10 首 FMA + 10 条语音，故改为参数）。
    """
    hosts = []
    if not real_only:
        # 每种合成宿主生成 n_hosts 条，种子固定保证可复现
        for i in range(n_hosts):
            hosts.append(("synth_music", synth_music(16.0 + 2.0 * i, sr, seed=100 + i)))
            hosts.append(("synth_speech", synth_speech_like(12.0, sr, seed=200 + i)))
    data_dir = ROOT / "data"
    # 接入 data/ 根目录下规范化好的真实 wav（music_*/speech_*）
    if data_dir.exists():
        files = sorted(data_dir.glob("*.wav"))
        if max_real and max_real > 0:
            files = files[:max_real]
        for p in files:
            x, s = read_audio(str(p))
            hosts.append((p.stem, x))
    return hosts


def build_attack_matrix(quick: bool = False):
    """构造攻击矩阵：返回 [(展示名, 攻击类型, 参数), ...]。quick 时缩小档位。"""
    matrix = [
        # 8 种常见信号处理攻击（CSP）
        ("gna", "gna", {}),
        ("mp3c", "mp3c", {}),
        ("am", "am", {}),
        ("rs", "rs", {}),
        ("lpf", "lpf", {}),
        ("mf", "mf", {}),
        ("rq", "rq", {}),
        ("cp", "cp", {}),
    ]
    # 时间拉伸 / 相位声码器变速，4 个因子（quick 仅 0.9）
    for f in ([0.8, 0.9, 1.1, 1.2] if not quick else [0.9]):
        matrix.append((f"tsm_{f}", "tsm", {"factor": f}))
        matrix.append((f"psm_{f}", "psm", {"factor": f}))
    # 大比例局部裁剪，5 个比例（quick 仅 0.5）
    for r in ([0.2, 0.5, 0.7, 0.8, 0.9] if not quick else [0.5]):
        matrix.append((f"lsc_{r}", "lsc", {"ratio": r}))
    # 数字重录信道模型，7 个距离（quick 仅 1/5 m）
    for d in ([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 13.0] if not quick else [1.0, 5.0]):
        matrix.append((f"recapture_{d}m", "recapture", {"distance_m": d}))
    # 混合攻击（论文表 XV：recapture@1m + ECSP/TSM/PSM）
    matrix.append(("hybrid_recap1m+ecsp", "hybrid", {"base": "recapture",
                                                      "second": "ecsp"}))
    matrix.append(("hybrid_recap1m+tsm", "hybrid", {"base": "recapture",
                                                     "second": "tsm"}))
    matrix.append(("hybrid_recap1m+psm", "hybrid", {"base": "recapture",
                                                     "second": "psm"}))
    return matrix


def run_one(emb, sr, attack_name, atk_type, attack_cfg, cfg, model):
    """emb 为预计算好的嵌入结果（宿主只嵌入一次，跨攻击复用，避免重复 DWT/SVD）。"""
    wm = emb["audio_wm"]
    if attack_cfg.get("base"):
        # 混合攻击：先重录再叠加（ECSP 表示把 8 种 CSP 全部串行施加）
        if attack_cfg["second"] == "ecsp":
            y = apply_attack(attack_cfg["base"], wm, sr, cfg,
                             **attack_cfg.get("kwargs", {}))
            for n2 in ["gna", "mp3c", "am", "rs", "lpf", "mf", "rq", "cp"]:
                y = apply_attack(n2, y, sr, cfg)
        else:
            y = hybrid_attack(wm, sr, base=attack_cfg["base"],
                              second=attack_cfg["second"], cfg=cfg,
                              **attack_cfg.get("kwargs", {}))
    else:
        y = apply_attack(atk_type, wm, sr, cfg, **attack_cfg)

    # ---- 时基归一的门限：只有“整体变速/时钟漂移”类攻击才允许把音频
    # 重采样回已知时长；裁剪类（lsc）是真实丢内容，归一反而会破坏索引对齐，
    # 所以仍用默认 3% 小门限（它对裁剪无效，因为裁剪的时长偏差远大于 3%）。
    tempo_attacks = {"tsm", "recapture"}
    if atk_type == "hybrid":
        tempo_attacks_case = attack_cfg.get("second") in ("tsm", "psm", "ecsp")
    else:
        tempo_attacks_case = atk_type in tempo_attacks
    tol = 0.25 if tempo_attacks_case else 0.03

    ext = extract_bits(y, model, cfg,
                       expected_len=emb["audio_wm"].size,
                       tol=tol,
                       repeat=emb.get("repeat", 1))
    # 变时长攻击（TSM 等）改变段数：两种对齐候选——
    #   1) 索引对齐截断（适合 CP/LSC 这类“局部删除”，剩余部分仍索引对齐）
    #   2) 概率序列长度归一化（适合 TSM/PSM 这类“整体拉伸”，需插值回 n 位）
    # 取恢复成功者；raw BER 使用恢复成功对应的候选序列。
    cands = [ext["bits"]]
    if ext["n"] != emb["n"]:
        cands.append(align_length(ext["probs"], emb["n"]))
    rec = None
    for cand in cands:
        r = recover_watermark(emb["bits"], cand, emb["key"], emb["L"],
                              emb["taps"])
        if r["anchor"] is not None:
            rec = r
            Wt = cand
            break
    if rec is None:
        rec = recover_watermark(emb["bits"], cands[0], emb["key"], emb["L"],
                                emb["taps"])
        Wt = cands[0]
    hit = bool(rec["anchor"] is not None)
    return {
        "ber_raw": ber(emb["bits"], Wt),
        "oracle_hit": hit,
        # 仅命中样本才有恢复后 BER；未命中记 None（汇总时不计入均值）
        "ber_recovered": rec["ber_after"] if hit else None,
        "snr_embed": emb["snr_voiced"],
        "n": emb["n"],
    }


def main():
    ap = argparse.ArgumentParser(description="攻击矩阵鲁棒性评估")
    ap.add_argument("--quick", action="store_true", help="快速子集档位")
    ap.add_argument("--n-hosts", type=int, default=3, help="每类合成宿主数量")
    ap.add_argument("--real-only", action="store_true",
                    help="只用 data/*.wav 真实宿主（不加合成宿主）")
    ap.add_argument("--max-real", type=int, default=0,
                    help="真实宿主条数上限，0 表示全部")
    ap.add_argument("--out-tag", default="",
                    help="结果文件名后缀，用于保留多组实验（如 real20）")
    ap.add_argument("--attacks", nargs="*", default=None,
                    help="只跑指定攻击名（覆盖默认矩阵）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每比特重复写入的段数 k（默认 1=论文口径；k>1 时容量降为 1/k）")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    model = load_model()  # 按优先级加载主链路分类器
    matrix = build_attack_matrix(args.quick)
    if args.attacks:      # 只保留命令行指定的攻击
        matrix = [m for m in matrix if m[0] in args.attacks]

    hosts = make_hosts(cfg, args.n_hosts, cfg.sr,
                       real_only=args.real_only, max_real=args.max_real)
    print(f"hosts: {len(hosts)}   attacks: {len(matrix)}   repeat={args.repeat} "
          f"(宿主先嵌入一次，跨攻击复用，避免重复 DWT/SVD)")
    # 每个宿主先嵌入一次，得到可跨攻击复用的嵌入结果
    host_embs = [(hn, embed(h, None, cfg, repeat=args.repeat)) for hn, h in hosts]

    rows = []
    for name, atk, kw in matrix:
        stats = {"ber_raw": [], "oracle_hit": [], "ber_recovered": [],
                 "snr_embed": []}
        for hname, emb in host_embs:
            try:
                r = run_one(emb, cfg.sr, name, atk, kw, cfg, model)
                stats["ber_raw"].append(r["ber_raw"])
                stats["oracle_hit"].append(r["oracle_hit"])
                stats["ber_recovered"].append(r["ber_recovered"])
                stats["snr_embed"].append(r["snr_embed"])
            except Exception as e:
                print(f"[skip] {name}/{hname}: {e}")

        def _mean(v):
            v = [x for x in v if x is not None and x == x]  # 剔 None 与 NaN
            return float(np.mean(v)) if v else None

        def _f(x, spec="6.2f"):
            # 空值（该攻击全部宿主被跳过，如缺 lameenc 导致 mp3c 不可用）显示 n/a，
            # 禁止对 None 使用数值格式符导致整个评估崩溃。
            return f"{x:{spec}}" if isinstance(x, (int, float)) and x == x else " n/a "

        n_hosts = len(stats["ber_raw"])
        mean_raw = _mean(stats["ber_raw"])
        mean_rec = _mean(stats["ber_recovered"])
        mean_snr = _mean(stats["snr_embed"])
        hit_rate = (float(np.mean(stats["oracle_hit"]))
                    if stats["oracle_hit"] else None)
        row = {
            "attack": name,
            "n_hosts": n_hosts,
            "ber_raw_mean_pct": (mean_raw * 100.0 if mean_raw is not None else None),
            "oracle_hit_rate": hit_rate,
            # 命中样本上的恢复后 BER（构造性≈0，仅先知上界参考）；无命中为空
            "oracle_ber_hit_pct": (mean_rec * 100.0 if mean_rec is not None else None),
            "snr_embed_db": mean_snr,
        }
        rows.append(row)
        hit_n = int(np.sum(stats["oracle_hit"]))
        note = "   [全部宿主跳过：依赖缺失或异常，详见上方 skip 提示]" if n_hosts == 0 else ""
        print(f"{name:<22} BER_raw={_f(row['ber_raw_mean_pct'])}% "
              f"oracle_hit={hit_n}/{n_hosts} "
              f"BER_hit={_f(row['oracle_ber_hit_pct'], '5.1f')}% "
              f"SNR_embed={_f(row['snr_embed_db'])}dB{note}")
        # ---- 每跑完一档就把结果落盘一次 ----
        # 目的：矩阵要跑二十几分钟，中途被中断（Ctrl-C / 抢 CPU）时不必从头再来，
        # 也让外部可以用 Get-Content 监控进度。
        OUT.mkdir(exist_ok=True)
        tag_now = f"_{args.out_tag}" if args.out_tag else ""
        with open(OUT / f"attack_matrix_results{tag_now}.csv", "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    OUT.mkdir(exist_ok=True)
    # 文件名可按 --out-tag 加后缀，便于同时保留合成宿主/真实宿主两组结果
    tag = f"_{args.out_tag}" if args.out_tag else ""
    csv_path = OUT / f"attack_matrix_results{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 写 JSON 备份
    json_path = OUT / f"attack_matrix_results{tag}.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")


# 脚本直接运行时执行 main()
if __name__ == "__main__":
    main()
