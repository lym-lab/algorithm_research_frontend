# -*- coding: utf-8 -*-
"""端到端演示：嵌入 ->（可选攻击）-> 提取 -> 恢复 -> 指标。

用法示例：
    python scripts/demo_embed_extract.py                          # 无攻击
    python scripts/demo_embed_extract.py --attack mp3c            # MP3 攻击
    python scripts/demo_embed_extract.py --attack recapture --distance 3.0
    python scripts/demo_embed_extract.py --hybrid recapture tsm
    python scripts/demo_embed_extract.py --audio data/xxx.wav     # 用真实音频
"""
import argparse  # 命令行参数解析
import json      # 结果元信息落盘 JSON
import sys       # 注入工程根到 import 路径
from pathlib import Path  # 路径处理

# 把工程根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # （供 pipeline 内部使用）

# 引入配置、音频读写、端到端流水线、合成宿主
from lraw import (Config, DEFAULT_CONFIG, read_audio, write_audio,
                  run_pipeline, synth_music, synth_speech_like)

# 输出目录：工程根 / output
OUT = Path(__file__).resolve().parent.parent / "output"


def main():
    ap = argparse.ArgumentParser(description="LRAW end-to-end demo")
    ap.add_argument("--audio", type=str, default=None, help="宿主音频路径（默认合成音乐）")
    ap.add_argument("--speech", action="store_true", help="用合成类语音做宿主")
    ap.add_argument("--duration", type=float, default=12.0, help="宿主时长(s)")
    ap.add_argument("--n", type=int, default=None, help="覆盖水印位数 n（默认由段数决定）")
    ap.add_argument("--attack", type=str, default=None, help="单攻击名")
    ap.add_argument("--distance", type=float, default=None, help="重录距离（m）")
    ap.add_argument("--hybrid", nargs=2, metavar=("BASE", "SECOND"), default=None,
                    help="两级混合攻击：基础攻击 + 第二级攻击")
    ap.add_argument("--model", type=str, default="assets/author_trained.pkl",
                    help="分类器路径（默认 assets/author_trained.pkl）")
    ap.add_argument("--save-wav", action="store_true", help="保存水印/攻击后 wav")
    args = ap.parse_args()

    import os
    # 通过环境变量指定本次使用的分类器（model.load_model 读取）
    os.environ["LRAW_MODEL_PATH"] = str(Path(args.model).resolve())

    cfg = DEFAULT_CONFIG  # 全局配置
    if args.n is not None:
        cfg = Config(n_segments=args.n)  # 覆盖水印段数

    # ---- 宿主音频选择：真实文件 > 合成语音 > 合成音乐 ----
    if args.audio:
        audio, sr = read_audio(args.audio)
        print(f"[host] {args.audio} sr={sr} len={audio.size/ sr:.1f}s")
        if sr != cfg.sr:
            print(f"[warn] 宿主采样率 {sr} != 配置 {cfg.sr}，已按 {sr} 处理")
            cfg.sr = sr
    elif args.speech:
        audio = synth_speech_like(args.duration, cfg.sr)
    else:
        audio = synth_music(args.duration, cfg.sr)

    attack_kw = {}
    if args.distance is not None:
        attack_kw["distance_m"] = args.distance  # 重录距离参数

    # 跑完整流水线：嵌入 -> 攻击 -> 特征提取 -> 分类 -> 恢复 -> 指标
    res = run_pipeline(audio, watermark_bits=None,
                       attack=args.attack, attack_kw=attack_kw,
                       hybrid=tuple(args.hybrid) if args.hybrid else None,
                       cfg=cfg)

    # 打印关键指标
    print("=" * 60)
    print(f"attack            : {res['attack']}")
    print(f"watermark length n: {res['n']}   (m-sequence L={res['L']})")
    print(f"SNR (voiced)      : {res['snr_voiced']:.2f} dB")
    print(f"extracted raw BER : {res['extracted_raw_ber']*100:.2f} %")
    if res["recovered_ber"] is not None:
        print(f"recovered BER     : {res['recovered_ber']*100:.2f} %   "
              f"(anchor @ {res['anchor']})")
    else:
        print("recovered BER     : 恢复失败（未找到 L 位一致窗口）")
    print("=" * 60)

    # 组装落盘元信息：去掉大数组/波形/密钥，单独把比特列表转成普通 list
    result_meta = {k: v for k, v in res.items()
                   if k not in ("bits", "extracted_bits", "recovered_bits",
                                "wm_audio", "attacked_audio", "key", "taps")}
    result_meta["bits_list"] = res["bits"].tolist()
    result_meta["extracted_list"] = res["extracted_bits"].tolist()
    result_meta["recovered_list"] = (None if res["recovered_bits"] is None
                                     else res["recovered_bits"].tolist())

    OUT.mkdir(exist_ok=True)
    # 输出文件名：demo[_attack-xxx][_hybrid-a+b]
    name = "demo" + (f"_attack-{args.attack}" if args.attack else "") + \
           (f"_hybrid-{args.hybrid[0]}+{args.hybrid[1]}" if args.hybrid else "")
    meta_path = OUT / f"{name}_meta.json"
    meta_path.write_text(json.dumps(result_meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[saved] {meta_path}")

    # 可选：保存含水印音频与攻击后音频
    if args.save_wav:
        write_audio(OUT / f"{name}_watermarked.wav", res["wm_audio"], cfg.sr)
        if res["attack"] != "none":
            write_audio(OUT / f"{name}_attacked.wav", res["attacked_audio"], cfg.sr)
        print(f"[saved] {OUT / name}_watermarked.wav")


# 脚本直接运行时执行 main()
if __name__ == "__main__":
    main()
