# -*- coding: utf-8 -*-
"""指标计算工具：对 (原始, 水印/攻击后) 音频对与 (原水印, 提取水印)
计算 SNR / BER /（可选）ODG。

用法：
    python scripts/calc_metrics.py --orig a.wav --proc b.wav --wm w.npy --ext e.npy
"""
import argparse  # 命令行参数解析
import json      # 结果以 JSON 输出
import sys       # 注入工程根到 import 路径
from pathlib import Path  # 路径处理

# 把工程根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # 读取比特数组

# 引入音频读取与指标函数
from lraw import read_audio, snr, ber, odg


def main():
    ap = argparse.ArgumentParser(description="计算 SNR / BER / ODG")
    ap.add_argument("--orig", required=True, help="原始音频 wav")
    ap.add_argument("--proc", required=True, help="处理后音频 wav（水印/攻击后）")
    ap.add_argument("--wm", help="原水印比特 npy")
    ap.add_argument("--ext", help="提取水印比特 npy")
    ap.add_argument("--odg", action="store_true", help="计算 ODG（需 peaqb）")
    args = ap.parse_args()

    x, sr = read_audio(args.orig)  # 原始波形与采样率
    y, _ = read_audio(args.proc)   # 处理后波形（采样率忽略，按 x 对齐）
    out = {"snr_db": snr(x, y)}    # 先算信噪比 SNR(dB)
    if args.odg:
        out["odg"] = odg(x, y, sr)  # 可选 PEAQ ODG 客观音质评分
    if args.wm and args.ext:
        w = np.load(args.wm)  # 原始水印比特
        e = np.load(args.ext) # 提取水印比特
        out["ber"] = ber(w, e)          # 误码率 BER
        out["accuracy"] = 1.0 - out["ber"]  # 比特正确率 = 1 - BER
    print(json.dumps(out, ensure_ascii=False, indent=2))


# 脚本直接运行时执行 main()
if __name__ == "__main__":
    main()
