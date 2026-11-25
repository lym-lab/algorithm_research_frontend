# -*- coding: utf-8 -*-
"""复现论文 IV-D 节：用作者开源 dataset.xlsx 训练 / 验证 LightGBM。

数据口径（论文 4219 页）：
  60 首 FMA 3s 音频 x 全0/全1 水印 = 120 份含水印录音；
  9 种攻击 -> 1080 份攻击录音；按 400 样本/段 -> 396,000 条特征样本；
  训练 356,400 / 测试 39,600（train_test_split 0.1，rs=42）。

用法：
    python scripts/train_model.py                    # 用作者超参数重训
    python scripts/train_model.py --bayes            # 先贝叶斯搜索（需 bayes_opt）
    python scripts/train_model.py --subset 50000     # 抽样快速验证流程
"""
import argparse  # 命令行参数解析
import json      # 写出训练元信息 JSON
import sys       # 注入工程根到 import 路径
from pathlib import Path  # 跨平台路径

# 把工程根目录加入 sys.path，使 import lraw 可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # 数值计算

# 引入路径常量与训练/评估函数
from lraw.config import MODEL_PATH, DATASET_PATH, HYPERPARAMS_PATH
from lraw.model import train_model, evaluate


def load_dataset(input_path, subset: int | None = None):
    """读取作者 xlsx，抽取 5 个奇异值特征与标签，剔除非有限行。

    参数:
        input_path: 特征数据集 xlsx 路径
        subset: 若给定则随机抽取该数量样本（快速验证流程用）
    返回:
        X 特征矩阵 (N,5), y 标签向量 (N,), info 数据口径字典
    """
    import pandas as pd
    df = pd.read_excel(input_path)  # 读取作者特征表
    print(f"[data] {df.shape[0]} rows x {df.shape[1]} cols")
    X = df.iloc[:, 0:5].values.astype(np.float64)   # 前 5 列：5 个归一化奇异值
    y = df.iloc[:, 6].values.astype(np.int64)        # 第 7 列（索引 6）：标签 0/1
    ok = np.isfinite(X).all(axis=1)  # 标记含 NaN/Inf 的行（全零分段会产生）
    if not ok.all():
        print(f"[warn] drop {int((~ok).sum())} non-finite rows (all-zero segments)")
        X, y = X[ok], y[ok]
    if subset:  # 抽样快速跑通流程
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], subset, replace=False)
        X, y = X[idx], y[idx]
    info = {
        "input": str(input_path),
        "rows": int(X.shape[0]),
        "feature_cols": "iloc[:,0:5] (5 个归一化奇异值)",
        "label_col": "iloc[:,6]",
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "subset": subset,
    }
    return X, y, info


def read_hyperparams() -> dict:
    """解析作者 hyper_parameters.txt 为参数字典（int/float/str 自动转换）。"""
    text = HYPERPARAMS_PATH.read_text(encoding="utf-8")
    params = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",")  # 去空白与尾部逗号
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)  # 按第一个 '=' 拆成键值
        k = k.strip()
        v = v.strip()
        try:
            params[k] = int(v)       # 优先按整数解析
        except ValueError:
            try:
                params[k] = float(v) # 其次浮点
            except ValueError:
                params[k] = v        # 都失败则保留字符串
    return params


def main():
    ap = argparse.ArgumentParser(description="用作者数据集训练 LightGBM")
    ap.add_argument("--subset", type=int, default=None, help="抽样样本数（快速验证）")
    ap.add_argument("--bayes", action="store_true", help="启用贝叶斯超参搜索")
    ap.add_argument("--out", type=str, default=str(MODEL_PATH), help="模型保存路径")
    ap.add_argument("--input", type=str, default=str(DATASET_PATH),
                    help="特征数据集 xlsx（默认作者 dataset.xlsx）")
    args = ap.parse_args()

    X, y, dinfo = load_dataset(args.input, args.subset)  # 读数据
    params = read_hyperparams()                           # 读作者超参
    print("[params]", params)

    # 训练并切分训练/验证集；bayes_opt 控制是否先做贝叶斯搜索
    model, (Xtr, Xva, ytr, yva) = train_model(
        X, y, params=params, bayes_opt=args.bayes)

    # 在验证集上计算 accuracy/precision/recall/F1
    res = evaluate(model, Xva, yva)
    print("val accuracy :", round(res["accuracy"], 4))
    print("val precision:", round(res["precision"], 4))
    print("val recall   :", round(res["recall"], 4))
    print("val F1       :", round(res["f1"], 4))

    import joblib
    joblib.dump(model, args.out)  # 序列化保存模型
    print(f"[saved] {args.out}")

    # 记录训练/验证样本数与口径，便于复盘对齐论文
    dinfo["train_rows"] = int(Xtr.shape[0])
    dinfo["val_rows"] = int(Xva.shape[0])
    out_meta = Path(args.out).with_suffix(".train_meta.json")
    out_meta.write_text(json.dumps({"dataset": dinfo, "metrics": res,
                                   "params": params},
                                  ensure_ascii=False, indent=2),
                        encoding="utf-8")


# 脚本直接运行时执行 main()
if __name__ == "__main__":
    main()
