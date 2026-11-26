# -*- coding: utf-8 -*-
"""LightGBM 模型封装：加载主模型 / 训练（含贝叶斯优化选项）/ 评估。"""
from __future__ import annotations

import os
from pathlib import Path

from .config import MODEL_PATH


def load_model(path: str | None = None):
    """加载 LightGBM 分类器。

    优先级：path 参数 > 环境变量 LRAW_MODEL_PATH > assets/author_trained.pkl。
    本包只保留本机训练的 author_trained.pkl；作者原模型 LGBClassifier.pkl
    已移出运行包，不再作为兜底。
    """
    import joblib

    if path is None:
        path = os.environ.get("LRAW_MODEL_PATH")
    if path is None:
        path = str(MODEL_PATH)

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在：{model_path}。请确认 assets/author_trained.pkl "
            f"存在，或用 LRAW_MODEL_PATH / --model 指向有效模型。"
        )
    return joblib.load(str(model_path))


def train_model(X, y, params: dict | None = None,
                test_size: float = 0.1,
                random_state: int = 42,
                bayes_opt: bool = False):
    """按论文 IV-D 节流程训练 LightGBM。

    - 默认使用作者开源 hyper_parameters.txt 中的优化参数；
    - bayes_opt=True 时先做贝叶斯超参搜索（需 pip install bayes_opt）；
    - 与作者 model_train.py 一致：train_test_split(test_size=0.1, rs=42)。
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    if bayes_opt:
        params = _bayesian_tune(X_train, y_train, params)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model, (X_train, X_val, y_train, y_val)


def _bayesian_tune(X_train, y_train, base_params: dict | None):
    """贝叶斯优化（论文 IV-D 节；搜索范围见表 II）。"""
    from bayes_opt import BayesianOptimization
    from sklearn.model_selection import cross_val_score
    import lightgbm as lgb

    base = dict(base_params or {})

    def lgb_cv(learning_rate, n_estimators, min_child_weight, colsample_bytree,
               min_child_samples, num_leaves, subsample, max_depth,
               reg_alpha, reg_lambda):
        m = lgb.LGBMClassifier(
            boosting_type="gbdt", objective="binary",
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators), n_jobs=-1, random_state=None,
            colsample_bytree=float(colsample_bytree),
            min_child_weight=float(min_child_weight),
            num_leaves=int(num_leaves), max_depth=int(max_depth),
            min_child_samples=int(min_child_samples),
            reg_alpha=float(reg_alpha), reg_lambda=float(reg_lambda),
            subsample=float(subsample))
        return cross_val_score(m, X_train, y_train, scoring="f1", cv=5).mean()

    bo = BayesianOptimization(lgb_cv, {
        "learning_rate": (0.001, 0.3),
        "n_estimators": (10, 10000),
        "min_child_weight": (0.0001, 0.5),
        "reg_alpha": (0, 5),
        "reg_lambda": (0, 5),
        "colsample_bytree": (0.7, 1),
        "min_child_samples": (2, 25),
        "num_leaves": (5, 250),
        "subsample": (0.7, 1),
        "max_depth": (2, 10),
    })
    bo.maximize(init_points=5, n_iter=20)
    best = bo.max["params"]
    best.update(base)
    return best


def evaluate(model, X, y):
    """accuracy / precision / recall / F1（论文公式 10 的 F1）。"""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    y_pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }
