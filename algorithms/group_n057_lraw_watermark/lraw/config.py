# -*- coding: utf-8 -*-
"""
LRAW —— LightGBM-based Robust Audio Watermarking（论文 TIFS 2025 复现项目）

全局配置。所有论文中未明确给定的参数均在此集中管理，便于对照论文做消融。
论文原文: Z. Su et al., "LightGBM-Based Audio Watermarking Robust to
Recapturing and Hybrid Attacks", IEEE TIFS, vol. 20, 2025.
"""
from dataclasses import dataclass, field, asdict
from math import ceil, log2
from pathlib import Path


# ---------------------------------------------------------------------------
# 路径约定（相对项目根目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"          # 主模型 / 数据集 / 超参数
DATA_DIR = PROJECT_ROOT / "data"              # 测试音频
OUTPUT_DIR = PROJECT_ROOT / "output"          # 指标与图表输出
FIG_DIR = OUTPUT_DIR / "figs"

MODEL_PATH = ASSETS_DIR / "author_trained.pkl"    # 唯一主模型（作者数据集 + 作者超参重训）
DATASET_PATH = ASSETS_DIR / "dataset.xlsx"        # 396,000 条 DWT-GBT-SVD 特征数据
HYPERPARAMS_PATH = ASSETS_DIR / "hyper_parameters.txt"


# ---------------------------------------------------------------------------
# 论文给出的关键常量
# ---------------------------------------------------------------------------
# 嵌入量化规则参数（论文 IV-C 节：a=5, b=2 在不可感知性与鲁棒性间折中）
A = 5
B = 2

# LFSR 阶数 L = ceil(log2(n+1))，n 为水印位数。论文实验固定 L=13（n=4096）
# 对应本原多项式 x^13 + x^4 + x^3 + x + 1（与 m-SW-LSC [25] 一致）
PRIMITIVE_POLYNOMIAL = {
    # L: 本原多项式（给出次数非零项的指数集合；最高次项恒为 L 次）
    5:  [5, 2],
    6:  [6, 1],
    7:  [7, 3],
    8:  [8, 4, 3, 2],
    9:  [9, 4],
    10: [10, 3],
    11: [11, 2],
    12: [12, 6, 4, 1],
    13: [13, 4, 3, 1],     # 论文所用
    14: [14, 10, 6, 1],
    15: [15, 1],
    16: [16, 12, 3, 1],
}


@dataclass
class Config:
    # ---- 信号与采样 ----
    sr: int = 44100                 # 采样率（论文：44.1 kHz / 16 bit）
    # ---- 分段与水印 ----
    n_segments: int = 4096          # 水印位数 n（论文实验用 4096；训练数据用 330）
    # ---- DWT-GBT-SVD 特征 ----
    wavelet: str = "db4"            # 小波基（论文未指定，默认 db4，可用 haar/db2... 消融）
    dwt_levels: int = 2             # 两层 DWT，取第 2 层近似系数（长度 = 段长/4）
    svd_rows: int = 5               # Y 矩阵行数（5×20 = 100 个稀疏系数）
    svd_cols: int = 20
    # ---- 嵌入 ----
    a: int = A                      # 量化规则参数 a（wi=0 时向 s4 拉近）
    b: int = B                      # 量化规则参数 b（wi=1 时向 s2 拉近）
    # ---- VAD（可选预处理；默认关闭——固定网格保证嵌入/提取段对齐） ----
    use_vad: bool = False           # False=固定网格（论文 330段/3s 口径）；True=能量VAD去静音
    vad_frame_ms: float = 20.0      # 能量 VAD 帧长
    vad_threshold_mode: str = "percentile"   # percentile / mean / fixed
    vad_threshold: float = 0.15     # 能量阈值：低于该分位数的帧视为静音（percentile 模式）
    # ---- LightGBM ----
    lgb_params: dict = field(default_factory=lambda: {
        "boosting_type": "gbdt",
        "objective": "binary",
        "colsample_bytree": 0.8218,
        "learning_rate": 0.2332,
        "max_depth": 9,
        "min_child_samples": 20,
        "min_child_weight": 0.4957,
        "n_estimators": 519,
        "num_leaves": 70,
        "reg_alpha": 0.6304,
        "reg_lambda": 3.487,
        "subsample": 0.7411,
        "n_jobs": -1,
        "random_state": None,
    })
    # ---- 攻击（表 III 默认值） ----
    attack_defaults: dict = field(default_factory=lambda: {
        "gna_snr_db": 25.0,          # GNA: 25 dB 高斯噪声
        "mp3c_bitrate": 64,          # MP3C: 64 kbps
        "am_factor": 0.8,            # AM: 幅度降 20%
        "rs_target_sr": 22050,       # RS: 降到 22.05 kHz 再升回
        "lpf_cutoff": 8000,          # LPF: Butterworth 低通（论文未给截止频率，可调）
        "lpf_order": 4,
        "mf_kernel": 3,              # MF: 窗口 3 的中值滤波
        "rq_bits": 8,                # RQ: 量化到 8 bit 再回到 16 bit
        "cp_zero_len": 2000,         # CP: 5 个随机位置各裁剪 2000 个样本
        "cp_num": 5,
        "lsc_ratios": [0.2, 0.5, 0.7, 0.8, 0.9],   # LSC: 裁剪比例
        "tsm_factors": [0.8, 0.9, 1.1, 1.2],
        "psm_factors": [0.8, 0.9, 1.1, 1.2],
        "recapture_distances": [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 13.0],
        "recapture_noise_snr": 35.0, # 数字重录信道模型：环境噪声 SNR（可调）
        "recapture_lp_cutoff": 8000, # 扬声器频响近似（低通）
        "recapture_speed_error": 0.0001,  # 播放/录制速率失配（0.01%，接近真实设备时钟漂移 ~100ppm）
        "recapture_echo": 0.15,      # 房间混响强度（数字近似）
    })

    # ---- 派生属性 ----
    @property
    def L(self) -> int:
        """m 序列阶数 L = ceil(log2(n+1))，论文命题 2。"""
        return ceil(log2(self.n_segments + 1))

    @property
    def lfsr_taps(self):
        return PRIMITIVE_POLYNOMIAL[self.L]

    @property
    def n(self) -> int:
        return self.n_segments

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = Config()
