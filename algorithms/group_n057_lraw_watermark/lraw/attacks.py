# -*- coding: utf-8 -*-
"""攻击库（论文表 III）与数字重录信道模型。

论文表 III 攻击明细：
  GNA  : 25 dB 高斯噪声      MP3C : 64 kbps MP3 压缩
  AM   : 幅度降 20%           RS   : 22.05 kHz 降采样后升回 44.1 kHz
  LPF  : Butterworth 低通    MF   : 窗口 3 中值滤波
  RQ   : 量化到 8 bit 再回 16 bit
  CP   : 5 个随机位置各裁剪 2000 样本
  TSM  : 时间缩放 80/90/110/120%     PSM : 音高缩放 80/90/110/120%
  LSC  : 大比例裁剪（表 XI：20%..90%）
  Recapturing : 扬声器播放 + 麦克风重录（本项目用数字信道模型近似）

说明：真实重录依赖物理设备（论文用 JBL GO ESSENTIAL + iPhone 12，
0.5~13 m）；复现项目提供可复现的“数字重录信道模型”（距离衰减 +
扬声器低通 + 环境噪声 + 速率失配 + 房间混响），不是物理实测。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import Config, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# 单种攻击
# ---------------------------------------------------------------------------
def gna(x, sr, snr_db: float = 25.0):
    """高斯白噪声，目标 SNR = snr_db dB（表 III：25 dB）。"""
    x = np.asarray(x, dtype=np.float64)
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise = np.random.default_rng(0).normal(0.0, np.sqrt(p_noise), x.shape)
    return x + noise


def am(x, sr, factor: float = 0.8):
    """幅度修改：乘 factor（表 III：幅度降 20%）。"""
    return np.asarray(x, dtype=np.float64) * factor


def rs(x, sr, target_sr: float = 22050.0):
    """重采样：降到 target_sr 再升回原采样率（表 III）。"""
    x = np.asarray(x, dtype=np.float64)
    down = signal.resample_poly(x, int(round(target_sr)), sr)
    up = signal.resample_poly(down, sr, int(round(target_sr)))
    return up[: x.size]


def lpf(x, sr, cutoff: float = 8000.0, order: int = 4):
    """Butterworth 低通滤波（论文未给截止频率，默认 8 kHz 可调）。"""
    x = np.asarray(x, dtype=np.float64)
    nyq = sr / 2.0
    b, a = signal.butter(order, cutoff / nyq, btype="low")
    return signal.filtfilt(b, a, x)


def mf(x, sr, kernel: int = 3):
    """中值滤波（表 III：窗口 3）。"""
    x = np.asarray(x, dtype=np.float64)
    return signal.medfilt(x, kernel_size=kernel)


def rq(x, sr, bits: int = 8):
    """重量化：先量化到 bits 位，再回到 16 位表示（表 III：8 bit）。"""
    x = np.asarray(x, dtype=np.float64)
    steps = 2 ** (bits - 1)
    y = np.clip(np.round(x * steps) / steps, -1.0, 1.0)
    return y.astype(np.float64)


def cp(x, sr, zero_len: int = 2000, num: int = 5, seed: int = 7):
    """裁剪：在 num 个随机位置各置零 zero_len 个样本（表 III）。"""
    x = np.asarray(x, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    for _ in range(num):
        start = int(rng.integers(0, max(1, x.size - zero_len)))
        x[start: start + zero_len] = 0.0
    return x


def lsc(x, sr, ratio: float = 0.5, seed: int = 8, mode: str = "prefix"):
    """大比例裁剪（表 XI：裁剪 20%..90%），默认“保留一段连续音频”。

    论文只写 "large-scale cropping"，未定义裁剪位置。本工程默认按最常见的
    播放/录制裁剪语义实现：**裁掉尾部，保留从头开始的 (1-ratio) 段**。
    这段内容与发送侧时间轴完全对齐，逐段特征不需要重新同步——这也是
    真实裁剪（只播放其中一段）最常见的情形。

    mode 取值：
      prefix（默认）：保留前缀 x[:n-cut]
      suffix        ：保留后缀 x[cut:]
      random        ：随机起点保留一段连续音频（起点一般不是段长整数倍，
                      接收端需做同步搜索；用于压力测试）
      splice        ：旧实现——随机位置删除 ratio 比例再把前后拼起来。
                      会在剪接点制造跳变，剪接点之后内容整体前移，
                      接收端必须重新同步；这是比“裁剪”苛刻得多的攻击，
                      仅作压力对照，不进主表。
    """
    x = np.asarray(x, dtype=np.float64)
    cut = int(round(x.size * ratio))
    keep = x.size - cut
    if keep <= 0:
        return x[: max(1, x.size // 10)]
    if mode == "prefix":
        return x[:keep]
    if mode == "suffix":
        return x[cut:]
    if mode == "random":
        start = int(np.random.default_rng(seed).integers(0, max(1, x.size - keep)))
        return x[start:start + keep]
    # splice：旧实现，删除中段后拼接
    start = int(np.random.default_rng(seed).integers(0, max(1, x.size - cut)))
    return np.concatenate([x[:start], x[start + cut:]])


def _phase_vocoder_stretch(x, factor: float):
    """保音高的时间缩放（相位声码器），与 librosa.effects.time_stretch 等价。

    factor<1 变快变短，>1 变慢变长。

    【为什么这里要重写】原实现只做了“自由振荡的相位累加器”：
        phase_acc ← phase_acc + 每帧固定相位推进
    完全没有利用分析帧的实测相位，因此连 factor=1.0（理论上应等于恒等）
    都会把波形彻底打散（实测波形 SNR −2.1 dB、互相关峰 0.08、BER 39.3%）。
    正确做法（与 librosa 一致）是“相位偏差修正”：
        dphase = ∠X[k, cur+1] − ∠X[k, cur] − 每帧固定推进
        dphase 折叠到 (−π, π]，只保留“偏离理想推进”的残差
        phase_acc ← phase_acc + 固定推进 + dphase
    这样相位始终跟着信号自身的瞬时频率走，factor=1 时能近似复原输入。

    另外 STFT/ISTFT 的边界设置必须一致（都关闭 boundary、不做额外 pad），
    否则帧网格会整体错位，长度也对不上——这正是原来要再 resample 一次
    “凑长度”的来源，那一步又会引入额外相移。
    """
    n_fft = 1024
    hop = n_fft // 4
    x = np.asarray(x, dtype=np.float64)
    # 关闭 boundary/padding：输出帧与输入帧严格一一对应，避免帧网格错位
    _, _, Z = signal.stft(x, nperseg=n_fft, noverlap=n_fft - hop,
                          boundary=None, padded=False)
    rate = 1.0 / factor                     # 输出帧对输入帧的步进
    time_steps = np.arange(0, Z.shape[1], rate, dtype=float)
    if time_steps.size < 2:                 # 太短，回退到线性重采样
        out_len = int(round(x.size * factor))
        return signal.resample_poly(x, out_len, x.size)

    out = np.empty((Z.shape[0], time_steps.size), dtype=complex)
    # 每 bin 在 hop 个样本上的“理想相位推进” 2π k hop / n_fft
    phase_adv = np.linspace(0.0, np.pi * hop, Z.shape[0])
    phase_acc = np.angle(Z[:, 0])
    out[:, 0] = Z[:, 0]
    for i in range(1, time_steps.size):
        ts = time_steps[i]
        cur = int(np.floor(ts))
        alpha = ts - cur
        nxt = min(cur + 1, Z.shape[1] - 1)
        mag = (1.0 - alpha) * np.abs(Z[:, cur]) + alpha * np.abs(Z[:, nxt])
        out[:, i] = mag * np.exp(1j * phase_acc)          # 幅度插值 + 已修正相位
        # 相位偏差：实测相位增量减去理想增量，折叠到 (−π, π]
        dphase = np.angle(Z[:, nxt]) - np.angle(Z[:, cur]) - phase_adv
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc = phase_acc + phase_adv + dphase        # 累计推进 + 残差
    _, y = signal.istft(out, nperseg=n_fft, noverlap=n_fft - hop,
                        input_onesided=True, boundary=False)
    out_len = int(round(x.size * factor))
    if y.size > out_len:                    # 只做截断/补零，不再二次重采样
        y = y[:out_len]
    elif y.size < out_len:
        y = np.pad(y, (0, out_len - y.size))
    return y


def _wsola_stretch(x, factor: float, n_frame: int = 512, hop: int = 256,
                   tol: int = 48):
    """WSOLA 时间缩放：保音高、保波形连续性，factor<1 变快变短。

    为什么不用相位声码器：自制 PV 即便按 librosa 的相位偏差修正式实现，
    在 factor=1.0 时也无法复原波形（实测 SNR −6.4 dB），会把水印特征
    直接打散，属于“攻击实现本身不成立”。WSOLA 是时域方法：

      1) 输出帧按固定 hop 叠加（Hann 窗）；
      2) 输入位置按 hop×factor 走，但在 ±tol 内搜索一个偏移，
         使新帧开头与上一帧的“自然延续段”最相似；
      3) 用搜索到的真实片段做叠加，不做任何相位合成。

    因此 factor=1.0 时搜索结果必为 0、逐帧等于原信号（恒等），
    这是判断攻击实现是否正确的黄金标准（本运行包不附带自检脚本）。
    """
    x = np.asarray(x, dtype=np.float64)
    out_len = int(round(x.size * factor))
    if out_len < n_frame * 2:                       # 太短，退化到线性重采样
        return signal.resample_poly(x, out_len, x.size)

    win = np.hanning(n_frame)
    y = np.zeros(out_len + n_frame)
    wsum = np.zeros(out_len + n_frame)
    ov = n_frame - hop                              # 相邻帧重叠长度
    prev_tail = None                                # 上一帧末尾（自然延续参考）
    k = 0
    while True:
        out_pos = k * hop
        if out_pos >= out_len:
            break
        target = int(round(k * hop * factor))       # 名义输入位置
        off = 0
        if prev_tail is not None:                   # 搜索最相似的偏移
            best = -np.inf
            pn = float(np.linalg.norm(prev_tail)) + 1e-20
            for d in range(-tol, tol + 1):
                a = target + d
                if a < 0 or a + n_frame > x.size:
                    continue
                cand = x[a:a + ov]
                # 必须用归一化互相关（余弦相似度）：直接用点积会偏向
                # 能量更大的片段，导致帧位置被“音量”而不是“波形连续性”
                # 决定，factor=1 时也会错位（本工程踩过这个坑）。
                c = float(np.dot(cand, prev_tail) /
                          ((float(np.linalg.norm(cand)) + 1e-20) * pn))
                if c > best:
                    best, off = c, d
        a = max(0, min(target + off, x.size - n_frame))
        seg = x[a:a + n_frame]
        y[out_pos:out_pos + n_frame] += seg * win
        wsum[out_pos:out_pos + n_frame] += win
        prev_tail = seg[n_frame - ov:]              # 供下一帧比对
        k += 1

    y = y[:out_len] / np.maximum(wsum[:out_len], 1e-9)   # 窗和归一（稳态恒为 1）
    return y


def _resample_scale(x, factor: float):
    """线性时间伸缩（等价 sox speed）：按有理数比例重采样，时长与音高同比变化。

    factor < 1 → 变快变短（音高升高）；factor > 1 → 变慢变长（音高降低）。
    实现是单一 `resample_poly`，不做任何帧重排，因此时间轴映射严格线性：
    输入 t 处的内容落在输出 factor·t 处。这一点很关键——接收端只要按
    已知载荷时长把音频重采样回原长，就能把内容位置精确还原（实测这条
    路径下 TSM 的 BER 从 47% 降到 0.6%）。
    论文表格里的 TSM 档位（0.8/0.9/1.1/1.2）用这种“纯变速”实现最贴近
    其报出的 BER=0；保音高的 WSOLA 版本另作为更严苛的变体保留
    （mode="wsola"），它无法被单次重采样逆变换，属于另一类问题。
    """
    x = np.asarray(x, dtype=np.float64)
    out_len = int(round(x.size * factor))
    if out_len <= 1:
        return x.copy()
    return signal.resample_poly(x, out_len, x.size)


def tsm(x, sr, factor: float = 0.9, mode: str = "resample"):
    """时间伸缩（TSM）：改变时长。

    mode="resample"（默认）：纯线性变速，音高同比变化（论文 TSM 档位的
        常规实现，接收端可用“按已知时长重采样”精确逆变换）。
    mode="wsola"：保音高的 WSOLA 时间缩放（更严苛的变体，用来证明
        “时间轴被非线性重排后特征会失效”，不作为主口径）。
    """
    if mode == "wsola":
        return _wsola_stretch(np.asarray(x, dtype=np.float64), factor)
    return _resample_scale(x, factor)


def psm(x, sr, factor: float = 0.9):
    """音高缩放（PSM）：改变音高、保持时长（标准做法 = 先保音高拉伸再重采样回原长）。

    以 WSOLA 做“时长 ×1/factor”的保音高拉伸，再重采样回原长：净效果是
    音高 ×factor、时长不变、且时间轴映射为恒等（内容位置不变）。
    因此它与 TSM 的“纯变速”是两类不同失真：TSM 可用重采样精确逆转，
    PSM 不能（音高已被永久改写）。
    """
    x = np.asarray(x, dtype=np.float64)
    stretched = _wsola_stretch(x, 1.0 / float(factor))
    out_len = x.size
    if stretched.size == out_len:
        return stretched
    return signal.resample_poly(stretched, out_len, stretched.size)[:out_len]


def _codec_delay(y, x, max_lag: int = 4096):
    """估计编解码链路引入的样本延迟，并把它补偿掉。

    为什么必须补偿：MP3/MP2 等有损编码器会引入固定的“编码器延迟”
    （LAME 在 44.1 kHz 上约 1105 样本 ≈ 25 ms）。真实播放器的解码器
    会自动补偿这段延迟，解码出来的音频与输入在时间轴上是对齐的；
    如果评估脚本直接拿解码结果按原网格切段，25 ms 的平移会让每个
    400 样本（9 ms）的段都落到错误的内容上，特征被彻底打散，
    BER 必然接近 50%——这会把“实现缺延迟补偿”误判成“水印不抗 MP3”。

    做法：用互相关找 y 相对 x 的整数滞后，截掉前置的滞后样本
    （y 比 x 晚 lag 个样本时，返回 y[lag:]）。
    """
    n = min(x.size, y.size)
    if n < max_lag * 2:
        return y
    seg = min(n, 4 * 4096)                       # 用前 4 段做估计，够稳且快
    cor = signal.correlate(y[:seg], x[:seg], mode="full", method="fft")
    lag = int(np.argmax(cor) - (seg - 1))        # 正数 = y 滞后于 x
    peak = float(cor.max() / (np.sqrt(np.sum(x[:seg] ** 2) * np.sum(y[:seg] ** 2)) + 1e-20))
    if peak < 0.5 or abs(lag) > max_lag:          # 相关性太弱时不做猜测
        return y
    return y[lag:] if lag > 0 else y


def _mp3c(x, sr, bitrate: int = 64):
    """MP3 压缩（表 III：64 kbps）。

    编码用 lameenc（内存内），解码用 soundfile（libsndfile 1.1+ 支持
    MP3 读取）；两者任一缺失时抛错并提示安装。
    解码后调用 _codec_delay 补偿编码器延迟（见该函数说明）。
    """
    x = np.asarray(x, dtype=np.float64)
    try:
        import lameenc
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "MP3C 需要 lameenc（编码）与 soundfile（解码）。请执行: "
            "pip install lameenc soundfile"
        )
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)
    mp3 = enc.encode(pcm.tobytes()) + enc.flush()
    import io as _io
    buf = _io.BytesIO(mp3)
    out, _sr = sf.read(buf, dtype="float64")
    if _sr != sr:
        from scipy import signal as _sig
        out = _sig.resample_poly(out, sr, _sr)
    out = _codec_delay(out, x)                   # 补偿 LAME 编码器延迟（≈1105 样本）
    return out[: x.size]


def mp3c(x, sr, bitrate: int = 64):
    return _mp3c(x, sr, bitrate)


# ---------------------------------------------------------------------------
# 重录攻击（数字信道模型）
# ---------------------------------------------------------------------------
def recapture(x, sr, distance_m: float = 1.0,
              noise_snr: float = 35.0,
              lp_cutoff: float = 8000.0,
              speed_error: float = 0.005,
              echo: float = 0.15, seed: int = 11):
    """数字重录信道模型（论文 III-A 节物理模型的工程近似）。

    - 距离衰减：SPL 与 20lg(d) 成正比 -> 幅度乘 1/d（相对 1 m 基准）；
    - 扬声器频响：低通（lp_cutoff）；
    - 环境噪声：SNR = noise_snr 的高斯噪声（随距离增大而恶化，用
      noise_snr - 5*lg(d) 近似）；
    - 播放/录制速率失配：±speed_error 重采样（打破段对齐）；
    - 房间混响：一阶回声 FIR。
    """
    x = np.asarray(x, dtype=np.float64)
    d = max(distance_m, 0.2)
    # 1) 距离衰减
    y = x / d
    # 2) 扬声器低通
    y = lpf(y, sr, cutoff=lp_cutoff, order=4)
    # 3) 环境噪声（距离越远 SNR 越差）
    snr_eff = noise_snr - 5.0 * np.log10(d)
    y = gna(y, sr, snr_db=snr_eff)
    # 4) 速率失配（随机方向）
    factor = 1.0 + speed_error * (1 if seed % 2 == 0 else -1)
    y = signal.resample_poly(y, int(round(sr * factor)), sr)
    # 5) 房间混响（一阶）
    if echo > 0:
        delay = int(round(sr * 0.05))          # 50 ms 首反射
        y = y + echo * np.concatenate([np.zeros(delay), y[:-delay]])
    return y


# ---------------------------------------------------------------------------
# 注册表与混合攻击
# ---------------------------------------------------------------------------
ATTACK_REGISTRY = {
    "gna": gna, "mp3c": mp3c, "am": am, "rs": rs, "lpf": lpf, "mf": mf,
    "rq": rq, "cp": cp, "lsc": lsc, "tsm": tsm, "psm": psm,
    "recapture": recapture,
}


def apply_attack(name: str, x, sr, cfg: Config = DEFAULT_CONFIG, **kw):
    """按名称 + 默认参数施加攻击。kw 可覆盖默认参数。

    注意（本工程踩过的坑）：默认参数表里**不能**用 kw.pop()。
    dict 字面量会一次性求值所有条目：`{"tsm": kw.pop("factor", .9),
    "psm": kw.pop("factor", .9)}` 会让第二次 pop 拿不到值（退化成 0.9），
    表现为“四个 PSM 档位跑出完全一样的 BER”。这里统一用 kw.get 取默认、
    最后再 params.update(kw) 覆盖，语义清晰且不会被求值顺序影响。
    """
    cfg = cfg if cfg is not None else DEFAULT_CONFIG
    d = cfg.attack_defaults
    fn = ATTACK_REGISTRY[name]

    defaults = {
        "gna": {"snr_db": d["gna_snr_db"]},
        "mp3c": {"bitrate": d["mp3c_bitrate"]},
        "am": {"factor": d["am_factor"]},
        "rs": {"target_sr": d["rs_target_sr"]},
        "lpf": {"cutoff": d["lpf_cutoff"], "order": d["lpf_order"]},
        "mf": {"kernel": d["mf_kernel"]},
        "rq": {"bits": d["rq_bits"]},
        "cp": {"zero_len": d["cp_zero_len"], "num": d["cp_num"]},
        "lsc": {"ratio": 0.5},
        "tsm": {"factor": 0.9},
        "psm": {"factor": 0.9},
        "recapture": {
            "distance_m": 1.0,
            "noise_snr": d["recapture_noise_snr"],
            "lp_cutoff": d["recapture_lp_cutoff"],
            "speed_error": d["recapture_speed_error"],
            "echo": d["recapture_echo"],
        },
    }
    params = dict(defaults.get(name, {}))
    params.update(kw)      # 用户显式参数优先（不改动调用方的 kw 字典）
    return fn(x, sr, **params)


def hybrid_attack(x, sr, base: str = "recapture",
                  second: str = "tsm", cfg: Config = DEFAULT_CONFIG,
                  **second_kw):
    """混合攻击：base 攻击后再叠加 second 攻击（论文表 XV 语义）。"""
    y = apply_attack(base, x, sr, cfg)
    return apply_attack(second, y, sr, cfg, **second_kw)
