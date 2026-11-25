# -*- coding: utf-8 -*-
"""LRAW 统一前端接入服务：POST /process + GET /health + GET /files/...

用法（在 LRAW-reproduction 根目录）：
    pip install -r requirements.txt gunicorn matplotlib
    gunicorn --worker-class gthread -w 1 --threads 4 --timeout 600 -b 0.0.0.0:8000 platform_service:app

环境变量：
    LRAW_MODEL_PATH  模型路径，默认 assets/author_trained.pkl
    PUBLIC_BASE_URL  对外地址，例如 http://1.2.3.4:8000；
                     不设时按请求的 url_root 生成
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(os.environ.get("LRAW_ROOT", Path(__file__).resolve().parent)).resolve()
sys.path.insert(0, str(ROOT))

from lraw import DEFAULT_CONFIG, read_audio, write_audio, embed, extract_bits, ber, blind_recover, align_probs  # noqa: E402
from lraw.attacks import apply_attack, hybrid_attack  # noqa: E402
from lraw.model import load_model  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
cfg = DEFAULT_CONFIG
MODEL = load_model()

OUTPUT_ROOT = ROOT / "output" / "platform"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# 平台下拉框的固定档位：与 output/attack_matrix_results_L4.csv 的 31 档主矩阵对齐，
# 额外保留 mp3c_128 和干净信道。键必须与 public/methods/group_n057_lraw_watermark.json 的 options 一致。
PRESETS = {
    "none": ("none", "无攻击（干净信道）", {}),
    "gna": ("gna", "GNA 高斯噪声 25 dB", {"snr_db": 25.0}),
    "mp3c_64": ("mp3c", "MP3C 64 kbps", {"bitrate": 64}),
    "mp3c_128": ("mp3c", "MP3C 128 kbps", {"bitrate": 128}),
    "am": ("am", "AM 幅度衰减 0.8×", {"factor": 0.8}),
    "rs": ("rs", "RS 重采样 22.05 kHz", {"target_sr": 22050}),
    "lpf": ("lpf", "LPF 低通 8 kHz / 4 阶", {"cutoff": 8000, "order": 4}),
    "mf": ("mf", "MF 中值滤波 3 点", {"kernel": 3}),
    "rq": ("rq", "RQ 再量化 8 bit", {"bits": 8}),
    "cp": ("cp", "CP 随机裁剪置零 5×2000 点", {"zero_len": 2000, "num": 5}),
    "tsm_0.8": ("tsm", "TSM 0.8× 纯变速", {"factor": 0.8}),
    "tsm_0.9": ("tsm", "TSM 0.9× 纯变速", {"factor": 0.9}),
    "tsm_1.1": ("tsm", "TSM 1.1× 纯变速", {"factor": 1.1}),
    "tsm_1.2": ("tsm", "TSM 1.2× 纯变速", {"factor": 1.2}),
    "psm_0.8": ("psm", "PSM 0.8× 相位声码器", {"factor": 0.8}),
    "psm_0.9": ("psm", "PSM 0.9× 相位声码器（方法边界）", {"factor": 0.9}),
    "psm_1.1": ("psm", "PSM 1.1× 相位声码器（方法边界）", {"factor": 1.1}),
    "psm_1.2": ("psm", "PSM 1.2× 相位声码器（方法边界）", {"factor": 1.2}),
    "lsc_0.2": ("lsc", "LSC 保留前缀 20%", {"ratio": 0.2}),
    "lsc_0.5": ("lsc", "LSC 保留前缀 50%", {"ratio": 0.5}),
    "lsc_0.7": ("lsc", "LSC 保留前缀 70%", {"ratio": 0.7}),
    "lsc_0.8": ("lsc", "LSC 保留前缀 80%", {"ratio": 0.8}),
    "lsc_0.9": ("lsc", "LSC 保留前缀 90%", {"ratio": 0.9}),
    "recapture_0.5m": ("recapture", "数字重录 0.5 m", {"distance_m": 0.5}),
    "recapture_1m": ("recapture", "数字重录 1 m", {"distance_m": 1.0}),
    "recapture_2m": ("recapture", "数字重录 2 m", {"distance_m": 2.0}),
    "recapture_3m": ("recapture", "数字重录 3 m", {"distance_m": 3.0}),
    "recapture_5m": ("recapture", "数字重录 5 m", {"distance_m": 5.0}),
    "recapture_10m": ("recapture", "数字重录 10 m", {"distance_m": 10.0}),
    "recapture_13m": ("recapture", "数字重录 13 m", {"distance_m": 13.0}),
    "hybrid_recap1m+ecsp": ("hybrid", "混合：重录 1 m + ECSP（8 种 CSP 串行）", {"base": "recapture", "second": "ecsp"}),
    "hybrid_recap1m+tsm": ("hybrid", "混合：重录 1 m + TSM 0.9×", {"base": "recapture", "second": "tsm"}),
    "hybrid_recap1m+psm": ("hybrid", "混合：重录 1 m + PSM 0.9×", {"base": "recapture", "second": "psm"}),
}
ECSP_ATTACKS = ("gna", "mp3c", "am", "rs", "lpf", "mf", "rq", "cp")


def apply_preset(attack: str, wm: np.ndarray, sr: int, cfg, kwargs: dict):
    """按平台档位施加攻击；hybrid 档位走混合链路。"""
    if attack == "none":
        return wm
    if attack == "hybrid":
        base = kwargs.get("base", "recapture")
        second = kwargs.get("second", "tsm")
        if second == "ecsp":
            y = apply_attack(base, wm, sr, cfg)
            for name in ECSP_ATTACKS:
                y = apply_attack(name, y, sr, cfg)
            return y
        return hybrid_attack(wm, sr, base=base, second=second, cfg=cfg)
    return apply_attack(attack, wm, sr, cfg, **kwargs)
INT_KEYS = {"bitrate"}


def text_to_bits(msg: str) -> np.ndarray:
    raw = msg.encode("ascii", "ignore")
    return np.array([int(c) for c in "".join(f"{b:08b}" for b in raw)], dtype=np.int64)


def bits_to_text(bits: np.ndarray) -> str:
    b = np.asarray(bits, dtype=np.int64).ravel()
    k = (b.size // 8) * 8
    if k == 0:
        return ""
    chars = []
    for row in b[:k].reshape(-1, 8):
        code = int("".join(map(str, row)), 2)
        chars.append(chr(code) if 32 <= code < 127 else ".")
    return "".join(chars)


def read_upload(file_storage) -> np.ndarray:
    data = file_storage.read()
    tmp = ROOT / "output" / f"_platform_upload_{uuid.uuid4().hex}.wav"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(data)
    try:
        x, sr = read_audio(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    if sr != cfg.sr:
        from scipy import signal as sg
        x = sg.resample_poly(x, cfg.sr, sr)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < cfg.sr * 3:
        raise ValueError("音频太短：至少 3 秒（保证 330 段的最小载荷）")
    return x


def make_waveform_png(path: Path, host: np.ndarray, wm: np.ndarray, attacked: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(host.size, wm.size, attacked.size)
    t = np.arange(n) / cfg.sr
    # 长音频（例如 4 分钟 ≈ 1100 万样本）不能把每个采样点都塞给 matplotlib：
    # 三面板各画 1100 万点会占用大量内存且拖慢请求。这里等间隔抽到
    # 4000 点展示波形轮廓；完整数据仍在 watermarked.wav / attacked.wav。
    max_plot_points = 4000
    if n > max_plot_points:
        idx = np.linspace(0, n - 1, max_plot_points, dtype=np.int64)
        t = t[idx]
        series = [np.asarray(y, dtype=np.float64).ravel()[:n][idx]
                  for y in (host, wm, attacked)]
    else:
        series = [np.asarray(y, dtype=np.float64).ravel()[:n]
                  for y in (host, wm, attacked)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, y, title in zip(
        axes,
        series,
        ["Host audio", "Watermarked", "Attacked"],
    ):
        ax.plot(t, y, linewidth=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_title(title, loc="left", fontsize=10)
    axes[-1].set_xlabel("Time / s")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_summary_png(path: Path, report: dict) -> None:
    """Render a human-readable one-page recovery summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    bg = "#F7F5EF"
    ink = "#1F3D33"
    muted = "#6C7A70"
    green = "#2F6B4F"
    red = "#A63D2F"
    card = "#FFFFFF"
    border = "#D8D2C4"

    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blind = report.get("blind", {})
    locked = bool(blind.get("locked"))
    status = "LOCKED" if locked else "UNLOCKED"
    status_color = green if locked else red
    text_match = report.get("text_match")
    match_text = "YES" if text_match else ("N/A" if not report.get("message") else "NO")

    ax.text(0.06, 0.93, "LRAW Recovery Summary", fontsize=24, fontweight="bold",
            color=ink, va="top")
    ax.text(0.06, 0.82, f"Attack preset: {report.get('attack_preset', '-')}"
            f"   |   repeat: {report.get('repeat', '-')}",
            fontsize=10.5, color=muted, va="top")

    ax.add_patch(FancyBboxPatch((0.71, 0.84), 0.22, 0.075,
                                boxstyle="round,pad=0.012,rounding_size=0.015",
                                linewidth=0, facecolor=status_color))
    ax.text(0.82, 0.877, status, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")

    metrics = [
        ("Blind BER", f"{float(blind.get('ber', 0.0)) * 100:.2f}%"),
        ("Raw BER", f"{float(report.get('raw_ber', 0.0)) * 100:.2f}%"),
        ("Embed SNR", f"{float(report.get('snr_embed_db', 0.0)):.2f} dB"),
        ("Agreement", f"{float(blind.get('agreement', 0.0)):.3f}"),
    ]
    x = 0.06
    y = 0.52
    w = 0.205
    h = 0.17
    gap = 0.025
    for label, value in metrics:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.015",
                                    linewidth=1, edgecolor=border, facecolor=card))
        ax.text(x + 0.018, y + h - 0.035, label, fontsize=9.5, color=muted, va="top")
        ax.text(x + 0.018, y + 0.045, value, fontsize=16, fontweight="bold",
                color=ink, va="bottom")
        x += w + gap

    ax.add_patch(FancyBboxPatch((0.06, 0.17), 0.88, 0.24,
                                boxstyle="round,pad=0.012,rounding_size=0.015",
                                linewidth=1, edgecolor=border, facecolor=card))
    ax.text(0.08, 0.36, "Recovered message", fontsize=9.5, color=muted, va="top")
    ax.text(0.08, 0.285, report.get("recovered_text") or "(empty)",
            fontsize=18, fontweight="bold", color=ink, family="monospace", va="center")
    ax.text(0.08, 0.205, f"Text match: {match_text}", fontsize=10.5,
            color=green if text_match else muted, va="center")

    ax.text(0.06, 0.08,
            f"candidate={blind.get('candidate', '-')}"
            f"   |   n_obs={blind.get('n_obs', '-')}"
            f"   |   elapsed={report.get('elapsed_s', '-')} s"
            "   |   full metrics: report.json",
            fontsize=9, color=muted, va="center")

    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)


def public_base() -> str:
    base = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if base:
        return base.rstrip("/")
    return request.url_root.rstrip("/")


def cleanup_old_tasks(max_age_s: int = 24 * 3600) -> None:
    now = time.time()
    for d in OUTPUT_ROOT.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > max_age_s:
            shutil.rmtree(d, ignore_errors=True)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "lraw", "model": "author_trained.pkl"})


@app.route("/files/<task_id>/<path:name>", methods=["GET"])
def files(task_id: str, name: str):
    return send_from_directory(OUTPUT_ROOT / task_id, name, as_attachment=False)


@app.route("/process", methods=["POST", "OPTIONS"])
def process():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        cleanup_old_tasks()
        f = request.files.get("input_file") or request.files.get("audio")
        if f is None:
            raise ValueError("缺少音频字段 input_file")
        host = read_upload(f)
        raw_params = request.form.get("parameters", "{}")
        params = json.loads(raw_params) if isinstance(raw_params, str) else dict(raw_params)
        preset = str(params.get("attack_preset", "none"))
        if preset not in PRESETS:
            raise ValueError(f"未知 attack_preset：{preset}")
        attack, attack_label, attack_kw = PRESETS[preset]
        repeat = max(1, min(5, int(params.get("repeat", 1))))
        msg = str(params.get("message", "") or "").strip()
        if len(msg) > 40:
            raise ValueError("message 最多 40 个 ASCII 字符")
        if any(ord(c) > 127 for c in msg):
            raise ValueError("message 仅支持 ASCII 字符")

        t0 = time.time()
        bits_in = text_to_bits(msg) if msg else None
        emb = embed(host, bits_in, cfg, repeat=repeat)
        n = int(emb["n"])
        if bits_in is not None and bits_in.size > n:
            raise ValueError(f"水印文本过长：{bits_in.size} bit > 容量 {n} bit")
        wm = np.asarray(emb["audio_wm"], dtype=np.float64)
        attacked = apply_preset(attack, wm, cfg.sr, cfg, attack_kw)
        attacked = np.asarray(attacked, dtype=np.float64).ravel()
        if attacked.size < emb["seg_len"]:
            raise ValueError("攻击后音频过短，切不出 400 样本段")

        tempo = attack in ("tsm", "recapture", "hybrid")
        ext = extract_bits(attacked, MODEL, cfg, expected_len=wm.size,
                           tol=0.25 if tempo else 0.03, repeat=repeat)
        if ext["n"] == 0:
            raise ValueError("攻击后音频太短，无法提取")
        probs = np.asarray(ext["probs"], dtype=np.float64)
        cands = [("raw", np.asarray(ext["bits"]), probs)]
        if ext["n"] != n:
            ap = align_probs(probs, n)
            cands.append(("align", (ap >= 0.5).astype(np.int64), ap))

        best = None
        for name, cand, prob in cands:
            bc = blind_recover(cand, emb["key"], emb["L"], emb["taps"])
            if best is None or (bc["z"], bc["agreement"]) > (best[2]["z"], best[2]["agreement"]):
                best = (name, cand, bc)
        cand_name, cand_bits, bc = best
        raw_ber = ber(emb["bits"], cand_bits)
        W_hat = np.asarray(bc["W"])
        blind_ber = ber(emb["bits"], W_hat)
        recovered = bits_to_text(W_hat[:bits_in.size]) if bits_in is not None else ""

        task_id = uuid.uuid4().hex[:12]
        out_dir = OUTPUT_ROOT / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_audio(str(out_dir / "watermarked.wav"), wm, cfg.sr)
        write_audio(str(out_dir / "attacked.wav"), np.clip(attacked, -1.0, 1.0), cfg.sr)
        report = {
            "method": "LRAW",
            "attack_preset": preset,
            "attack": attack,
            "attack_label": attack_label,
            "attack_kwargs": attack_kw,
            "repeat": repeat,
            "n_bits": n,
            "n_seg_embed": int(emb["n_seg"]),
            "snr_embed_db": round(float(emb["snr_voiced"]), 2),
            "raw_ber": round(float(raw_ber), 6),
            "blind": {
                "locked": bool(bc["locked"]),
                "ber": round(float(blind_ber), 6),
                "agreement": round(float(bc["agreement"]), 6),
                "margin": round(float(bc["margin"]), 6),
                "phase_d": int(bc["phase_d"]),
                "n_obs": int(bc["n_obs"]),
                "threshold": round(float(bc["threshold"]), 6),
                "z": round(float(bc["z"]), 3),
                "candidate": cand_name,
            },
            "message": msg,
            "recovered_text": recovered,
            "text_match": bool(bits_in is not None and recovered == msg),
            "elapsed_s": round(time.time() - t0, 2),
            "notes": [
                "盲恢复只使用 K、L、taps，不借真值 W；结果以 locked 为准。",
                "重录为数字信道模型，不是 JBL+iPhone 物理实测。",
                "PSM 是方法层边界，BER 约 50% 属预期失败。",
            ],
        }
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            make_summary_png(out_dir / "summary.png", report)
        except Exception:
            pass
        try:
            make_waveform_png(out_dir / "waveform.png", host, wm, attacked)
        except Exception:
            pass

        base = public_base()
        artifacts = [
            {"kind": "audio", "label": "含水印音频", "filename": "watermarked.wav",
             "url": f"{base}/files/{task_id}/watermarked.wav"},
            {"kind": "audio", "label": "攻击后音频", "filename": "attacked.wav",
             "url": f"{base}/files/{task_id}/attacked.wav"},
        ]
        if (out_dir / "summary.png").exists():
            artifacts.append({"kind": "image", "label": "恢复结果摘要", "filename": "summary.png",
                              "url": f"{base}/files/{task_id}/summary.png"})
        artifacts.append({"kind": "json", "label": "恢复报告", "filename": "report.json",
                          "url": f"{base}/files/{task_id}/report.json"})
        if (out_dir / "waveform.png").exists():
            artifacts.append({"kind": "image", "label": "波形对比", "filename": "waveform.png",
                              "url": f"{base}/files/{task_id}/waveform.png"})
        return jsonify({"message": "LRAW 处理完成", "artifacts": artifacts})
    except json.JSONDecodeError:
        return jsonify({"error": "parameters 不是合法 JSON"}), 400
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
