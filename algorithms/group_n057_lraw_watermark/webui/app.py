# -*- coding: utf-8 -*-
"""LRAW 内容保护演示前端（Flask 单页应用，三阶段分步向导）。

交互流程（与论文 III 节的发送端 / 信道 / 接收端一一对应）：
    阶段 1 发送端：上传宿主 → 填文本/重复度 → 嵌入 → 得到含水印音频 + key.npz
    阶段 2 信道  ：选一种攻击并设参数 → 得到攻击后音频 + 失真指标
    阶段 3 接收端：双口径恢复 → 分别判定“盲恢复 / 论文口径”，再合成四态总判

三个阶段的中间状态都存在服务端（token 指向一份状态），前端只拿 token，
因此每一步的产物都能单独下载，也能按顺序推进而不必一次跑完。

页面只做展示，判定与计算都在服务端：
    - 盲恢复：只持 key.npz、不知道原水印，枚举相位看一致率（真实接收方口径）；
    - 论文口径：Algorithm 1，已知原水印 W，找连续 L 位全对锚点（条件性口径）。
    两者判据独立、互不包含，所以分别给出结果，不做短路合并：
        PASS（两者皆成）/ CONDITIONAL（仅论文口径）/ PASS_BLIND（仅盲）/ FAIL（皆败）。

启动：
    python webui/app.py            # http://127.0.0.1:5000
    python webui/app.py --port 8000
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_file          # noqa: E402

# 主链路接口：嵌入 / 提取 / 恢复 / 指标 / 配置
from lraw import (DEFAULT_CONFIG, read_audio, write_audio, embed,  # noqa: E402
                  extract_bits, recover_watermark, blind_recover, align_probs,
                  ber, snr)
from lraw.attacks import apply_attack                        # noqa: E402
from lraw.model import load_model                            # noqa: E402

app = Flask(__name__)
# 上传体积上限：100 MB（防止超大文件把内存吃满；12 s 单声道 wav 约 1 MB）
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

cfg = DEFAULT_CONFIG
MODEL = load_model()          # 进程级单例：LightGBM 加载一次即可复用

# token -> 一次实验的完整中间状态（三阶段共享）
#   wm / att : 含水印音频、攻击后音频（float64 一维数组，便于第二阶段直接施加攻击）
#   bits     : 逻辑水印比特（长度 n，含文本位 + 随机填充位）
#   key      : K = W ⊕ M（接收端唯一凭据）
#   wm_bytes / att_bytes / key_bytes : 对应下载产物（生成一次后缓存）
STASH: dict[str, dict] = {}
STASH_MAX = 8                # 只保留最近 8 次实验（每次约 8.5 MB，合计 <70 MB）


# ---------------------------------------------------------------------------
# 攻击元信息：前端下拉框与服务端默认参数共用一份定义（避免前后端口径不一致）
# ---------------------------------------------------------------------------
ATTACKS = {
    "none":     dict(label="无攻击（干净信道）", param=None, hint="基准：应恢复成功，恢复后 BER=0"),
    "gna":      dict(label="GNA 高斯噪声", param=dict(key="snr_db", min=5, max=40, step=1, val=25, unit="dB")),
    "mp3c":     dict(label="MP3C 压缩（LAME）", param=dict(key="bitrate", min=64, max=320, step=32, val=64, unit="kbps")),
    "am":       dict(label="AM 幅度衰减", param=dict(key="factor", min=0.3, max=1.0, step=0.05, val=0.8, unit="×")),
    "rs":       dict(label="RS 重采样 22.05 kHz", param=dict(key="target_sr", min=8000, max=44100, step=1000, val=22050, unit="Hz")),
    "lpf":      dict(label="LPF 低通滤波", param=dict(key="cutoff", min=3000, max=20000, step=500, val=8000, unit="Hz")),
    "mf":       dict(label="MF 中值滤波", param=dict(key="kernel", min=3, max=9, step=2, val=3, unit="点")),
    "rq":       dict(label="RQ 再量化", param=dict(key="bits", min=4, max=16, step=1, val=8, unit="bit")),
    "cp":       dict(label="CP 裁剪置零", param=dict(key="num", min=1, max=20, step=1, val=5, unit="处")),
    "tsm":      dict(label="TSM 时间伸缩（纯变速）", param=dict(key="factor", min=0.7, max=1.4, step=0.1, val=0.9, unit="×")),
    "psm":      dict(label="PSM 音高缩放（保时长）", param=dict(key="factor", min=0.7, max=1.4, step=0.1, val=0.9, unit="×")),
    "lsc":      dict(label="LSC 大比例裁剪", param=dict(key="ratio", min=0.1, max=0.9, step=0.1, val=0.5, unit="比例")),
    "recapture": dict(label="重录（数字信道模型）", param=dict(key="distance_m", min=0.5, max=13, step=0.5, val=1.0, unit="m")),
}

# 参数值是整数的键（其余按浮点处理）：避免把 22050 Hz 传成 22050.0 之类的口径差异
INT_KEYS = ("bitrate", "target_sr", "cutoff", "kernel", "bits", "num")


# ---------------------------------------------------------------------------
# 文本 < <-> 比特（8 bit ASCII，与论文“比特串即载荷”一致）
# ---------------------------------------------------------------------------
def text_to_bits(msg: str) -> np.ndarray:
    """把文本转成比特串（8 bit/字符）。调用前已校验只含 ASCII 可打印字符。"""
    raw = msg.encode("ascii", "ignore")
    return np.array([int(c) for c in "".join(f"{b:08b}" for b in raw)],
                    dtype=np.int64)


def bits_to_text(bits: np.ndarray) -> str:
    """把比特按 8 位一组还原为文本；非可打印字符显示为 '.'，避免页面出现乱码。"""
    b = np.asarray(bits, dtype=np.int64).ravel()
    k = (b.size // 8) * 8
    if k == 0:
        return ""
    out = []
    for row in b[:k].reshape(-1, 8):
        code = int("".join(map(str, row)), 2)
        out.append(chr(code) if 32 <= code < 127 else ".")
    return "".join(out)


def load_upload(file_storage) -> np.ndarray:
    """读取上传音频：转单声道，必要时重采样到 44.1 kHz（论文采样率口径）。"""
    data = file_storage.read()
    tmp = ROOT / "output" / f"_upload_{uuid.uuid4().hex}.wav"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(data)
    try:
        x, sr = read_audio(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)      # 临时文件用完就删，不污染 output 目录
    if sr != cfg.sr:
        from scipy import signal as sg
        x = sg.resample_poly(x, cfg.sr, sr)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# 三阶段的服务端实现（每个函数只负责一个阶段，便于 /api/run 复用）
# ---------------------------------------------------------------------------
def _state_new(**kw) -> str:
    """新建一份实验状态并返回 token；超出上限时按 FIFO 淘汰最旧的。"""
    tok = uuid.uuid4().hex[:12]
    STASH[tok] = kw
    if len(STASH) > STASH_MAX:
        for k in list(STASH)[:-STASH_MAX]:
            STASH.pop(k, None)
    return tok


def _get_state(token: str) -> dict:
    """取出状态；token 过期时抛 ValueError（前端会展示成“请重新执行上一步”）。"""
    st = STASH.get(token)
    if st is None:
        raise ValueError("结果已过期（服务端只保留最近 8 次），请从第 1 步重新开始")
    return st


def _parse_embed_form():
    """解析第 1 步的表单：宿主音频、水印文本、重复度 k，并做口径校验。"""
    f = request.files.get("audio")
    x = load_upload(f) if f else read_audio(str(ROOT / "data" / "music_01.wav"))[0]
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < cfg.sr * 3:
        raise ValueError("音频太短：至少需要 3 秒（论文口径 330 段）")
    msg = (request.form.get("message") or "").strip()
    if len(msg) > 40:
        raise ValueError("水印文本最多 40 个字符（40×8=320 bit，约需 3.6 s 音频）")
    if any(ord(c) > 127 for c in msg):
        raise ValueError("水印文本仅支持 ASCII（英文/数字/符号），中文无法编码")
    repeat = _repeat_or_err(request.form.get("repeat"))
    return x, msg, repeat


def _repeat_or_err(raw) -> int:
    """把表单里的 k 解析成 1–5 的整数；非法输入给出中文提示而不是 500。"""
    try:
        v = int(raw or 1)
    except (TypeError, ValueError):
        raise ValueError("重复度 k 必须是 1/3/5 这样的整数")
    return max(1, min(5, v))


def _do_embed(x: np.ndarray, msg: str, repeat: int) -> tuple[dict, dict]:
    """阶段 1：嵌入。返回 (状态, 给前端的 payload)。"""
    t0 = time.time()
    bits_in = text_to_bits(msg) if msg else None
    emb = embed(x, bits_in, cfg, repeat=repeat)
    n = int(emb["n"])
    if bits_in is not None and bits_in.size > n:
        raise ValueError(f"水印文本过长：{bits_in.size} bit > 容量 {n} bit"
                         f"（约 {n // 8} 个字符），请缩短文本或换更长的音频")
    wm = np.asarray(emb["audio_wm"], dtype=np.float64)
    n_seg = int(emb["n_seg"])
    text_bits = int(bits_in.size) if bits_in is not None else 0
    fill_bits = n - text_bits                     # 文本之后的随机填充位数

    # 下载产物：含水印 wav + 密钥包（各生成一次，后续复用）
    wb = io.BytesIO()
    write_audio(wb, wm, cfg.sr)
    kb = io.BytesIO()
    np.savez(kb, key=emb["key"], W=emb["bits"], L=emb["L"],
             taps=np.asarray(emb["taps"]), n=n, repeat=repeat,
             seg_len=emb["seg_len"], msg_n=text_bits, sr=cfg.sr,
             expected_len=int(wm.size))

    st = dict(kind="lraw", wm=wm, att=None, bits=np.asarray(emb["bits"]),
              key=np.asarray(emb["key"]), L=int(emb["L"]),
              taps=list(emb["taps"]), n=n, repeat=repeat, seg_len=int(emb["seg_len"]),
              msg=msg, msg_n=text_bits, host_len=int(x.size), sr=cfg.sr,
              snr_embed=float(emb["snr_voiced"]), n_seg_embed=n_seg,
              attack=None, kw=None,
              wm_bytes=wb.getvalue(), att_bytes=None, key_bytes=kb.getvalue())
    tok = _state_new(**st)

    payload = {
        "token": tok,
        "host": {"samples": int(x.size), "sr": cfg.sr,
                 "duration_s": round(x.size / cfg.sr, 2)},
        "metrics": {
            "n_bits": n,                       # 逻辑载荷比特（= 容量，k=1 时每段 1 bit）
            "n_seg_embed": n_seg,              # 实际写入的段数（= n × k）
            "text_bits": text_bits,            # 文本占用的比特数（11 字符 = 88 bit）
            "fill_bits": fill_bits,            # 随机填充的比特数
            "capacity_bits": n,                # 容量（bit）—— 修正旧版 n×8 的错误显示
            "capacity_chars": n // 8,          # 容量（ASCII 字符）
            "snr_embed": round(float(emb["snr_voiced"]), 2),
            "L": int(emb["L"]), "seg_len": int(emb["seg_len"]), "repeat": repeat,
            "elapsed": round(time.time() - t0, 2),
        },
        "text": msg,
        "downloads": {
            "下载含水印音频": f"/download/{tok}/wm",
            "下载密钥与真值 key.npz": f"/download/{tok}/key",
        },
        "detail_table": (
            "<tr><th>项目</th><th class='n'>值</th></tr>"
            f"<tr><td>宿主时长 / 采样率</td><td class='n'>{x.size / cfg.sr:.2f} s · {cfg.sr} Hz</td></tr>"
            f"<tr><td>段长 / 重复度</td><td class='n'>{emb['seg_len']} 样本 · k={repeat}</td></tr>"
            f"<tr><td>载荷比特 / 嵌入段数</td><td class='n'>{n} bit · {n_seg} 段</td></tr>"
            f"<tr><td>文本位 / 随机填充位</td><td class='n'>{text_bits} bit · {fill_bits} bit</td></tr>"
            f"<tr><td>m 序列阶数 L / 抽头</td><td class='n'>{emb['L']} · {emb['taps']}</td></tr>"
            f"<tr><td>嵌入 SNR（本文件有声段）</td><td class='n'>{float(emb['snr_voiced']):.2f} dB"
            f"（论文要求 &gt;20 dB）</td></tr>"
        ),
    }
    return STASH[tok], payload


def _do_attack(st: dict, atk: str, param) -> dict:
    """阶段 2：施加攻击。返回给前端的 payload（状态内联更新）。"""
    t0 = time.time()
    if atk not in ATTACKS:
        raise ValueError(f"未知攻击 {atk}")
    meta = ATTACKS[atk]["param"]
    kw = {}
    if meta:
        raw = param if param not in (None, "", "null") else meta["val"]
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"参数 {meta['key']} 必须是数字")
        # 范围校验：前端虽然用滑块限制，但手动键入可以越界，服务端必须兜住
        if not (meta["min"] <= val <= meta["max"]):
            raise ValueError(f"{meta['key']} 超出范围 [{meta['min']}, {meta['max']}]")
        kw[meta["key"]] = int(round(val)) if meta["key"] in INT_KEYS else val

    wm = st["wm"]
    y = wm if atk == "none" else apply_attack(atk, wm, cfg.sr, cfg, **kw)
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < st["seg_len"]:
        raise ValueError("攻击后音频过短，切不出任何 400 样本段")

    ab = io.BytesIO()
    write_audio(ab, np.clip(y, -1.0, 1.0), cfg.sr)      # 写盘前限幅，避免破音
    st.update(att=y, att_bytes=ab.getvalue(), attack=atk, kw=kw)

    # 攻击后 SNR：以含水印音频为参考；注意它包含整体电平变化（如重录的 1/d 衰减）
    snr_att = float(snr(wm, y))
    payload = {
        "token": request_token(st),
        "attack": {"key": atk, "label": ATTACKS[atk]["label"], "kwargs": kw,
                   "hint": ATTACKS[atk].get("hint", "")},
        "metrics": {
            "snr_att": (round(snr_att, 2) if np.isfinite(snr_att) else None),
            "len_ratio": round(y.size / wm.size, 4),
            "len_after_s": round(y.size / cfg.sr, 2),
            "elapsed": round(time.time() - t0, 2),
        },
        "downloads": {"下载攻击后音频": f"/download/{request_token(st)}/att"},
        "detail_table": (
            "<tr><th>项目</th><th class='n'>值</th></tr>"
            f"<tr><td>攻击</td><td class='n'>{ATTACKS[atk]['label']} {kw if kw else ''}</td></tr>"
            f"<tr><td>攻击后时长比</td><td class='n'>{y.size / wm.size:.4f}</td></tr>"
            f"<tr><td>攻击后 SNR（含电平变化）</td><td class='n'>"
            f"{snr_att:.2f} dB</td></tr>"
        ),
    }
    return payload


def request_token(st: dict) -> str:
    """反查状态对应的 token（状态对象里不存 token 本身，避免循环引用）。"""
    for k, v in STASH.items():
        if v is st:
            return k
    raise ValueError("内部错误：状态未登记")


def _do_recover(st: dict) -> dict:
    """阶段 3：双口径恢复与判定。

    两个判据彼此独立、互不包含，所以分别计算、分别展示：
      - 盲恢复  ：只持密钥，枚举相位，用全序列一致率决定是否锁定（真实接收方）；
      - 论文口径：Algorithm 1，已知原水印 W，找连续 L 位全对锚点（条件性口径）。
    总判四态：PASS / CONDITIONAL / PASS_BLIND / FAIL，不做“论文口径失败即整体失败”的短路。
    """
    t0 = time.time()
    if st.get("att") is None:
        raise ValueError("请先完成第 2 步（施加攻击），再进行恢复")

    atk = st["attack"]
    # 时基归一：变速/重录是“整体时长变化”，可按已知载荷时长重采样回原长；
    # 裁剪类是真实丢内容，不能归一（否则会把“删段”误当“变速”）
    tempo = atk in ("tsm", "recapture")
    ext = extract_bits(st["att"], MODEL, cfg, expected_len=st["wm"].size,
                       tol=0.25 if tempo else 0.03, repeat=st["repeat"])
    if ext["n"] == 0:
        raise ValueError("攻击后音频过短，无法切出任何 400 样本段")

    # 候选：原始网格的判决；段数变化时再加一个“概率插值回原长”的候选。
    # 每个候选带 bits 与 probs：probs 用于构造对齐候选；盲恢复判据当前用硬判决
    #（概率未校准，软打分实测略降分离度），保留 probs 供后续校准实验。
    probs = np.asarray(ext["probs"], dtype=np.float64)
    cands = [("raw", np.asarray(ext["bits"]), probs)]
    if ext["n"] != st["n"]:
        ap = align_probs(probs, st["n"])
        cands.append(("align", (ap >= 0.5).astype(np.int64), ap))

    # 候选选择必须走**盲口径**（只看相位显著性），不能用真值挑 —— 旧版用
    # min(BER) 等于偷看真值，会把鲁棒性算高，这里已修正。
    # 用 z（相对随机一致率的显著性）而不是裸一致率：裁剪后只有几十位的候选
    # 一致率容易虚高，但证据强度其实不如长候选，按 z 挑可避免短候选假锁定。
    best = None
    for name, c, _ in cands:
        bc = blind_recover(c, st["key"], st["L"], st["taps"])
        if best is None or (bc["z"], bc["agreement"]) > (best[2]["z"], best[2]["agreement"]):
            best = (name, c, bc)
    cand_name, cand_bits, bc = best

    raw_ber = ber(st["bits"], cand_bits)        # 裸 BER：同一候选，不纠错不归一
    W_hat = np.asarray(bc["W"])
    # 恢复后 BER 只用于展示与自检，**不参与成功判定**（否则等于用真值判卷）
    blind_ber = ber(st["bits"], W_hat)
    blind_ok = bool(bc["locked"])

    # 论文口径（Algorithm 1）：需要原始 W，演示模式由 key.npz 提供真值；
    # 真实部署若密钥不含 W，该口径应显示“不可用”，而不是伪造成功。
    oracle_available = st.get("bits") is not None
    oracle_ok, oracle_anchor, oracle_ber = False, None, None
    if oracle_available:
        ro = recover_watermark(st["bits"], cands[0][1], st["key"], st["L"], st["taps"])
        oracle_anchor = ro.get("anchor")
        oracle_ok = oracle_anchor is not None
        oracle_ber = ro.get("ber_after")

    # 四态总判：两判据独立，谁也不短路谁
    if blind_ok and oracle_ok:
        state = "PASS"
    elif oracle_ok:
        state = "CONDITIONAL"
    elif blind_ok:
        state = "PASS_BLIND"
    else:
        state = "FAIL"

    msg, msg_n = st["msg"], st["msg_n"]
    dec = bits_to_text(W_hat[:msg_n]) if msg_n else ""
    ok_txt = bool(msg_n and dec == msg)

    # ---- 盲恢复卡片文案 ----
    n_obs = int(bc.get("n_obs", 0))
    thr = float(bc.get("threshold", min(1.0, 0.5 + 2.75 / np.sqrt(max(n_obs, 1)))))
    zval = float(bc.get("z", 0.0))
    if blind_ok:
        btxt = (f"用实际观测到的 {n_obs} 位打分（自适应阈值 {thr*100:.1f}%），"
                f"相位 d={bc['phase_d']}，一致率 {bc['agreement']*100:.1f}%"
                f"（领先次优 {bc['margin']*100:.1f} 个百分点，z={zval:.1f}σ）。")
        if blind_ber > 0.05:
            btxt += "注意：已锁定但与真值 BER 偏高，可能锁到错误相位（真实系统需 CRC 才能发现）。"
        elif msg_n:
            btxt += "文本自检：" + ("与原始一致 ✔" if ok_txt else "与原始不一致 ✘")
    else:
        if n_obs < 20:
            btxt = (f"未锁定：只观测到 {n_obs} 位，低于盲恢复最少 20 位的要求"
                    f"（13 位虽能定相位，但 1323 个候选里会碰巧命中，虚警约 15%）。")
        else:
            btxt = (f"未锁定：实际观测到 {n_obs} 位，一致率 {bc['agreement']*100:.1f}%、"
                    f"margin {bc['margin']*100:.1f} 个百分点，低于自适应阈值 {thr*100:.1f}% "
                    f"或 margin 不足 3 个百分点；裸 BER {raw_ber*100:.2f}%。")

    # ---- 论文口径卡片文案 ----
    if not oracle_available:
        otxt = "密钥不含原始水印真值，该口径不可用。"
    elif oracle_ok:
        otxt = (f"在原始提取流中找到连续 {st['L']} 位全对锚点（第 {oracle_anchor} 位起），"
                f"恢复后 BER {(oracle_ber or 0.0)*100:.2f}%。该口径需要已知原水印 W。")
    else:
        otxt = (f"未找到连续 {st['L']} 位全对窗口，论文口径失败——"
                "残余证据不足以支撑已知水印的锚点匹配。")

    verdict_text = {
        "PASS": "两种口径均成功：盲恢复已锁定，论文口径也找到锚点。",
        "CONDITIONAL": "论文口径成功，但盲恢复未锁定：仅“已知原水印”可恢复，盲提取失败（不是整体恢复失败）。",
        "PASS_BLIND": "盲恢复成功，但论文口径未找到锚点：两判据不同，属极少见情形。",
        "FAIL": "两种口径均失败：水印已无法可靠恢复。",
    }[state]

    # 逐位错误图：最多展示前 400 位，红点=错
    m = min(st["bits"].size, W_hat.size)
    err_map = [int(v) for v in (np.asarray(st["bits"])[:m] != W_hat[:m]).astype(int)][:400]

    payload = {
        "token": request_token(st),
        # 向后兼容：旧字段 success 的语义固定为“盲恢复是否成功”
        "success": blind_ok,
        "verdict": state,
        "verdict_text": verdict_text,
        "blind": {
            "label": "盲恢复 · 真实接收方（只持密钥）",
            "success": blind_ok,
            "locked": bool(bc["locked"]),
            "agree": round(bc["agreement"] * 100, 1),
            "margin": round(bc["margin"] * 100, 1),
            "phase_d": int(bc["phase_d"]),
            "n_obs": n_obs,
            "threshold": round(thr * 100, 1),
            "min_obs": 20,
            "z": round(zval, 1),
            "ber": round(blind_ber * 100, 2),
            "text": btxt,
        },
        "oracle": {
            "label": "论文口径 · 已知原水印（Algorithm 1）",
            "available": bool(oracle_available),
            "success": bool(oracle_ok),
            "hit": bool(oracle_ok),
            "anchor": (int(oracle_anchor) if oracle_anchor is not None else None),
            "ber": (round(float(oracle_ber) * 100, 2) if oracle_ber is not None else None),
            "text": otxt,
        },
        "metrics": {
            "raw_ber": round(raw_ber * 100, 2),      # 裸 BER（%）
            "rec_ber": round(blind_ber * 100, 2),    # 恢复后 BER（%，仅自检）
            "agree": round(bc["agreement"] * 100, 1),
            "margin": round(bc["margin"] * 100, 1),
            "phase_d": int(bc["phase_d"]),
            "n_seg_rx": int(ext["n_seg"]),
            "n_bits_rx": int(ext["n"]),
            "blind_n_obs": n_obs,
            "blind_threshold": round(thr * 100, 1),
            "blind_z": round(zval, 1),
            "oracle_hit": bool(oracle_ok),
            "oracle_anchor": (int(oracle_anchor) if oracle_anchor is not None else None),
            "oracle_ber": (round(float(oracle_ber) * 100, 2) if oracle_ber is not None else None),
            "elapsed": round(time.time() - t0, 2),
        },
        "message_html": (
            f"<b>恢复出的水印文本</b>：<span class='mono'>{dec!r}</span><br>"
            f"<b>原始文本</b>：<span class='mono'>{msg!r}</span>"
            f"<br><span class='hintline'>文本比对使用演示环境中的真值，仅供自检，不参与成功判定。</span>"
            if msg_n else
            "<b>未指定文本水印</b>：本次为随机比特载荷，只校验比特恢复准确率。"
        ),
        "error_map": err_map,
        "downloads": {"下载密钥与真值 key.npz": f"/download/{request_token(st)}/key"},
        "detail_table": (
            "<tr><th>项目</th><th class='n'>值</th></tr>"
            f"<tr><td>攻击</td><td class='n'>{ATTACKS[atk]['label']} {st['kw'] or ''}</td></tr>"
            f"<tr><td>判决候选</td><td class='n'>{cand_name}"
            f"（{'概率插值回原长' if cand_name == 'align' else '原始网格'}）</td></tr>"
            f"<tr><td>接收段数 / 载荷比特</td><td class='n'>{ext['n_seg']} 段 · {ext['n']} bit</td></tr>"
            "<tr><td colspan='2'><b>盲恢复（真实接收方：只持密钥）</b></td></tr>"
            f"<tr><td>锁定 / 相位 d</td><td class='n'>{bc['locked']} · {bc['phase_d']}</td></tr>"
            f"<tr><td>观测位数 / 锁定阈值</td><td class='n'>{n_obs} 位 · {thr*100:.1f}%"
            f"（最少 20 位，5.5σ）</td></tr>"
            f"<tr><td>一致率 / margin / z</td><td class='n'>{bc['agreement']*100:.2f}% · "
            f"{bc['margin']*100:.2f} 个百分点 · {zval:.1f}σ</td></tr>"
            f"<tr><td>恢复后 BER（对照真值自检）</td><td class='n'>{blind_ber*100:.2f}%</td></tr>"
            "<tr><td colspan='2'><b>论文口径（Algorithm 1：已知原水印）</b></td></tr>"
            f"<tr><td>锚点</td><td class='n'>"
            f"{('第 ' + str(oracle_anchor) + ' 位') if oracle_anchor is not None else ('不可用' if not oracle_available else '未命中')}</td></tr>"
            f"<tr><td>恢复后 BER</td><td class='n'>"
            f"{(f'{oracle_ber*100:.2f}%' if oracle_ber is not None else '—')}</td></tr>"
            f"<tr><td>总判</td><td class='n'>{state}</td></tr>"
        ),
    }
    return payload


# ---------------------------------------------------------------------------
# 页面（三阶段向导：每步控件 + 该步结果，互不干扰）
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LRAW 音频水印 · 嵌入 / 攻击 / 恢复 三步演示</title>
<style>
  :root{
    --bg:#f5f7fa; --panel:#fff; --ink:#1f2328; --muted:#6b7280;
    --line:#e5e7eb; --accent:#0f766e; --accent-2:#0d9488;
    --ok:#15803d; --ok-bg:#ecfdf3; --bad:#b91c1c; --bad-bg:#fef2f2;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.65 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
  header{padding:20px 28px 16px;background:linear-gradient(135deg,#0f766e,#115e59);color:#fff}
  header h1{margin:0;font-size:19px;font-weight:600;letter-spacing:.3px}
  header p{margin:6px 0 0;font-size:12.5px;opacity:.85}
  .steps{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
  .step{font-size:12px;padding:5px 12px;border-radius:999px;background:rgba(255,255,255,.16);
        color:#e6fffb;border:1px solid rgba(255,255,255,.28)}
  .step.on{background:#fff;color:#0f766e;font-weight:700;border-color:#fff}
  .step.done{background:#ccfbf1;color:#115e59;font-weight:600;border-color:#ccfbf1}
  main{max-width:1000px;margin:20px auto 40px;padding:0 18px;display:flex;flex-direction:column;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;
        box-shadow:0 1px 2px rgba(16,24,40,.04)}
  .card h2{margin:0 0 4px;font-size:15px;font-weight:600;display:flex;align-items:center;gap:8px}
  .card h2 span{display:inline-flex;width:21px;height:21px;border-radius:50%;background:var(--accent);
        color:#fff;font-size:12px;align-items:center;justify-content:center}
  .sub{font-size:12px;color:var(--muted);margin:0 0 14px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px 14px}
  label{display:block;font-size:12px;color:var(--muted);margin:10px 0 5px}
  input[type=file],input[type=text],input[type=number],select{
    width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:9px;
    background:#fff;font:13px inherit;color:var(--ink)}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .prow{display:grid;grid-template-columns:1fr 100px;gap:8px}
  .btn{margin-top:14px;padding:11px 18px;border:0;border-radius:10px;background:var(--accent);
       color:#fff;font-size:14px;font-weight:600;cursor:pointer}
  .btn:hover:not(:disabled){background:var(--accent-2)}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn.ghost{background:#fff;color:var(--accent);border:1px solid var(--accent);font-weight:600}
  .hintline{font-size:12px;color:var(--muted);margin-top:6px}
  .res{margin-top:16px;border-top:1px dashed var(--line);padding-top:14px}
  .empty{font-size:12.5px;color:var(--muted);background:#f8fafc;border:1px dashed var(--line);
         border-radius:10px;padding:14px}
  .verdict{padding:14px 16px;border-radius:12px;display:flex;align-items:center;gap:14px;margin-bottom:12px}
  .verdict.ok{background:var(--ok-bg);border:1px solid #bbf7d0}
  .verdict.bad{background:var(--bad-bg);border:1px solid #fecaca}
  .verdict.warn{background:#fffbeb;border:1px solid #fde68a}
  .badge{font-size:13px;font-weight:700;padding:6px 12px;border-radius:999px;white-space:nowrap}
  .badge.ok{background:#dcfce7;color:var(--ok)} .badge.bad{background:#fee2e2;color:var(--bad)}
  .badge.warn{background:#fef3c7;color:#92400e}
  .duo{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:12px}
  .mini{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:#fcfcfd}
  .mini.ok{border-color:#bbf7d0;background:var(--ok-bg)}
  .mini.bad{border-color:#fecaca;background:var(--bad-bg)}
  .mini.off{opacity:.6}
  .mini h3{margin:0 0 2px;font-size:13.5px;display:flex;align-items:center;gap:8px;justify-content:space-between}
  .mini .msub{font-size:11.5px;color:var(--muted);margin-bottom:8px}
  .mini .kpis{grid-template-columns:repeat(auto-fill,minmax(110px,1fr))}
  .mini .kpi{padding:7px 9px}
  .mini .kpi .v{font-size:15px}
  .mini .vtext{font-size:12px;color:#374151;margin-top:8px}
  .vtext{font-size:12.5px;color:#374151}
  .kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:8px}
  .kpi{border:1px solid var(--line);border-radius:10px;padding:9px 11px;background:#fcfcfd}
  .kpi .k{font-size:11px;color:var(--muted)}
  .kpi .v{font-size:17px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
  .kpi .v small{font-size:11px;font-weight:500;color:var(--muted);margin-left:3px}
  .mono{font-family:ui-monospace,Consolas,"Courier New",monospace}
  .msgbox{margin-top:12px;padding:11px;border-radius:10px;background:#f8fafc;border:1px solid var(--line);
          word-break:break-all;font-size:13px}
  table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px}
  th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}
  th{color:var(--muted);font-weight:500;font-size:12px}
  th.n{text-align:right}
  td.n{text-align:right;font-variant-numeric:tabular-nums}
  .bits{display:flex;gap:3px;flex-wrap:wrap;margin-top:8px}
  .bit{width:9px;height:9px;border-radius:2px;background:#d1fae5}
  .bit.e{background:#ef4444}
  .dl{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
  .dl a{flex:1;min-width:160px;text-align:center;text-decoration:none;padding:9px 12px;
        border:1px solid var(--line);border-radius:9px;color:var(--accent);font-size:13px;background:#fff}
  .dl a:hover{background:#f0fdfa}
  .foot{font-size:11.5px;color:var(--muted);margin-top:10px}
  .err{color:var(--bad);font-size:13px;margin-top:8px}
  .warn{color:#92400e;font-size:12.5px;margin-top:8px}
</style>
</head>
<body>
<header>
  <h1>LRAW 音频水印演示 · 嵌入 → 攻击 → 恢复</h1>
  <p>DWT-GBT-SVD 奇异值量化（a=5, b=2）+ LightGBM 判决 + m 序列滑窗恢复｜IEEE TIFS 2025 复现</p>
  <div class="steps">
    <div class="step on" id="st1">① 发送端嵌入</div>
    <div class="step" id="st2">② 信道攻击</div>
    <div class="step" id="st3">③ 接收端恢复</div>
  </div>
</header>

<main>
  <!-- ================= 阶段 1：嵌入 ================= -->
  <section class="card">
    <h2><span>1</span>发送端：嵌入水印</h2>
    <p class="sub">选宿主音频 → 填水印文本（可选）→ 选择重复度 → 生成含水印音频与密钥包。这一步只做嵌入，不施加任何攻击。</p>
    <div class="grid">
      <div>
        <label>宿主音频（wav / flac；建议 ≥3 s，44.1 kHz）</label>
        <input type="file" id="audio" accept=".wav,.flac,.mp3,.ogg">
        <div class="hintline">不上传则用内置示例 <a href="/api/sample" style="color:var(--accent)">music_01.wav</a>（12 s 真实音乐）</div>
      </div>
      <div>
        <label>水印文本（可选，仅 ASCII；留空=随机比特）</label>
        <input type="text" id="message" placeholder="例如: LRAW-2025-demo" maxlength="40">
        <div class="hintline">11 个字符 = 88 bit；剩余容量自动用随机比特补齐</div>
      </div>
      <div>
        <label>重复度 k（每比特写入 k 段）</label>
        <select id="repeat">
          <option value="1" selected>k = 1（论文口径，每段 1 比特）</option>
          <option value="3">k = 3（容量 1/3，抗强攻击）</option>
          <option value="5">k = 5（容量 1/5，最强鲁棒）</option>
        </select>
        <div class="hintline">k&gt;1 时同一比特连写 k 段，提取端概率域软融合</div>
      </div>
    </div>
    <button class="btn" id="b1">① 生成含水印音频</button>
    <div class="err" id="e1"></div>

    <div class="res" id="r1"><div class="empty">完成第 1 步后，这里显示嵌入指标（载荷比特、嵌入 SNR、容量）与下载入口。</div></div>
  </section>

  <!-- ================= 阶段 2：攻击 ================= -->
  <section class="card">
    <h2><span>2</span>信道：施加攻击</h2>
    <p class="sub">在第 1 步产物上施加一种攻击（可设参数），得到攻击后音频与失真指标。第 2 步之前必须先完成第 1 步。</p>
    <div class="grid">
      <div>
        <label>攻击方式</label>
        <select id="attack"></select>
        <div class="hintline" id="attackHint"></div>
      </div>
      <div id="paramBox" style="display:none">
        <label id="paramLabel">参数</label>
        <div class="prow">
          <input type="range" id="paramRange">
          <input type="number" id="paramNum">
        </div>
      </div>
    </div>
    <button class="btn" id="b2" disabled>② 施加攻击</button>
    <div class="err" id="e2"></div>

    <div class="res" id="r2"><div class="empty">完成第 2 步后，这里显示攻击档位、攻击后 SNR、时长比与攻击后音频下载。</div></div>
  </section>

  <!-- ================= 阶段 3：恢复 ================= -->
  <section class="card">
    <h2><span>3</span>接收端：双口径恢复与判定</h2>
    <p class="sub">同时给出两个独立判据：① 盲恢复（只持 key.npz、不知道原水印，真实接收方口径）；② 论文口径（Algorithm 1，已知原水印找 13 位全对锚点）。两者分别判定，再做四态总判。</p>
    <button class="btn" id="b3" disabled>③ 恢复与判定</button>
    <div class="err" id="e3"></div>

    <div class="res" id="r3"><div class="empty">完成第 3 步后，这里并排显示“盲恢复”和“论文口径”两张结果卡、四态总判、逐位判决图与下载。</div></div>
  </section>

</main>

<script>
const ATTACKS = __ATTACKS__;
const S = {token: null, stage: 0};       // stage: 0 未开始 / 1 已嵌入 / 2 已攻击 / 3 已恢复

/* ---- 攻击下拉框与参数联动 ---- */
const sel = document.getElementById('attack');
for (const [k, v] of Object.entries(ATTACKS)) {
  const o = document.createElement('option');
  o.value = k; o.textContent = v.label; sel.appendChild(o);
}
const pBox = document.getElementById('paramBox');
const pRange = document.getElementById('paramRange');
const pNum = document.getElementById('paramNum');
const pLabel = document.getElementById('paramLabel');
const pHint = document.getElementById('attackHint');

function syncParam() {
  const meta = ATTACKS[sel.value];
  pHint.textContent = meta.hint || '';
  if (!meta.param) { pBox.style.display = 'none'; return; }
  const p = meta.param;
  pBox.style.display = 'block';
  pLabel.textContent = `${p.key}（${p.unit}）`;
  pRange.min = p.min; pRange.max = p.max; pRange.step = p.step; pRange.value = p.val;
  pNum.min = p.min; pNum.max = p.max; pNum.step = p.step; pNum.value = p.val;
}
sel.addEventListener('change', syncParam);
pRange.addEventListener('input', () => pNum.value = pRange.value);
pNum.addEventListener('input', () => { pRange.value = pNum.value; });
syncParam();

/* ---- 小工具 ---- */
const EMPTY2 = '<div class="empty">完成第 2 步后，这里显示攻击档位、攻击后 SNR、时长比与攻击后音频下载。</div>';
const EMPTY3 = '<div class="empty">完成第 3 步后，这里显示恢复判定、原始/恢复后 BER、一致率与逐位判决图。</div>';

/* 把“当前推进到第几阶段”落成界面状态：n=1/2/3，严格设定（不是取最大） */
function setStage(n) {
  S.stage = n;
  ['st1','st2','st3'].forEach((id, i) => {
    const el = document.getElementById(id);
    el.className = 'step' + (n > i + 1 ? ' done' : (n === i + 1 ? ' on' : ''));
  });
  document.getElementById('b2').disabled = n < 1;
  document.getElementById('b3').disabled = n < 2;
}

/* 重新执行上游步骤时，把下游的旧结果清掉，避免“新 token 配旧结果” */
function clearDownstream(from) {
  if (from <= 1) document.getElementById('r2').innerHTML = EMPTY2;
  if (from <= 2) document.getElementById('r3').innerHTML = EMPTY3;
}
function num(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'string') return v;        // 攻击名称、组合值等文本直接展示
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}
function kpis(el, defs) {
  el.innerHTML = '';
  for (const [label, val, unit] of defs) {
    if (val === null || val === undefined) continue;
    const d = document.createElement('div');
    d.className = 'kpi';
    d.innerHTML = `<div class="k">${label}</div><div class="v">${num(val)}` +
                  `<small>${unit || ''}</small></div>`;
    el.appendChild(d);
  }
}
function downloads(el, map) {
  el.innerHTML = '';
  for (const [label, href] of Object.entries(map || {})) {
    const a = document.createElement('a');
    a.href = href; a.textContent = label; el.appendChild(a);
  }
}
async function post(url, fd) {
  const r = await fetch(url, {method: 'POST', body: fd});
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

/* ---- 阶段 1 ---- */
document.getElementById('b1').addEventListener('click', async () => {
  const btn = document.getElementById('b1'), err = document.getElementById('e1');
  err.textContent = ''; btn.disabled = true; btn.textContent = '嵌入中…';
  try {
    const fd = new FormData();
    const f = document.getElementById('audio').files[0];
    if (f) fd.append('audio', f);
    fd.append('message', document.getElementById('message').value);
    fd.append('repeat', document.getElementById('repeat').value);
    const j = await post('/api/embed', fd);
    S.token = j.token;
    setStage(1); clearDownstream(1);      // 新一次嵌入 → 后面两步的结果作废
    const m = j.metrics;
    const box = document.getElementById('r1');
    box.innerHTML = `<div class="kpis" id="k1"></div>
      <div class="dl" id="d1"></div>
      <table id="t1"></table>`;
    kpis(document.getElementById('k1'), [
      ['载荷比特', m.n_bits, 'bit'],
      ['嵌入段数', m.n_seg_embed, '段'],
      ['嵌入 SNR（本文件）', m.snr_embed, 'dB'],
      ['容量', m.capacity_bits, 'bit'],
      ['容量（字符）', m.capacity_chars, '字符'],
      ['文本位 / 填充位', m.text_bits + ' / ' + m.fill_bits, 'bit'],
      ['m 序列阶数 L', m.L, ''],
      ['重复度 k', m.repeat, ''],
      ['耗时', m.elapsed, 's']
    ]);
    downloads(document.getElementById('d1'), j.downloads);
    document.getElementById('t1').innerHTML = j.detail_table;
    if (m.snr_embed < 20) {
      box.insertAdjacentHTML('beforeend',
        `<div class="warn">嵌入 SNR ${m.snr_embed} dB 低于论文 20 dB 阈值，请检查宿主或调整段长。</div>`);
    }
  } catch (e) {
    err.textContent = '嵌入失败：' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '① 生成含水印音频';
  }
});

/* ---- 阶段 2 ---- */
document.getElementById('b2').addEventListener('click', async () => {
  const btn = document.getElementById('b2'), err = document.getElementById('e2');
  if (!S.token) { err.textContent = '请先完成第 1 步'; return; }
  err.textContent = ''; btn.disabled = true; btn.textContent = '攻击中…';
  try {
    const fd = new FormData();
    fd.append('token', S.token);
    fd.append('attack', sel.value);
    if (ATTACKS[sel.value].param) fd.append('param', pNum.value);
    const j = await post('/api/attack', fd);
    setStage(2); clearDownstream(2);      // 换了攻击 → 第 3 步结果作废
    const m = j.metrics, a = j.attack;
    const box = document.getElementById('r2');
    box.innerHTML = `<div class="kpis" id="k2"></div><div class="dl" id="d2"></div><table id="t2"></table>`;
    kpis(document.getElementById('k2'), [
      ['攻击档位', a.label, ''],
      ['攻击后 SNR', m.snr_att, 'dB'],
      ['时长比', m.len_ratio, '×'],
      ['攻击后时长', m.len_after_s, 's'],
      ['耗时', m.elapsed, 's']
    ]);
    downloads(document.getElementById('d2'), j.downloads);
    document.getElementById('t2').innerHTML = j.detail_table;
  } catch (e) {
    err.textContent = '攻击失败：' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '② 施加攻击';
  }
});

/* ---- 阶段 3 ---- */
document.getElementById('b3').addEventListener('click', async () => {
  const btn = document.getElementById('b3'), err = document.getElementById('e3');
  if (!S.token) { err.textContent = '请先完成第 1、2 步'; return; }
  err.textContent = ''; btn.disabled = true; btn.textContent = '恢复中…';
  try {
    const fd = new FormData();
    fd.append('token', S.token);
    const j = await post('/api/recover', fd);
    setStage(3);
    const m = j.metrics, b = j.blind, o = j.oracle;
    const box = document.getElementById('r3');
    const vsCls = j.verdict === 'FAIL' ? 'bad' : (j.verdict === 'CONDITIONAL' ? 'warn' : 'ok');
    const vsTxt = ({PASS:'可恢复', CONDITIONAL:'条件性可恢复', PASS_BLIND:'可恢复（盲）', FAIL:'恢复失败'})[j.verdict] || '结论未知';
    const bCls = b.success ? 'ok' : 'bad';
    const oCls = !o.available ? 'off' : (o.success ? 'ok' : 'bad');
    const oTxt = !o.available ? '不可用' : (o.success ? '成功' : '失败');
    const oBadge = !o.available ? 'warn' : (o.success ? 'ok' : 'bad');
    box.innerHTML = `
      <div class="verdict ${vsCls}">
        <span class="badge ${vsCls}">${vsTxt}</span>
        <span class="vtext">${j.verdict_text}</span>
      </div>
      <div class="duo">
        <div class="mini ${bCls}">
          <h3>盲恢复 <span class="badge ${bCls}">${b.success ? '成功' : '失败'}</span></h3>
          <div class="msub">${b.label}</div>
          <div class="kpis" id="kb3"></div>
          <div class="vtext">${b.text}</div>
        </div>
        <div class="mini ${oCls}">
          <h3>论文口径 <span class="badge ${oBadge}">${oTxt}</span></h3>
          <div class="msub">${o.label}</div>
          <div class="kpis" id="ko3"></div>
          <div class="vtext">${o.text}</div>
        </div>
      </div>
      <div class="kpis" id="k3"></div>
      <div class="msgbox" id="m3"></div>
      <table id="t3"></table>
      <div class="hintline">逐位判决（红=错，最多显示前 400 位）</div>
      <div class="bits" id="b3map"></div>
      <div class="dl" id="d3"></div>`;
    kpis(document.getElementById('kb3'), [
      ['一致率', b.agree, '%'],
      ['领先 margin', b.margin, '百分点'],
      ['显著性 z', b.z, 'σ'],
      ['观测位数', b.n_obs, '位'],
      ['锁定阈值', b.threshold, '%'],
      ['相位 d', b.phase_d, ''],
      ['恢复后 BER', b.ber, '%']
    ]);
    kpis(document.getElementById('ko3'), [
      ['锚点位置', (o.anchor === null ? '未命中' : ('第 ' + o.anchor + ' 位')), ''],
      ['恢复后 BER', (o.ber === null ? '—' : o.ber), '%'],
      ['已知原水印', (o.available ? '是' : '否'), '']
    ]);
    kpis(document.getElementById('k3'), [
      ['提取原始 BER', m.raw_ber, '%'],
      ['接收段数', m.n_seg_rx, '段'],
      ['耗时', m.elapsed, 's']
    ]);
    document.getElementById('m3').innerHTML = j.message_html;
    document.getElementById('t3').innerHTML = j.detail_table;
    const bm = document.getElementById('b3map');
    bm.innerHTML = '';
    for (const e of (j.error_map || [])) {
      const s = document.createElement('span');
      s.className = 'bit' + (e ? ' e' : '');
      bm.appendChild(s);
    }
    downloads(document.getElementById('d3'), j.downloads);
  } catch (e) {
    err.textContent = '恢复出错：' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '③ 恢复与判定';
  }
});
</script>
</body></html>"""


@app.route("/")
def index():
    """渲染主页面；攻击元信息用服务端定义注入，前后端只有一份口径。"""
    meta = {k: {"label": v["label"], "hint": v.get("hint", ""), "param": v.get("param")}
            for k, v in ATTACKS.items()}
    return HTML.replace("__ATTACKS__", json.dumps(meta, ensure_ascii=False))


@app.route("/api/sample")
def sample():
    """内置示例宿主，方便不传文件时直接体验。"""
    p = ROOT / "data" / "music_01.wav"
    if not p.exists():
        return jsonify({"error": "缺少内置示例音频 data/music_01.wav"}), 404
    return send_file(str(p), mimetype="audio/wav")


# ---------------------------------------------------------------------------
# 三阶段接口（前端分步调用）
# ---------------------------------------------------------------------------
@app.route("/api/embed", methods=["POST"])
def api_embed():
    """阶段 1：嵌入，返回 token + 嵌入指标 + 下载入口。"""
    try:
        x, msg, repeat = _parse_embed_form()
        _, payload = _do_embed(x, msg, repeat)
        return jsonify(payload)
    except ValueError as e:            # 用户输入问题：400，前端可直接提示
        return jsonify({"error": str(e)}), 400
    except Exception as e:             # 代码问题：500，并把 traceback 打到控制台
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/attack", methods=["POST"])
def api_attack():
    """阶段 2：在已嵌入的音频上施加攻击（需要阶段 1 的 token）。"""
    try:
        st = _get_state(request.form.get("token", ""))
        payload = _do_attack(st, request.form.get("attack", "none"),
                             request.form.get("param"))
        return jsonify(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/recover", methods=["POST"])
def api_recover():
    """阶段 3：双口径恢复与判定（需要阶段 1、2 的 token）。"""
    try:
        st = _get_state(request.form.get("token", ""))
        return jsonify(_do_recover(st))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    """一键跑完三步（演示/测试用）：返回扁平字段 + 三阶段嵌套结果。"""
    t0 = time.time()
    try:
        x, msg, repeat = _parse_embed_form()
        st, p1 = _do_embed(x, msg, repeat)
        p2 = _do_attack(st, request.form.get("attack", "none"),
                        request.form.get("param"))
        p3 = _do_recover(st)
        # 扁平字段：兼容旧前端/冒烟测试的取数方式
        flat = dict(p3["metrics"])
        flat.update({"snr_embed": p1["metrics"]["snr_embed"],
                     "n_bits": p1["metrics"]["n_bits"],
                     "n_seg": p3["metrics"]["n_seg_rx"],
                     "capacity_bits": p1["metrics"]["capacity_bits"],
                     "capacity_chars": p1["metrics"]["capacity_chars"],
                     "snr_att": p2["metrics"]["snr_att"],
                     "len_ratio": p2["metrics"]["len_ratio"],
                     "elapsed": round(time.time() - t0, 2)})
        return jsonify({
            "success": p3["success"], "verdict": p3["verdict"],
            "verdict_text": p3["verdict_text"],
            "blind": p3["blind"], "oracle": p3["oracle"],
            "metrics": flat,
            "message_html": p3["message_html"],
            "error_map": p3["error_map"],
            "detail_table": (p1["detail_table"] + p2["detail_table"] + p3["detail_table"]),
            "downloads": {**p1["downloads"], **p2["downloads"]},
            "token": p1["token"],
            "embed": p1, "attack": p2, "recover": p3,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/download/<token>/<which>")
def download(token: str, which: str):
    """下载三阶段产物：wm（含水印）/ att（攻击后）/ key（密钥与真值）。"""
    item = STASH.get(token)
    key = {"wm": "wm_bytes", "att": "att_bytes", "key": "key_bytes"}.get(which)
    if not item or key is None or item.get(key) is None:
        return jsonify({"error": "该产物尚未生成或结果已过期，请按顺序执行第 1/2 步"}), 404
    name = {"wm": "watermarked.wav", "att": "attacked.wav", "key": "key.npz"}[which]
    return send_file(io.BytesIO(item[key]), as_attachment=True, download_name=name,
                     mimetype="application/zip" if which == "key" else "audio/wav")


@app.route("/api/embed_only", methods=["POST"])
def api_embed_only():
    """只做嵌入并返回 zip（含水印音频 + key.npz），供“先分发、后检测”场景。"""
    try:
        x, msg, repeat = _parse_embed_form()
        st, _ = _do_embed(x, msg, repeat)
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("watermarked.wav", st["wm_bytes"])
            zf.writestr("key.npz", st["key_bytes"])
        z.seek(0)
        return send_file(z, mimetype="application/zip", download_name="lraw_embed.zip")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    print(f"LRAW 演示前端：http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
