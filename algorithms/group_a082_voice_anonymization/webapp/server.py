"""Web front end: static pages (webapp/static) plus a small JSON API over a082.service.

    python webapp/server.py --config webapp/config.yaml [--host 0.0.0.0] [--port 8000]

API
    GET  /api/systems                       systems available with the configured checkpoints
    POST /api/anonymize                     form: audio (16 kHz mono WAV, the page converts), system, seed, p, similarity
    GET  /api/result/{id}/{orig|anon}.wav   audio of a recent result (kept in memory only, last 32)
    GET  /api/result/{id}/{orig|anon}.png   log-mel spectrogram, both on the same colour scale
    GET  /api/results                       Table 1 rows (paper, ours where measured) and the admixture comparison
    GET  /api/samples, /samples/<file>      optional example clips (config `samples_dir`)
"""
from __future__ import annotations

import argparse
import io
import json
import secrets
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import soundfile as sf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from a082 import audio, viz  # noqa: E402
from a082.eval import targets  # noqa: E402
from a082.service import AnonService  # noqa: E402

GROUPS = [("kNN-VC 系", ("knn1", "knn1m", "knn2", "knn3")), ("级联", ("tts",)), ("混合", ("mix",)),
          ("WavLM 转换模型", ("wc5", "wc6", "wc7", "wc8"))]
UI = {  # number, name, one-line description, as in the design
    "knn1": ("1", "kNN-VC", "每句随机取一个目标说话人，k = 4 近邻替换"),
    "knn1m": ("1M", "多目标池化", "两个目标的近邻结果逐帧加权平均"),
    "knn2": ("2", "+ 长度扰动", "语速随机拉伸 0.8–1.2 倍，音高不变"),
    "knn3": ("3", "+ 长度扰动 + 特征加噪", "在特征上加均匀噪声，隐私更强、可懂度下降"),
    "tts": ("4", "Whisper → Piper", "先转写成文字，再用 904 个合成音色之一读出来"),
    "mix": ("1+4", "随机混合", "每句以概率 p 用级联，否则用 kNN-VC"),
    "wc5": ("5", "WavLM 转换（基础）", "学习 kNN-VC 的输出，所有句子转到同一个目标"),
    "wc6": ("6", "+ 说话人对抗", "与说话人分类器对抗训练"),
    "wc7": ("7", "+ 离散目标", "k-means 单元上的 CTC 损失"),
    "wc8": ("8", "+ 两者", "对抗与离散同时使用"),
}
MAX_SECONDS = 30
KEEP = 32


def create_app(cfg: dict, service: AnonService | None = None) -> FastAPI:
    service = service or AnonService(cfg)
    lock = threading.Lock()  # one model instance: serialise inference
    results: OrderedDict[str, dict] = OrderedDict()
    app = FastAPI(title="A082 speaker anonymization")

    @app.get("/api/systems")
    def systems():
        avail = service.systems()
        groups = []
        for name, keys in GROUPS:
            items = [{"key": k, "num": UI[k][0], "name": UI[k][1], "desc": UI[k][2]} for k in keys if k in avail]
            if items:
                groups.append({"name": name, "items": items})
        return {"groups": groups, "similarity": bool(cfg.get("speaker_encoder", True)),
                "threshold": cfg.get("similarity_threshold", 0.25), "max_seconds": MAX_SECONDS}

    @app.post("/api/anonymize")
    def anonymize(audio_file: UploadFile = File(..., alias="audio"), system: str = Form(...),
                  seed: str = Form(""), p: float = Form(0.4), similarity: bool = Form(True)):
        if system not in service.systems():
            raise HTTPException(400, f"unknown or unavailable system {system!r}")
        try:
            data, sr = sf.read(io.BytesIO(audio_file.file.read()), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"cannot read audio: {e}")
        wav = audio.resample(torch.from_numpy(data.mean(axis=1))[None], sr)
        if wav.shape[-1] > MAX_SECONDS * audio.SR:
            raise HTTPException(400, f"audio longer than {MAX_SECONDS} s")
        if wav.shape[-1] < int(0.5 * audio.SR):
            raise HTTPException(400, "audio shorter than 0.5 s")
        seed = seed.strip() or secrets.token_hex(4)
        with lock:
            t0 = time.time()
            res = service.anonymize(wav, audio.SR, system, seed=seed, p=p)
            elapsed = time.time() - t0
            sim = service.speaker_similarity(wav, res.wav) if similarity else None
        rid = secrets.token_hex(8)
        results[rid] = {"orig": wav, "anon": res.wav, "scale": viz.shared_scale(wav, res.wav)}
        while len(results) > KEEP:
            results.popitem(last=False)
        return {"id": rid, "seed": seed, "system": res.system, "system_name": UI[res.system][1],
                "requested": system, "details": res.details, "similarity": sim,
                "seconds_in": round(wav.shape[-1] / audio.SR, 2), "seconds_out": round(res.wav.shape[-1] / audio.SR, 2),
                "elapsed": round(elapsed, 2)}

    def _get(rid: str, which: str):
        if rid not in results or which not in ("orig", "anon"):
            raise HTTPException(404, "result expired or unknown")
        return results[rid]

    @app.get("/api/result/{rid}/{which}.wav")
    def result_wav(rid: str, which: str):
        r = _get(rid, which)
        buf = io.BytesIO()
        sf.write(buf, r[which].reshape(-1).clamp(-1, 1).numpy(), audio.SR, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")

    @app.get("/api/result/{rid}/{which}.png")
    def result_png(rid: str, which: str):
        r = _get(rid, which)
        return Response(viz.spectrogram_png(r[which], *r["scale"]), media_type="image/png")

    @app.get("/api/results")
    def table1():
        path = cfg.get("results_json")
        rows = json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else None
        if rows is None:  # paper only
            rows = [{"row": k, "suffix": v[0], "name": targets.NAMES_ZH[k],
                     "paper": {"EER": v[1], "UAR": v[2], "WER": v[3]}, "ours": None, "verdict": None}
                    for k, v in targets.TABLE1.items()]
        bonus_path = cfg.get("bonus_json")
        bonus = json.loads(Path(bonus_path).read_text(encoding="utf-8")) if bonus_path and Path(bonus_path).exists() else {}
        return {"rows": rows, "bonus": bonus}

    samples_dir = Path(cfg["samples_dir"]) if cfg.get("samples_dir") else None

    @app.get("/api/samples")
    def samples():
        if samples_dir is None or not samples_dir.exists():
            return {"samples": []}
        return {"samples": [{"name": f.stem, "url": f"/samples/{f.name}"} for f in sorted(samples_dir.glob("*.wav"))]}

    if samples_dir is not None and samples_dir.exists():
        app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")
    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

    @app.exception_handler(Exception)
    def on_error(_, exc):  # keep the page informative instead of a bare 500
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    uvicorn.run(create_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
