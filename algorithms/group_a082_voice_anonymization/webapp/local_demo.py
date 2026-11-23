"""Run the demo locally with the offline adapters from local_shim.

    python webapp/local_demo.py --config F:\\...\\config.local.yaml --port 8013

Same pages and API as webapp/server.py; the only difference is that Whisper runs
through transformers and ECAPA is built by hand (see local_shim.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from local_shim import LocalService, install_whisper_stub  # noqa: E402
from server import create_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8013)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("whisper_model_dir"):
        install_whisper_stub(cfg["whisper_model_dir"])
    app = create_app(cfg, service=LocalService(cfg))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
