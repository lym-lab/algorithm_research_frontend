import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="NMR watermark paper reproduction")
    sub = p.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("manifest")
    cache.add_argument("output")
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--raw-stft", action="store_true", help="Legacy diagnostic convention only")
    train = sub.add_parser("search")
    train.add_argument("manifests")
    train.add_argument("cache")
    train.add_argument("output")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--device", default="cuda")
    select = sub.add_parser("select")
    select.add_argument("output")
    embed = sub.add_parser("embed")
    embed.add_argument("checkpoint")
    embed.add_argument("input")
    embed.add_argument("output")
    embed.add_argument("--message", default="")
    extract = sub.add_parser("extract")
    extract.add_argument("checkpoint")
    extract.add_argument("input")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("checkpoint")
    evaluation.add_argument("manifest")
    evaluation.add_argument("output")
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--peaq-audio", action="store_true")
    ui = sub.add_parser("ui")
    ui.add_argument("--registry", default="runs/paper/selected.json")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    if args.command == "cache":
        from .training import cache_masks
        cache_masks(args.manifest, args.output, args.device, normalized=not args.raw_stft)
    elif args.command == "search":
        from .training import search
        search(args.manifests, args.cache, args.output, args.device, args.batch_size)
    elif args.command == "select":
        from .training import select
        print(json.dumps(select(args.output), indent=2))
    elif args.command == "embed":
        from .inference import embed_file
        print(json.dumps(embed_file(args.checkpoint, args.input, args.output, args.message), indent=2))
    elif args.command == "extract":
        from .inference import load_model, extract_array
        from .audio import read_audio
        model, _ = load_model(args.checkpoint)
        for block in extract_array(model, read_audio(args.input)):
            print("".join(map(str, (block >= .5).astype(int))))
    elif args.command == "evaluate":
        from .evaluation import evaluate
        print(json.dumps(evaluate(args.checkpoint, args.manifest, args.output, args.device, args.peaq_audio), indent=2))
    elif args.command == "ui":
        import os
        import torch
        torch.set_num_threads(4)
        os.environ['NO_PROXY'] = os.environ.get('NO_PROXY', '') + ',127.0.0.1,localhost'
        os.environ['MPLCONFIGDIR'] = str(Path('.cache/matplotlib').resolve())
        from .app import build_app
        build_app(args.registry).queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port)

if __name__ == "__main__":
    main()
