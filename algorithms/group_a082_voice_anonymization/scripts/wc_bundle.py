"""Inputs of rows 5-8 as one archive, so that the training can move to another machine (e.g. Huawei Cloud) and
survives the release of the rented one.

    python scripts/wc_bundle.py pack <work dir> <archive.tar>     # on the machine that has them
    python scripts/wc_bundle.py unpack <archive.tar> <work dir>   # on the new machine (any work dir)

Contents (paths relative to the work dir): wc/target.pt (target speaker a and its k-means units), wc/spk2.pt
(speaker classifier, validation accuracy 0.990), the latest wc/wc*/last.pt checkpoints, the target speaker's
audio files, and wc/target_manifest.json listing those files. `pack` also writes the manifest with absolute
paths into the work dir; `unpack` rewrites it for the new work dir. The training reads the target speaker's
audio only from that manifest (`--manifest`), so the 1251-speaker pool is not needed.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

MANIFEST = "wc/target_manifest.json"


def pack(work: Path, archive: Path) -> None:
    import torch
    speaker = torch.load(work / "wc/target.pt", map_location="cpu")["speaker"]
    files = json.loads((work / "pool_vox1.json").read_text())[speaker]
    rel = [str(Path(f).relative_to(work)) for f in files]
    (work / MANIFEST).write_text(json.dumps({speaker: [str(work / r) for r in rel]}, indent=1))
    members = ["wc/target.pt", "wc/spk2.pt"] + sorted(str(p.relative_to(work)) for p in work.glob("wc/wc*/last.pt"))
    with tarfile.open(archive, "w") as tar:
        for m in members + rel:
            tar.add(work / m, arcname=m)
        info = tarfile.TarInfo(MANIFEST)
        data = json.dumps({speaker: rel}, indent=1).encode()
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(f"{archive}: target {speaker}, {len(rel)} audio files, {', '.join(members)}")


def unpack(archive: Path, work: Path) -> None:
    with tarfile.open(archive) as tar:
        tar.extractall(work)
    m = json.loads((work / MANIFEST).read_text())
    (work / MANIFEST).write_text(json.dumps({s: [str(work / r) for r in rel] for s, rel in m.items()}, indent=1))
    missing = [f for fs in json.loads((work / MANIFEST).read_text()).values() for f in fs if not Path(f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} target files missing, e.g. {missing[0]}")
    print(f"unpacked into {work}: {', '.join(m)} with {sum(len(v) for v in m.values())} files")


if __name__ == "__main__":
    cmd, a, b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {"pack": pack, "unpack": unpack}[cmd](a, b)
