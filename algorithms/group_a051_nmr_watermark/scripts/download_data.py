"""Resume downloads, verify checksums, and safely extract only needed subsets."""
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
import socket
import http.client
import urllib.error
from pathlib import Path

SOURCES = {
    "fma": ("https://os.unil.cloud.switch.ch/fma/fma_medium.zip", "sha1", "c67b69ea232021025fca9231fc1c7c1a063ab50b", "fma_medium/"),
    "metadata": ("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip", "sha1", "f0df49ffe5f2a6008d7dc83c6915b31835dfe733", "fma_metadata/"),
    "rwc": ("https://zenodo.org/records/17177919/files/RWC-CGJPR.zip?download=1", "md5", "272309ed06121e6ed2ac5c8f55664baa", "RWC-G/"),
}

def download(key, root):
    url, algorithm, expected, prefix = SOURCES[key]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / (key + ".zip")
    partial = archive.with_suffix(".zip.part")
    # A server can end a stream early without raising an exception. Check the
    # authoritative range length before promoting a partial to an archive.
    probe = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(probe, timeout=60) as response:
        content_range = response.headers.get('Content-Range', '')
        if response.status != 206 or '/' not in content_range:
            raise RuntimeError('Dataset server must support byte ranges for safe resume')
        total = int(content_range.split('/')[-1])
    if archive.exists() and archive.stat().st_size < total:
        if partial.exists():
            raise RuntimeError('Both incomplete ZIP and partial exist; inspect before resuming')
        archive.rename(partial)
    if not archive.exists():
        failures = 0
        while (partial.stat().st_size if partial.exists() else 0) < total:
            offset = partial.stat().st_size if partial.exists() else 0
            before = offset
            req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "User-Agent": "NMR-course-reproduction/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Server returned a mismatched range; partial preserved')
                    last = time.monotonic()
                    with partial.open('ab') as dst:
                        while chunk := response.read(1024 * 1024):
                            dst.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last > 20:
                                print(f'{key}: {offset / 2**30:.2f}/{total / 2**30:.2f} GiB', flush=True)
                                last = time.monotonic()
            except (OSError, urllib.error.URLError, socket.timeout, http.client.IncompleteRead) as exc:
                print(f'{key}: connection interrupted: {exc}; resuming', flush=True)
            if offset < total:
                failures = failures + 1 if offset == before else 0
                if failures >= 20:
                    raise RuntimeError('20 download attempts without progress; partial preserved')
                time.sleep(min(30, 2**min(failures, 5)))
        if partial.stat().st_size != total:
            raise RuntimeError('Unexpected downloaded size; partial preserved')
        partial.rename(archive)
    digest = hashlib.new(algorithm)
    with archive.open("rb") as src:
        while chunk := src.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"{key}: checksum mismatch; archive retained for investigation")
    count = 0
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            # RWC archive may contain a version/root prefix.
            normalized = member.filename.replace("\\", "/")
            parts = normalized.split("/")
            wanted = prefix.rstrip("/")
            if wanted not in parts:
                continue
            relative = Path(*parts[parts.index(wanted):])
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive path")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, target.open("wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                count += 1
    if count == 0:
        raise RuntimeError(f"No {prefix} files found; inspect archive layout")
    (root / f"{key}.verified.json").write_text(json.dumps({"url": url, algorithm: expected, "files": count}, indent=2))
    print(f"{key}: verified and extracted {count} files", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    download(args.dataset, args.root)
