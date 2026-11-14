"""Extract a few CRC-checked tracks from the completed prefix of FMA download.
Pilot use only: full-dataset experiments require the published archive checksum.
"""
import bz2
import struct
import zlib
from pathlib import Path

root = Path("data/pilot")
root.mkdir(parents=True, exist_ok=True)
count = 0
with Path("data/fma.zip.part").open("rb") as f:
    while count < 32:
        header = f.read(30)
        if len(header) < 30 or header[:4] != b"PK\x03\x04":
            break
        _, _, flag, method, _, _, crc, compressed, size, nl, el = struct.unpack("<IHHHHHIIIHH", header)
        name = f.read(nl).decode("utf-8")
        f.read(el)
        if flag & 8:
            raise RuntimeError("Streaming descriptors unsupported")
        raw = f.read(compressed)
        if len(raw) != compressed:
            break
        if not name.endswith(".mp3"):
            continue
        content = bz2.decompress(raw) if method == 12 else zlib.decompress(raw, -15) if method == 8 else raw
        if len(content) != size or zlib.crc32(content) != crc:
            raise RuntimeError("Corrupt pilot member")
        (root / Path(name).name).write_bytes(content)
        count += 1
print(f"Extracted {count} CRC-verified pilot MP3s; not the full paper dataset.")
