#!/usr/bin/env python3
"""
pack_project.py

Pack the entire pxOS project directory into a single Project Boot Pixel.
"""

import os
import io
import tarfile
import zlib
import json
import random
from pathlib import Path
from PIL import Image

REGISTRY = Path("project_boot_registry.json")
BLOBS_DIR = Path("project_blobs")
BLOBS_DIR.mkdir(exist_ok=True)

def load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}

def save_registry(reg):
    REGISTRY.write_text(json.dumps(reg, indent=2))

def tar_directory(root: Path, exclude=None) -> bytes:
    exclude = exclude or []
    buf = io.BytesIO()
    # Use gzip compression for the tarball
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for item in root.iterdir():
            if item.name in exclude:
                continue
            tf.add(item, arcname=item.name)
    return buf.getvalue()

def pack_project(project_root: str, name: str, description: str):
    root = Path(project_root).resolve()
    reg = load_registry()

    print(f"[PACK] Project root: {root}")
    # Exclude git history, project blobs, and caches
    raw_tar = tar_directory(root, exclude=[".git", "project_blobs", "__pycache__", ".DS_Store"])
    compressed = zlib.compress(raw_tar, level=9)
    print(f"[PACK] Raw tar: {len(raw_tar)} bytes, compressed: {len(compressed)} bytes")

    # Pick random unused world_id
    while True:
        pid = random.getrandbits(32)
        if str(pid) not in reg:
            break

    blob_path = BLOBS_DIR / f"{pid:08x}.bin"
    blob_path.write_bytes(compressed)

    R = (pid >> 24) & 0xFF
    G = (pid >> 16) & 0xFF
    B = (pid >> 8) & 0xFF
    A = pid & 0xFF

    boot_px = Image.new("RGBA", (1,1), (R,G,B,A))
    out_name = f"project_{name.replace(' ', '_')}.boot.png"
    boot_px.save(out_name)

    reg[str(pid)] = {
        "name": name,
        "description": description,
        "root": str(root),
        "blob": str(blob_path),
        "pixel": [R,G,B,A]
    }
    save_registry(reg)

    print(f"[OK] Project Boot Pixel written to {out_name}")
    print(f"     Color RGBA = ({R},{G},{B},{A})   ID = 0x{pid:08X}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: pack_project.py <project_root> [name] [desc]")
        sys.exit(1)
    project_root = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "pxOS"
    desc = sys.argv[3] if len(sys.argv) > 3 else "Complete pxOS God Pixel project"
    pack_project(project_root, name, desc)
