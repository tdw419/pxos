#!/usr/bin/env python3
"""
boot_project.py

Hyper-bootstrap: start entire pxOS from a single Project Boot Pixel.
"""

import sys
import zlib
import json
import tarfile
import io
from pathlib import Path
from PIL import Image
import tempfile
import importlib
import os

REGISTRY = Path("project_boot_registry.json")
BLOBS_DIR = Path("project_blobs")

def load_registry():
    if not REGISTRY.exists():
        raise RuntimeError("project_boot_registry.json not found")
    return json.loads(REGISTRY.read_text())

def boot_from_pixel(pixel_path: str):
    img = Image.open(pixel_path).convert("RGBA")
    if img.size != (1,1):
        raise RuntimeError("Project Boot Pixel must be 1x1")

    R,G,B,A = img.getpixel((0,0))
    pid = (R << 24) | (G << 16) | (B << 8) | A

    reg = load_registry()
    entry = reg.get(str(pid))
    if not entry:
        raise RuntimeError(f"No project registered for ID=0x{pid:08X}")

    blob_path = Path(entry["blob"])
    compressed = blob_path.read_bytes()
    raw_tar = zlib.decompress(compressed)

    tmpdir = Path(tempfile.mkdtemp(prefix="pxos_boot_"))
    print(f"[BOOT] Decompressing project into {tmpdir}")

    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:gz") as tf:
        tf.extractall(tmpdir)

    # Add to sys.path and import pxos_boot.py from inside
    sys.path.insert(0, str(tmpdir))
    os.chdir(tmpdir) # Change directory to the temp dir
    print("[BOOT] Importing pxos_boot from extracted project...")
    pxos_boot = importlib.import_module("pxos_boot")

    # Hand control to your universal bootloader
    # Here you can define a convention, e.g. pxos_boot.main()
    if hasattr(pxos_boot, "main"):
        # We will now create a god pixel and boot it.
        print("[BOOT] Handing control to the resurrected pxOS...")
        # 1. Create the God Pixel for LifeSim
        god_pixel_cli = importlib.import_module("god_pixel_cli")
        lifesim_god_pixel_path = "lifesim_god_pixel.png"
        god_pixel_cli.create_pixel("LifeSim v0.1", lifesim_god_pixel_path)

        # 2. Boot the God Pixel
        pxos_boot.boot(lifesim_god_pixel_path)
    else:
        print("[WARN] pxos_boot.main() not found; nothing to run")

def main(argv):
    if len(argv) != 2:
        print("Usage: boot_project.py project_pxos.boot.png")
        sys.exit(1)
    boot_from_pixel(argv[1])

if __name__ == "__main__":
    main(sys.argv)
