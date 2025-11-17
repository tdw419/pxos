#!/usr/bin/env python3
"""
pxvm/examples/run_dot_test.py

Run the dot_test.pxi program through the pxVM interpreter and
print the resulting dot product stored at (0,3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program
from .make_dot_test import make_dot_test_image, save_pxi


def load_pxi(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def main() -> None:
    base = Path(__file__).parent
    pxi_path = base / "dot_test.pxi"

    if not pxi_path.exists():
        # Generate if missing
        img = make_dot_test_image()
        save_pxi(pxi_path, img)
        print(f"Generated {pxi_path}")

    img = load_pxi(pxi_path)

    # Run the program
    img = run_program(img)

    # Read result at (0,3)
    low = int(img[3, 0, 0])
    high = int(img[3, 0, 1])
    dot_val = low + (high << 8)

    print(f"Dot product read from pixel (0,3): {dot_val}")
    print(f"Encoded as bytes: R={low}, G={high}")


if __name__ == "__main__":
    main()
