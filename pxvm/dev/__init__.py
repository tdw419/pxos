"""
pxvm.dev - Development Tools for Pixel Programs

Philosophy:
- Pixels are the PRIMARY format (source of truth)
- Dev tools are LENSES to view/understand pixels
- Tools HELP humans/LLMs, don't REPLACE pixels

Tools:
- inspector: View program structure and data
- assembler: Write programs in DSL, compile to pixels
- debugger: Step through execution with symbolic names (future)
- diff: Compare programs semantically (future)
- canvas: Render programs visually (future)

All tools work WITH pixels, never INSTEAD of pixels.
"""

from .inspector import PixelInspector
from .assembler import PixelAssembler

__all__ = ["PixelInspector", "PixelAssembler"]
