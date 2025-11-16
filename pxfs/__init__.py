"""
pxfs - Pixel Filesystem Reference Implementation
Compresses files into single-pixel representations with full reconstruction capability.
"""

from .pixel import Pixel, PixelType, PixelVisual, Position, Metadata, DataReference
from .storage import ContentStore, MetadataStore
from .compressor import PixelCompressor
from .decompressor import PixelDecompressor
from .visualizer import PixelMapVisualizer

__version__ = "0.1.0"
__spec__ = "pxos-pixel-v0.1"

__all__ = [
    "Pixel",
    "PixelType",
    "PixelVisual",
    "Position",
    "Metadata",
    "DataReference",
    "ContentStore",
    "MetadataStore",
    "PixelCompressor",
    "PixelDecompressor",
    "PixelMapVisualizer",
]
