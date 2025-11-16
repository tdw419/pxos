#!/usr/bin/env python3
"""
InfiniteMap - 2D Spatial Memory System

Manages an infinite 2D grid of pixels backed by PixelFS storage.
Supports sparse storage, spatial queries, and neighbor operations.

Features:
- Infinite 2D coordinate space (negative coordinates supported)
- Region-based storage (tiles of configurable size)
- Spatial neighbor queries
- Efficient sparse storage (only used regions consume disk)
- Transparent compression and caching
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import json

# Use absolute import for archive compatibility
try:
    from pixel_llm.core.pixelfs import PixelFS
except ImportError:
    # Fallback for legacy relative imports
    from pixelfs import PixelFS


# Default region size (64x64 pixels)
DEFAULT_REGION_SIZE = 64


@dataclass
class MapRegion:
    """A rectangular region in the infinite map"""
    x: int  # Top-left X coordinate
    y: int  # Top-left Y coordinate
    width: int  # Region width
    height: int  # Region height
    data: np.ndarray  # Pixel data (height, width, 3) uint8

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside this region"""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)

    def overlaps(self, other: 'MapRegion') -> bool:
        """Check if this region overlaps another"""
        return not (self.x + self.width <= other.x or
                    other.x + other.width <= self.x or
                    self.y + self.height <= other.y or
                    other.y + other.height <= self.y)

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2)"""
        return (self.x, self.y,
                self.x + self.width - 1,
                self.y + self.height - 1)


class InfiniteMap:
    """
    Infinite 2D grid of RGB pixels

    Usage:
        # Create map
        imap = InfiniteMap("my_map")

        # Write a region
        data = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        imap.write_region(x=100, y=200, data=data)

        # Read a region
        read_data = imap.read_region(x=100, y=200, width=64, height=64)

        # Get neighbors
        neighbors = imap.get_neighbors(x=100, y=200, radius=1)
    """

    def __init__(self, storage_dir: str, region_size: int = DEFAULT_REGION_SIZE):
        """
        Initialize InfiniteMap

        Args:
            storage_dir: Directory to store map data
            region_size: Size of each region tile (default 64x64)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.region_size = region_size
        self.pfs = PixelFS()

        # Region index: maps region coordinates to filenames
        self.index_file = self.storage_dir / "map_index.json"
        self.index = self._load_index()

        # Cache for frequently accessed regions
        self.cache = {}
        self.cache_size = 16  # Keep 16 regions in memory

    def _load_index(self) -> dict:
        """Load or create the region index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save the region index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _region_key(self, rx: int, ry: int) -> str:
        """Get region coordinate key"""
        return f"{rx},{ry}"

    def _world_to_region(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        Convert world coordinates to region coordinates + local offset.

        Args:
            x, y: World coordinates

        Returns:
            (region_x, region_y, local_x, local_y)
        """
        # Python's // and % handle negatives correctly for us!
        # For x=-100, region_size=64:
        #   -100 // 64 = -2  (floor division)
        #   -100 % 64 = 28   (positive remainder)
        # This means pixel -100 is in region -2 at local position 28 ✓
        rx = x // self.region_size
        ry = y // self.region_size
        local_x = x % self.region_size
        local_y = y % self.region_size

        return rx, ry, local_x, local_y

    def _region_to_world(self, rx: int, ry: int) -> Tuple[int, int]:
        """Convert region coordinates to world coordinates (top-left)"""
        return rx * self.region_size, ry * self.region_size

    def _get_region_filename(self, rx: int, ry: int) -> str:
        """Get filename for a region"""
        key = self._region_key(rx, ry)
        if key not in self.index:
            # Create new filename
            filename = f"region_{rx}_{ry}.pxi"
            self.index[key] = filename
            self._save_index()
        return self.index[key]

    def _load_region(self, rx: int, ry: int) -> Optional[np.ndarray]:
        """Load a region from disk"""
        # Check cache first
        cache_key = (rx, ry)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Check if region exists
        key = self._region_key(rx, ry)
        if key not in self.index:
            return None

        # Load from disk
        filename = self.index[key]
        filepath = self.storage_dir / filename

        if not filepath.exists():
            return None

        # Read pixel data
        data = self.pfs.read(str(filepath))

        # Convert to numpy array
        pixels = np.frombuffer(data, dtype=np.uint8)
        expected_size = self.region_size * self.region_size * 3

        if len(pixels) < expected_size:
            # Pad if needed
            pixels = np.pad(pixels, (0, expected_size - len(pixels)), mode='constant')

        pixels = pixels[:expected_size].reshape((self.region_size, self.region_size, 3))

        # Cache it
        self._cache_region(rx, ry, pixels)

        return pixels

    def _save_region(self, rx: int, ry: int, data: np.ndarray):
        """Save a region to disk"""
        filename = self._get_region_filename(rx, ry)
        filepath = self.storage_dir / filename

        # Convert to bytes
        data_bytes = data.tobytes()

        # Write to PixelFS
        self.pfs.write(str(filepath), data_bytes, width=self.region_size)

        # Cache it
        self._cache_region(rx, ry, data)

    def _cache_region(self, rx: int, ry: int, data: np.ndarray):
        """Add region to cache (LRU-like)"""
        cache_key = (rx, ry)
        self.cache[cache_key] = data.copy()

        # Simple cache eviction (keep most recent regions)
        if len(self.cache) > self.cache_size:
            # Remove oldest (first in dict)
            oldest = next(iter(self.cache))
            del self.cache[oldest]

    def write_region(self, x: int, y: int, data: np.ndarray):
        """
        Write a rectangular region of pixels

        Args:
            x, y: Top-left world coordinates
            data: Pixel data (height, width, 3) uint8 array
        """
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError(f"Data must be (height, width, 3) shape, got {data.shape}")

        if data.dtype != np.uint8:
            # Try to convert
            data = data.astype(np.uint8)

        height, width = data.shape[:2]

        # Determine which regions this write spans
        x1, y1 = x, y
        x2, y2 = x + width - 1, y + height - 1

        rx1, ry1, _, _ = self._world_to_region(x1, y1)
        rx2, ry2, _, _ = self._world_to_region(x2, y2)

        # Write to all affected regions
        for ry in range(ry1, ry2 + 1):
            for rx in range(rx1, rx2 + 1):
                # Load or create region
                region_data = self._load_region(rx, ry)
                if region_data is None:
                    region_data = np.zeros((self.region_size, self.region_size, 3), dtype=np.uint8)

                # Calculate write bounds in region space
                region_wx, region_wy = self._region_to_world(rx, ry)

                # Region bounds
                reg_x1 = max(0, x - region_wx)
                reg_y1 = max(0, y - region_wy)
                reg_x2 = min(self.region_size, x + width - region_wx)
                reg_y2 = min(self.region_size, y + height - region_wy)

                # Source data bounds
                src_x1 = max(0, region_wx - x)
                src_y1 = max(0, region_wy - y)
                src_x2 = src_x1 + (reg_x2 - reg_x1)
                src_y2 = src_y1 + (reg_y2 - reg_y1)

                # Copy data
                region_data[reg_y1:reg_y2, reg_x1:reg_x2] = data[src_y1:src_y2, src_x1:src_x2]

                # Save region
                self._save_region(rx, ry, region_data)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Read a rectangular region of pixels

        Args:
            x, y: Top-left world coordinates
            width, height: Region dimensions

        Returns:
            Pixel data (height, width, 3) uint8 array (zeros if unwritten)
        """
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        # Result array (initially zeros)
        result = np.zeros((height, width, 3), dtype=np.uint8)

        # Determine which regions to read
        x1, y1 = x, y
        x2, y2 = x + width - 1, y + height - 1

        rx1, ry1, _, _ = self._world_to_region(x1, y1)
        rx2, ry2, _, _ = self._world_to_region(x2, y2)

        # Read from all affected regions
        for ry in range(ry1, ry2 + 1):
            for rx in range(rx1, rx2 + 1):
                # Load region (may be None if never written)
                region_data = self._load_region(rx, ry)
                if region_data is None:
                    continue  # Leave as zeros

                # Calculate read bounds
                region_wx, region_wy = self._region_to_world(rx, ry)

                # Region bounds
                reg_x1 = max(0, x - region_wx)
                reg_y1 = max(0, y - region_wy)
                reg_x2 = min(self.region_size, x + width - region_wx)
                reg_y2 = min(self.region_size, y + height - region_wy)

                # Destination bounds
                dst_x1 = max(0, region_wx - x)
                dst_y1 = max(0, region_wy - y)
                dst_x2 = dst_x1 + (reg_x2 - reg_x1)
                dst_y2 = dst_y1 + (reg_y2 - reg_y1)

                # Copy data
                result[dst_y1:dst_y2, dst_x1:dst_x2] = region_data[reg_y1:reg_y2, reg_x1:reg_x2]

        return result

    def get_neighbors(self, x: int, y: int, radius: int = 1) -> List[Tuple[int, int]]:
        """
        Get coordinates of neighboring regions

        Args:
            x, y: World coordinates
            radius: Radius in regions (1 = immediate neighbors)

        Returns:
            List of (rx, ry) region coordinates
        """
        rx, ry, _, _ = self._world_to_region(x, y)

        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip center
                neighbor_rx = rx + dx
                neighbor_ry = ry + dy

                # Check if region exists
                key = self._region_key(neighbor_rx, neighbor_ry)
                if key in self.index:
                    neighbors.append((neighbor_rx, neighbor_ry))

        return neighbors

    def get_all_regions(self) -> List[Tuple[int, int]]:
        """Get coordinates of all stored regions"""
        regions = []
        for key in self.index.keys():
            rx, ry = map(int, key.split(','))
            regions.append((rx, ry))
        return regions

    def get_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of all data (world coordinates)

        Returns:
            (x1, y1, x2, y2) or None if empty
        """
        regions = self.get_all_regions()
        if not regions:
            return None

        # Find min/max region coordinates
        rx_min = min(rx for rx, _ in regions)
        ry_min = min(ry for _, ry in regions)
        rx_max = max(rx for rx, _ in regions)
        ry_max = max(ry for _, ry in regions)

        # Convert to world coordinates
        x1, y1 = self._region_to_world(rx_min, ry_min)
        x2, y2 = self._region_to_world(rx_max + 1, ry_max + 1)
        x2 -= 1
        y2 -= 1

        return (x1, y1, x2, y2)

    def clear(self):
        """Clear all regions (delete all data)"""
        import shutil
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.cache = {}
        self._save_index()


def demo():
    """Demonstration of InfiniteMap capabilities"""
    import tempfile
    import shutil

    print("=" * 70)
    print("InfiniteMap Demonstration")
    print("=" * 70)
    print()

    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    print(f"Working directory: {tmpdir}\n")

    try:
        # Create map
        imap = InfiniteMap(tmpdir, region_size=64)

        # Demo 1: Write and read single region
        print("Demo 1: Write and read a single region")
        print("-" * 70)
        red_data = np.zeros((64, 64, 3), dtype=np.uint8)
        red_data[:, :, 0] = 255  # Red
        imap.write_region(x=0, y=0, data=red_data)

        read_data = imap.read_region(x=0, y=0, width=64, height=64)
        np.testing.assert_array_equal(read_data, red_data)
        print(f"✅ Single region verified (64x64)\n")

        # Demo 2: Multiple regions
        print("Demo 2: Write multiple non-overlapping regions")
        print("-" * 70)
        green_data = np.zeros((64, 64, 3), dtype=np.uint8)
        green_data[:, :, 1] = 255  # Green
        imap.write_region(x=100, y=0, data=green_data)

        blue_data = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_data[:, :, 2] = 255  # Blue
        imap.write_region(x=0, y=100, data=blue_data)

        print(f"  Total regions: {len(imap.get_all_regions())}")
        print(f"✅ Multiple regions written\n")

        # Demo 3: Negative coordinates
        print("Demo 3: Negative coordinates")
        print("-" * 70)
        yellow_data = np.zeros((32, 32, 3), dtype=np.uint8)
        yellow_data[:, :, 0] = 255  # Red
        yellow_data[:, :, 1] = 255  # Green = Yellow
        imap.write_region(x=-100, y=-100, data=yellow_data)

        read_yellow = imap.read_region(x=-100, y=-100, width=32, height=32)
        np.testing.assert_array_equal(read_yellow, yellow_data)
        print(f"✅ Negative coordinates verified\n")

        # Demo 4: Large cross-region write
        print("Demo 4: Write spanning multiple regions")
        print("-" * 70)
        large_data = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        imap.write_region(x=1000, y=1000, data=large_data)

        read_large = imap.read_region(x=1000, y=1000, width=128, height=128)
        np.testing.assert_array_equal(read_large, large_data)
        print(f"  Spans {len(imap.get_neighbors(1000, 1000, radius=2))} neighbor regions")
        print(f"✅ Cross-region write verified\n")

        # Demo 5: Sparse map
        print("Demo 5: Sparse distant regions")
        print("-" * 70)
        positions = [(0, 0), (10000, 0), (0, 10000), (-5000, -5000)]
        for i, (x, y) in enumerate(positions):
            data = np.zeros((64, 64, 3), dtype=np.uint8)
            data[:, :, i % 3] = (i + 1) * 50
            imap.write_region(x=x, y=y, data=data)

        bounds = imap.get_bounds()
        print(f"  Map bounds: ({bounds[0]}, {bounds[1]}) to ({bounds[2]}, {bounds[3]})")
        print(f"  Total regions: {len(imap.get_all_regions())}")
        print(f"✅ Sparse map verified\n")

        # Demo 6: Read unwritten region
        print("Demo 6: Read unwritten region (returns zeros)")
        print("-" * 70)
        empty = imap.read_region(x=50000, y=50000, width=64, height=64)
        assert np.all(empty == 0)
        print(f"✅ Unwritten region returns zeros\n")

        # Demo 7: Neighbors
        print("Demo 7: Get neighboring regions")
        print("-" * 70)
        neighbors = imap.get_neighbors(x=0, y=0, radius=1)
        print(f"  Neighbors of (0, 0): {neighbors}")
        print(f"  Count: {len(neighbors)}\n")

        print("=" * 70)
        print("All demos completed successfully! ✅")
        print("=" * 70)

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)
        print(f"\nCleaned up: {tmpdir}")


if __name__ == "__main__":
    demo()
