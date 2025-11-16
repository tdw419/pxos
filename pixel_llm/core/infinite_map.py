#!/usr/bin/env python3
"""
Infinite Map: 2D Spatial Memory System

A theoretically infinite 2D coordinate space where data is organized spatially.
Perfect for:
- Storing LLM weights with spatial relationships
- Organizing attention heads as pixel neighborhoods
- Enabling spatial reasoning for the AI
- Visual navigation of memory

Key Features:
    - Sparse storage (only allocated regions use memory)
    - Quadtree-based spatial indexing
    - Tile-based loading/unloading
    - Pixel-aware operations
    - Spatial queries (neighbors, regions, etc.)

Example:
    >>> map = InfiniteMap(tile_size=256)
    >>> map.write_region(0, 0, pixel_data)  # Write to origin
    >>> map.write_region(1000, 2000, more_data)  # Write far away
    >>> neighbors = map.get_neighbors(500, 500, radius=10)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set, Union
from dataclasses import dataclass
from collections import defaultdict
import pickle


@dataclass
class Tile:
    """
    A tile in the infinite map.

    Each tile is a fixed-size chunk of the 2D space.
    Tiles are loaded on-demand and can be persisted.
    """
    x: int  # Tile coordinates (not pixel coordinates)
    y: int
    size: int  # Tile size in pixels (e.g., 256x256)
    data: Optional[np.ndarray] = None  # RGB data: shape (size, size, 3)
    dirty: bool = False  # Needs to be saved
    loaded: bool = False

    def __post_init__(self):
        if self.data is None:
            # Create empty tile (black)
            self.data = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            self.loaded = True

    def get_pixel_bounds(self) -> Tuple[int, int, int, int]:
        """Get pixel bounds of this tile (x1, y1, x2, y2)"""
        x1 = self.x * self.size
        y1 = self.y * self.size
        x2 = x1 + self.size
        y2 = y1 + self.size
        return x1, y1, x2, y2

    def contains_pixel(self, px: int, py: int) -> bool:
        """Check if pixel coordinate is within this tile"""
        x1, y1, x2, y2 = self.get_pixel_bounds()
        return x1 <= px < x2 and y1 <= py < y2

    def get_local_coords(self, px: int, py: int) -> Tuple[int, int]:
        """Convert world pixel coords to tile-local coords"""
        x1, y1, _, _ = self.get_pixel_bounds()
        return px - x1, py - y1


class QuadTreeNode:
    """
    Quadtree node for spatial indexing.

    Enables fast spatial queries like:
    - Find all tiles in a region
    - Find nearest neighbors
    - Range queries
    """

    def __init__(self, x: int, y: int, width: int, height: int, max_tiles: int = 4):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_tiles = max_tiles
        self.tiles: Set[Tuple[int, int]] = set()
        self.children: Optional[List['QuadTreeNode']] = None

    def insert(self, tile_x: int, tile_y: int) -> bool:
        """Insert a tile coordinate into the quadtree"""
        # Check if tile is in bounds
        if not (self.x <= tile_x < self.x + self.width and
                self.y <= tile_y < self.y + self.height):
            return False

        # If we have children, insert into appropriate child
        if self.children:
            for child in self.children:
                if child.insert(tile_x, tile_y):
                    return True
            return False

        # Add to this node
        self.tiles.add((tile_x, tile_y))

        # Split if we exceed max_tiles
        if len(self.tiles) > self.max_tiles and self.width > 1 and self.height > 1:
            self._split()

        return True

    def _split(self):
        """Split this node into 4 children"""
        hw = self.width // 2
        hh = self.height // 2

        self.children = [
            QuadTreeNode(self.x, self.y, hw, hh, self.max_tiles),  # Top-left
            QuadTreeNode(self.x + hw, self.y, self.width - hw, hh, self.max_tiles),  # Top-right
            QuadTreeNode(self.x, self.y + hh, hw, self.height - hh, self.max_tiles),  # Bottom-left
            QuadTreeNode(self.x + hw, self.y + hh, self.width - hw, self.height - hh, self.max_tiles),  # Bottom-right
        ]

        # Redistribute tiles to children
        for tile_x, tile_y in self.tiles:
            for child in self.children:
                child.insert(tile_x, tile_y)

        self.tiles.clear()

    def query_region(self, x: int, y: int, width: int, height: int) -> Set[Tuple[int, int]]:
        """Find all tiles in a region"""
        result = set()

        # Check if region overlaps this node
        if not (x < self.x + self.width and x + width > self.x and
                y < self.y + self.height and y + height > self.y):
            return result

        # If we have children, query them
        if self.children:
            for child in self.children:
                result.update(child.query_region(x, y, width, height))
        else:
            # Check tiles in this node
            for tile_x, tile_y in self.tiles:
                if (x <= tile_x < x + width and y <= tile_y < y + height):
                    result.add((tile_x, tile_y))

        return result


class InfiniteMap:
    """
    Infinite 2D pixel space with sparse storage.

    The map is divided into tiles for efficient memory usage.
    Only allocated tiles consume memory.
    """

    def __init__(
        self,
        tile_size: int = 256,
        storage_path: Optional[Path] = None,
        cache_size: int = 100
    ):
        """
        Initialize infinite map.

        Args:
            tile_size: Size of each tile in pixels
            storage_path: Path to persist tiles
            cache_size: Maximum number of tiles to keep in memory
        """
        self.tile_size = tile_size
        self.storage_path = Path(storage_path) if storage_path else Path("pixel_llm/data/infinite_map")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.cache_size = cache_size
        self.tiles: Dict[Tuple[int, int], Tile] = {}  # (tile_x, tile_y) -> Tile
        self.access_order: List[Tuple[int, int]] = []  # LRU cache

        # Spatial index
        self.quadtree = QuadTreeNode(-1000000, -1000000, 2000000, 2000000)

        # Load manifest (which tiles exist on disk)
        self._load_manifest()

    def _load_manifest(self):
        """Load tile manifest from disk"""
        manifest_path = self.storage_path / "manifest.pkl"
        if manifest_path.exists():
            with open(manifest_path, 'rb') as f:
                tile_coords = pickle.load(f)
                for tx, ty in tile_coords:
                    self.quadtree.insert(tx, ty)

    def _save_manifest(self):
        """Save tile manifest to disk"""
        # Collect all tile coordinates (in memory + known on disk)
        all_tiles = set(self.tiles.keys())

        # Add tiles we know exist on disk
        for tile_file in self.storage_path.glob("tile_*.pkl"):
            parts = tile_file.stem.split('_')
            if len(parts) == 3:
                tx, ty = int(parts[1]), int(parts[2])
                all_tiles.add((tx, ty))

        manifest_path = self.storage_path / "manifest.pkl"
        with open(manifest_path, 'wb') as f:
            pickle.dump(list(all_tiles), f)

    def _pixel_to_tile(self, px: int, py: int) -> Tuple[int, int, int, int]:
        """
        Convert pixel coordinates to tile coordinates + local offset.

        Returns:
            (tile_x, tile_y, local_x, local_y)
        """
        tile_x = px // self.tile_size
        tile_y = py // self.tile_size
        local_x = px % self.tile_size
        local_y = py % self.tile_size

        # Handle negative coordinates
        if px < 0 and local_x != 0:
            tile_x -= 1
            local_x = self.tile_size + local_x

        if py < 0 and local_y != 0:
            tile_y -= 1
            local_y = self.tile_size + local_y

        return tile_x, tile_y, local_x, local_y

    def _get_tile(self, tile_x: int, tile_y: int, create: bool = True) -> Optional[Tile]:
        """
        Get or load a tile.

        Args:
            tile_x, tile_y: Tile coordinates
            create: Create tile if it doesn't exist

        Returns:
            Tile object or None
        """
        tile_key = (tile_x, tile_y)

        # Check cache
        if tile_key in self.tiles:
            # Update LRU
            if tile_key in self.access_order:
                self.access_order.remove(tile_key)
            self.access_order.append(tile_key)
            return self.tiles[tile_key]

        # Try to load from disk
        tile = self._load_tile(tile_x, tile_y)

        if tile is None and create:
            # Create new tile
            tile = Tile(tile_x, tile_y, self.tile_size)
            tile.dirty = True

        if tile:
            # Add to cache
            self.tiles[tile_key] = tile
            self.access_order.append(tile_key)
            self.quadtree.insert(tile_x, tile_y)

            # Evict old tiles if cache is full
            self._evict_if_needed()

        return tile

    def _load_tile(self, tile_x: int, tile_y: int) -> Optional[Tile]:
        """Load tile from disk"""
        tile_path = self.storage_path / f"tile_{tile_x}_{tile_y}.pkl"

        if not tile_path.exists():
            return None

        try:
            with open(tile_path, 'rb') as f:
                tile = pickle.load(f)
                tile.loaded = True
                tile.dirty = False
                return tile
        except Exception as e:
            print(f"Warning: Failed to load tile ({tile_x}, {tile_y}): {e}")
            return None

    def _save_tile(self, tile: Tile):
        """Save tile to disk"""
        if not tile.dirty:
            return

        tile_path = self.storage_path / f"tile_{tile.x}_{tile.y}.pkl"

        try:
            with open(tile_path, 'wb') as f:
                pickle.dump(tile, f)
            tile.dirty = False
        except Exception as e:
            print(f"Warning: Failed to save tile ({tile.x}, {tile.y}): {e}")

    def _evict_if_needed(self):
        """Evict least recently used tiles if cache is full"""
        while len(self.tiles) > self.cache_size:
            # Get least recently used tile
            lru_key = self.access_order.pop(0)
            tile = self.tiles[lru_key]

            # Save if dirty
            if tile.dirty:
                self._save_tile(tile)

            # Remove from cache
            del self.tiles[lru_key]

    def get_pixel(self, px: int, py: int) -> Tuple[int, int, int]:
        """
        Get RGB value at pixel coordinate.

        Returns:
            (R, G, B) tuple
        """
        tile_x, tile_y, local_x, local_y = self._pixel_to_tile(px, py)
        tile = self._get_tile(tile_x, tile_y, create=False)

        if tile is None:
            return (0, 0, 0)  # Empty space is black

        return tuple(tile.data[local_y, local_x])

    def set_pixel(self, px: int, py: int, rgb: Tuple[int, int, int]):
        """Set RGB value at pixel coordinate"""
        tile_x, tile_y, local_x, local_y = self._pixel_to_tile(px, py)
        tile = self._get_tile(tile_x, tile_y, create=True)

        tile.data[local_y, local_x] = rgb
        tile.dirty = True

    def write_region(
        self,
        px: int,
        py: int,
        data: np.ndarray,
        source_is_pixels: bool = False
    ):
        """
        Write a region of data to the map.

        Args:
            px, py: Top-left pixel coordinate
            data: Either pixel array (H, W, 3) or raw bytes
            source_is_pixels: If True, data is already pixel array
        """
        if not source_is_pixels:
            # Convert bytes to pixels
            # Each pixel stores 3 bytes (RGB)
            num_pixels = (len(data) + 2) // 3
            width = min(self.tile_size, num_pixels)
            height = (num_pixels + width - 1) // width

            # Pad to full size
            padded = data + b'\x00' * (width * height * 3 - len(data))
            pixel_array = np.frombuffer(padded, dtype=np.uint8)
            pixel_array = pixel_array.reshape((height, width, 3))
        else:
            pixel_array = data

        height, width, _ = pixel_array.shape

        # Write pixels tile by tile
        for y in range(height):
            for x in range(width):
                world_x = px + x
                world_y = py + y
                rgb = tuple(pixel_array[y, x])
                self.set_pixel(world_x, world_y, rgb)

    def read_region(
        self,
        px: int,
        py: int,
        width: int,
        height: int,
        as_pixels: bool = True
    ) -> Union[np.ndarray, bytes]:
        """
        Read a region from the map.

        Args:
            px, py: Top-left coordinate
            width, height: Region size
            as_pixels: Return pixel array vs raw bytes

        Returns:
            Pixel array (H, W, 3) or bytes
        """
        # Create output array
        region = np.zeros((height, width, 3), dtype=np.uint8)

        # Read pixels
        for y in range(height):
            for x in range(width):
                world_x = px + x
                world_y = py + y
                rgb = self.get_pixel(world_x, world_y)
                region[y, x] = rgb

        if as_pixels:
            return region
        else:
            # Flatten to bytes
            return region.tobytes()

    def get_neighbors(
        self,
        px: int,
        py: int,
        radius: int = 1
    ) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """
        Get neighboring pixels.

        Returns:
            List of (x, y, (r, g, b)) tuples
        """
        neighbors = []

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                x = px + dx
                y = py + dy
                rgb = self.get_pixel(x, y)
                neighbors.append((x, y, rgb))

        return neighbors

    def find_tiles_in_region(
        self,
        px: int,
        py: int,
        width: int,
        height: int
    ) -> List[Tuple[int, int]]:
        """Find all tiles that intersect a pixel region"""
        # Convert to tile coordinates
        tile_x1 = px // self.tile_size
        tile_y1 = py // self.tile_size
        tile_x2 = (px + width) // self.tile_size
        tile_y2 = (py + height) // self.tile_size

        # Query quadtree
        return list(self.quadtree.query_region(
            tile_x1, tile_y1,
            tile_x2 - tile_x1 + 1,
            tile_y2 - tile_y1 + 1
        ))

    def flush(self):
        """Save all dirty tiles to disk"""
        for tile in self.tiles.values():
            if tile.dirty:
                self._save_tile(tile)

        self._save_manifest()

    def get_stats(self) -> Dict:
        """Get map statistics"""
        num_tiles_in_memory = len(self.tiles)
        num_dirty = sum(1 for t in self.tiles.values() if t.dirty)

        # Count tiles on disk
        num_on_disk = len(list(self.storage_path.glob("tile_*.pkl")))

        return {
            "tile_size": self.tile_size,
            "tiles_in_memory": num_tiles_in_memory,
            "tiles_on_disk": num_on_disk,
            "dirty_tiles": num_dirty,
            "cache_usage": f"{num_tiles_in_memory}/{self.cache_size}",
        }


# CLI for testing
if __name__ == "__main__":
    print("\n=== Infinite Map Demo ===\n")

    # Create map
    map = InfiniteMap(tile_size=64)

    # Write some data at origin
    print("Writing 'Hello World' at origin (0, 0)...")
    data = b"Hello, Infinite Pixel World!"
    map.write_region(0, 0, data)

    # Write data far away
    print("Writing data at (10000, 20000)...")
    far_data = b"This data is stored 10,000+ pixels away!"
    map.write_region(10000, 20000, far_data)

    # Read back
    print("\nReading from origin...")
    region = map.read_region(0, 0, 64, 1, as_pixels=False)
    text = region[:len(data)].decode('utf-8', errors='ignore')
    print(f"  Got: {text}")

    # Test neighbors
    print("\nGetting neighbors of (5, 0)...")
    neighbors = map.get_neighbors(5, 0, radius=1)
    print(f"  Found {len(neighbors)} neighbors")

    # Stats
    print("\nMap statistics:")
    stats = map.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save
    print("\nFlushing to disk...")
    map.flush()

    print("\n✓ Demo complete!")
