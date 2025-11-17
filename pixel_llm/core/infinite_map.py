#!/usr/bin/env python3
"""
Infinite Map: 2D Spatial Memory System
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set, Union
from dataclasses import dataclass
from collections import defaultdict
import pickle

@dataclass
class Tile:
    x: int
    y: int
    data: np.ndarray
    dirty: bool = True

class InfiniteMap:
    def __init__(self, storage_path: str, tile_size: int = 256, cache_size: int = 100):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.tile_size = tile_size
        self.cache_size = cache_size
        self.tile_cache: Dict[Tuple[int, int], Tile] = {}
        self.lru: List[Tuple[int, int]] = []

    def _get_tile_coords(self, x: int, y: int) -> Tuple[int, int]:
        return x // self.tile_size, y // self.tile_size

    def _get_tile(self, tx: int, ty: int) -> Tile:
        if (tx, ty) in self.tile_cache:
            self.lru.remove((tx, ty))
            self.lru.append((tx, ty))
            return self.tile_cache[(tx, ty)]

        tile_path = self.storage_path / f"{tx}_{ty}.tile"
        if tile_path.exists():
            with tile_path.open('rb') as f:
                data = pickle.load(f)
        else:
            data = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)

        tile = Tile(x=tx, y=ty, data=data, dirty=False)
        self._add_to_cache(tile)
        return tile

    def _add_to_cache(self, tile: Tile):
        if len(self.tile_cache) >= self.cache_size:
            oldest_coords = self.lru.pop(0)
            oldest_tile = self.tile_cache.pop(oldest_coords)
            if oldest_tile.dirty:
                self._flush_tile(oldest_tile)

        self.tile_cache[(tile.x, tile.y)] = tile
        self.lru.append((tile.x, tile.y))

    def _flush_tile(self, tile: Tile):
        tile_path = self.storage_path / f"{tile.x}_{tile.y}.tile"
        with tile_path.open('wb') as f:
            pickle.dump(tile.data, f)
        tile.dirty = False

    def flush(self):
        for tile in self.tile_cache.values():
            if tile.dirty:
                self._flush_tile(tile)

    def write_region(self, x: int, y: int, data: np.ndarray):
        tx, ty = self._get_tile_coords(x, y)
        tile = self._get_tile(tx, ty)

        rel_x, rel_y = x % self.tile_size, y % self.tile_size
        h, w, _ = data.shape

        tile.data[rel_y:rel_y+h, rel_x:rel_x+w] = data
        tile.dirty = True

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        tx, ty = self._get_tile_coords(x, y)
        tile = self._get_tile(tx, ty)

        rel_x, rel_y = x % self.tile_size, y % self.tile_size
        return tile.data[rel_y:rel_y+height, rel_x:rel_x+width]

def demo():
    print("=== Infinite Map Demo ===")
    map_storage = "map_storage_demo"
    infinite_map = InfiniteMap(map_storage)

    test_data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    infinite_map.write_region(10000, 20000, test_data)

    read_back = infinite_map.read_region(10000, 20000, 32, 32)
    assert np.array_equal(test_data, read_back)

    infinite_map.flush()
    print("Flushing to disk...")

    new_map = InfiniteMap(map_storage)
    read_again = new_map.read_region(10000, 20000, 32, 32)
    assert np.array_equal(test_data, read_again)

    print("✓ Demo complete!")

if __name__ == '__main__':
    demo()
