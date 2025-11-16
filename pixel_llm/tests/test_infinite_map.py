#!/usr/bin/env python3
"""
Comprehensive unit tests for InfiniteMap

Tests the 2D spatial memory system including:
- Tile management and coordinate conversion
- Quadtree spatial indexing
- Sparse storage and LRU caching
- Region read/write operations
- Persistence and manifest management
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Import InfiniteMap components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from infinite_map import InfiniteMap, Tile, QuadTreeNode


class TestTile:
    """Test Tile dataclass and coordinate operations"""

    def test_tile_initialization(self):
        """Test tile creates with default empty data"""
        tile = Tile(x=0, y=0, size=256)

        assert tile.x == 0
        assert tile.y == 0
        assert tile.size == 256
        assert tile.data is not None
        assert tile.data.shape == (256, 256, 3)
        assert tile.loaded is True
        assert tile.dirty is False

    def test_get_pixel_bounds(self):
        """Test pixel bounds calculation"""
        tile = Tile(x=2, y=3, size=256)
        x1, y1, x2, y2 = tile.get_pixel_bounds()

        assert x1 == 512  # 2 * 256
        assert y1 == 768  # 3 * 256
        assert x2 == 768  # 512 + 256
        assert y2 == 1024  # 768 + 256

    def test_contains_pixel(self):
        """Test pixel containment check"""
        tile = Tile(x=0, y=0, size=256)

        # Inside
        assert tile.contains_pixel(0, 0) is True
        assert tile.contains_pixel(100, 100) is True
        assert tile.contains_pixel(255, 255) is True

        # Outside
        assert tile.contains_pixel(256, 0) is False
        assert tile.contains_pixel(0, 256) is False
        assert tile.contains_pixel(-1, 0) is False

    def test_get_local_coords(self):
        """Test world to local coordinate conversion"""
        tile = Tile(x=2, y=3, size=256)

        # World coords (512, 768) should map to local (0, 0)
        local_x, local_y = tile.get_local_coords(512, 768)
        assert local_x == 0
        assert local_y == 0

        # World coords (612, 868) should map to local (100, 100)
        local_x, local_y = tile.get_local_coords(612, 868)
        assert local_x == 100
        assert local_y == 100


class TestQuadTreeNode:
    """Test QuadTree spatial indexing"""

    def test_insert_single_tile(self):
        """Test inserting a single tile"""
        tree = QuadTreeNode(0, 0, 100, 100, max_tiles=4)

        result = tree.insert(5, 5)
        assert result is True
        assert (5, 5) in tree.tiles

    def test_insert_out_of_bounds(self):
        """Test inserting outside bounds fails"""
        tree = QuadTreeNode(0, 0, 10, 10, max_tiles=4)

        result = tree.insert(20, 20)
        assert result is False

    def test_split_on_overflow(self):
        """Test quadtree splits when max_tiles exceeded"""
        tree = QuadTreeNode(0, 0, 100, 100, max_tiles=2)

        # Insert 3 tiles (exceeds max_tiles=2)
        tree.insert(10, 10)
        tree.insert(20, 20)
        tree.insert(30, 30)

        # Should have split into children
        assert tree.children is not None
        assert len(tree.children) == 4
        assert len(tree.tiles) == 0  # Tiles moved to children

    def test_query_region_simple(self):
        """Test querying tiles in a region"""
        tree = QuadTreeNode(0, 0, 100, 100, max_tiles=10)

        tree.insert(5, 5)
        tree.insert(10, 10)
        tree.insert(90, 90)

        # Query region that contains only first two tiles
        tiles = tree.query_region(0, 0, 20, 20)
        assert len(tiles) == 2
        assert (5, 5) in tiles
        assert (10, 10) in tiles
        assert (90, 90) not in tiles

    def test_query_region_empty(self):
        """Test querying empty region"""
        tree = QuadTreeNode(0, 0, 100, 100, max_tiles=10)
        tree.insert(50, 50)

        # Query region with no tiles
        tiles = tree.query_region(0, 0, 10, 10)
        assert len(tiles) == 0

    def test_query_region_after_split(self):
        """Test querying after quadtree split"""
        tree = QuadTreeNode(0, 0, 100, 100, max_tiles=2)

        tree.insert(10, 10)
        tree.insert(20, 20)
        tree.insert(30, 30)
        tree.insert(80, 80)

        # Should work even after split
        tiles = tree.query_region(0, 0, 40, 40)
        assert len(tiles) == 3  # First three tiles


class TestInfiniteMap:
    """Test InfiniteMap core operations"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def imap(self, temp_storage):
        """Create InfiniteMap instance"""
        return InfiniteMap(tile_size=64, storage_path=temp_storage / "map")

    def test_pixel_to_tile_positive(self, imap):
        """Test pixel to tile coordinate conversion (positive coords)"""
        tile_x, tile_y, local_x, local_y = imap._pixel_to_tile(100, 200)

        assert tile_x == 1  # 100 // 64
        assert tile_y == 3  # 200 // 64
        assert local_x == 36  # 100 % 64
        assert local_y == 8  # 200 % 64

    def test_pixel_to_tile_negative(self, imap):
        """Test pixel to tile coordinate conversion (negative coords)"""
        tile_x, tile_y, local_x, local_y = imap._pixel_to_tile(-10, -20)

        # Negative coords should map to negative tiles
        assert tile_x == -1
        assert tile_y == -1
        assert local_x == 54  # 64 - 10
        assert local_y == 44  # 64 - 20

    def test_set_get_pixel(self, imap):
        """Test basic pixel read/write"""
        # Set pixel
        imap.set_pixel(10, 20, (255, 128, 64))

        # Get pixel
        rgb = imap.get_pixel(10, 20)
        assert rgb == (255, 128, 64)

    def test_get_pixel_unallocated(self, imap):
        """Test getting pixel from unallocated region returns black"""
        rgb = imap.get_pixel(10000, 20000)
        assert rgb == (0, 0, 0)

    def test_write_read_region_bytes(self, imap):
        """Test writing and reading region with byte data"""
        test_data = b"Hello, Infinite World!"

        # Write
        imap.write_region(0, 0, test_data)

        # Read back
        region = imap.read_region(0, 0, 64, 1, as_pixels=False)
        read_data = region[:len(test_data)]

        assert read_data == test_data

    def test_write_read_region_pixels(self, imap):
        """Test writing and reading region with pixel array"""
        # Create 10x10 test pattern
        test_pixels = np.zeros((10, 10, 3), dtype=np.uint8)
        test_pixels[0:5, 0:5] = [255, 0, 0]  # Red square
        test_pixels[5:10, 5:10] = [0, 255, 0]  # Green square

        # Write
        imap.write_region(100, 100, test_pixels, source_is_pixels=True)

        # Read back
        region = imap.read_region(100, 100, 10, 10, as_pixels=True)

        assert np.array_equal(region, test_pixels)

    def test_write_distant_regions(self, imap):
        """Test writing to widely separated regions (sparse storage)"""
        # Write at origin
        imap.write_region(0, 0, b"Origin")

        # Write far away
        imap.write_region(10000, 20000, b"Distant")

        # Both should be readable
        origin_data = imap.read_region(0, 0, 64, 1, as_pixels=False)
        distant_data = imap.read_region(10000, 20000, 64, 1, as_pixels=False)

        assert origin_data[:6] == b"Origin"
        assert distant_data[:7] == b"Distant"

    def test_get_neighbors(self, imap):
        """Test neighbor pixel retrieval"""
        # Set center pixel
        imap.set_pixel(50, 50, (255, 0, 0))

        # Set some neighbors
        imap.set_pixel(51, 50, (0, 255, 0))
        imap.set_pixel(50, 51, (0, 0, 255))

        # Get neighbors with radius=1
        neighbors = imap.get_neighbors(50, 50, radius=1)

        # Should have 8 neighbors (3x3 grid - center)
        assert len(neighbors) == 8

        # Check that our set neighbors are in the list
        neighbor_dict = {(x, y): rgb for x, y, rgb in neighbors}
        assert neighbor_dict[(51, 50)] == (0, 255, 0)
        assert neighbor_dict[(50, 51)] == (0, 0, 255)

    def test_find_tiles_in_region(self, imap):
        """Test finding tiles that intersect a region"""
        # Write data to create some tiles
        imap.write_region(0, 0, b"tile1")
        imap.write_region(100, 100, b"tile2")

        # Find tiles in region covering both
        tiles = imap.find_tiles_in_region(0, 0, 200, 200)

        # Should find at least the tiles we created
        assert len(tiles) >= 2

    def test_lru_cache_eviction(self, temp_storage):
        """Test that LRU cache evicts old tiles"""
        # Create map with small cache
        imap = InfiniteMap(
            tile_size=64,
            storage_path=temp_storage / "map",
            cache_size=2
        )

        # Create 3 tiles (exceeds cache_size=2)
        imap.set_pixel(0, 0, (255, 0, 0))      # Tile (0, 0)
        imap.set_pixel(100, 100, (0, 255, 0))  # Tile (1, 1)
        imap.set_pixel(200, 200, (0, 0, 255))  # Tile (3, 3)

        # Cache should only have 2 tiles
        assert len(imap.tiles) <= 2

    def test_persistence_flush(self, imap):
        """Test saving tiles to disk"""
        # Write data
        imap.write_region(0, 0, b"Persistent data")

        # Flush to disk
        imap.flush()

        # Check that tile file was created
        tile_files = list(imap.storage_path.glob("tile_*.pkl"))
        assert len(tile_files) > 0

    def test_persistence_reload(self, temp_storage):
        """Test loading tiles from disk"""
        storage_path = temp_storage / "map"

        # Create map and write data
        imap1 = InfiniteMap(tile_size=64, storage_path=storage_path)
        test_data = b"Reload test data"
        imap1.write_region(0, 0, test_data)
        imap1.flush()

        # Create new map instance (simulates restart)
        imap2 = InfiniteMap(tile_size=64, storage_path=storage_path)

        # Should be able to read the data
        region = imap2.read_region(0, 0, 64, 1, as_pixels=False)
        assert region[:len(test_data)] == test_data

    def test_get_stats(self, imap):
        """Test statistics reporting"""
        # Write some data
        imap.write_region(0, 0, b"test")
        imap.write_region(100, 100, b"test2")

        stats = imap.get_stats()

        assert "tile_size" in stats
        assert "tiles_in_memory" in stats
        assert "tiles_on_disk" in stats
        assert stats["tile_size"] == 64
        assert stats["tiles_in_memory"] >= 1


class TestInfiniteMapEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def imap(self):
        temp_dir = tempfile.mkdtemp()
        map_instance = InfiniteMap(tile_size=64, storage_path=Path(temp_dir) / "map")
        yield map_instance
        shutil.rmtree(temp_dir)

    def test_empty_region_write(self, imap):
        """Test writing empty data"""
        empty_data = b""
        imap.write_region(0, 0, empty_data)

        # Should not crash
        stats = imap.get_stats()
        assert stats is not None

    def test_large_region(self, imap):
        """Test writing/reading large region"""
        # 10KB of data
        large_data = b"x" * 10000

        imap.write_region(0, 0, large_data)

        # Calculate expected region size
        # 10000 bytes = 3334 pixels, with tile_size=64: 64 wide x 53 tall
        num_pixels = (len(large_data) + 2) // 3
        width = min(imap.tile_size, num_pixels)
        height = (num_pixels + width - 1) // width

        region = imap.read_region(0, 0, width, height, as_pixels=False)

        # Should be able to read back (might have padding)
        assert region[:len(large_data)] == large_data

    def test_negative_coordinates(self, imap):
        """Test operations with negative coordinates"""
        # Write at negative coords
        imap.set_pixel(-100, -100, (255, 128, 64))

        # Read back
        rgb = imap.get_pixel(-100, -100)
        assert rgb == (255, 128, 64)

    def test_tile_boundary_write(self, imap):
        """Test writing across tile boundaries"""
        # Tile size is 64, so write from (60, 60) to cross boundary
        test_pattern = np.ones((10, 10, 3), dtype=np.uint8) * 255

        imap.write_region(60, 60, test_pattern, source_is_pixels=True)

        # Read back
        region = imap.read_region(60, 60, 10, 10, as_pixels=True)
        assert np.array_equal(region, test_pattern)

    def test_multiple_flushes(self, imap):
        """Test multiple flush operations"""
        imap.write_region(0, 0, b"data1")
        imap.flush()

        imap.write_region(100, 100, b"data2")
        imap.flush()

        # Both should be on disk
        stats = imap.get_stats()
        assert stats["tiles_on_disk"] >= 2

    def test_zero_radius_neighbors(self, imap):
        """Test get_neighbors with radius=0"""
        imap.set_pixel(50, 50, (255, 0, 0))

        # Radius 0 should return no neighbors (just excludes center)
        neighbors = imap.get_neighbors(50, 50, radius=0)
        assert len(neighbors) == 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
