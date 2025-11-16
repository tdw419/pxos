#!/usr/bin/env python3
"""
Unit tests for InfiniteMap

Tests the 2D spatial memory system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Import InfiniteMap
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from infinite_map import InfiniteMap, Tile, QuadTreeNode


class TestTile:
    """Tests for Tile data structure"""

    def test_tile_creation(self):
        """Test creating a tile"""
        tile = Tile(x=0, y=0, size=64)

        assert tile.x == 0
        assert tile.y == 0
        assert tile.size == 64
        assert tile.loaded is True
        assert tile.dirty is False
        assert tile.data is not None
        assert tile.data.shape == (64, 64, 3)

    def test_tile_pixel_bounds(self):
        """Test tile pixel bounds calculation"""
        tile = Tile(x=2, y=3, size=64)

        x1, y1, x2, y2 = tile.get_pixel_bounds()

        assert x1 == 128  # 2 * 64
        assert y1 == 192  # 3 * 64
        assert x2 == 192  # 128 + 64
        assert y2 == 256  # 192 + 64

    def test_tile_contains_pixel(self):
        """Test tile.contains_pixel"""
        tile = Tile(x=0, y=0, size=64)

        # Inside tile
        assert tile.contains_pixel(0, 0) is True
        assert tile.contains_pixel(32, 32) is True
        assert tile.contains_pixel(63, 63) is True

        # Outside tile
        assert tile.contains_pixel(64, 0) is False
        assert tile.contains_pixel(0, 64) is False
        assert tile.contains_pixel(-1, 0) is False

    def test_tile_local_coords(self):
        """Test converting world coords to local coords"""
        tile = Tile(x=2, y=3, size=64)

        local_x, local_y = tile.get_local_coords(130, 195)

        # World (130, 195) - tile origin (128, 192) = local (2, 3)
        assert local_x == 2
        assert local_y == 3


class TestQuadTree:
    """Tests for QuadTree spatial indexing"""

    def test_quadtree_insert(self):
        """Test inserting tiles into quadtree"""
        qt = QuadTreeNode(0, 0, 100, 100, max_tiles=4)

        # Insert tiles
        assert qt.insert(5, 5) is True
        assert qt.insert(10, 10) is True

        # Tiles should be stored
        assert len(qt.tiles) == 2

    def test_quadtree_splitting(self):
        """Test quadtree splits when full"""
        qt = QuadTreeNode(0, 0, 100, 100, max_tiles=2)

        # Insert enough tiles to trigger split
        qt.insert(1, 1)
        qt.insert(2, 2)
        qt.insert(3, 3)  # This should trigger split

        # Should have children now
        assert qt.children is not None
        assert len(qt.children) == 4

    def test_quadtree_query_region(self):
        """Test querying tiles in a region"""
        qt = QuadTreeNode(0, 0, 100, 100)

        # Insert tiles
        qt.insert(10, 10)
        qt.insert(50, 50)
        qt.insert(90, 90)

        # Query region that contains only some tiles
        results = qt.query_region(0, 0, 30, 30)

        # Should find only (10, 10)
        assert (10, 10) in results
        assert (50, 50) not in results


class TestInfiniteMapBasic:
    """Basic InfiniteMap functionality"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(tile_size=64, storage_path=temp_dir)

    def test_map_creation(self, map_instance):
        """Test creating an InfiniteMap"""
        assert map_instance is not None
        assert map_instance.tile_size == 64
        assert hasattr(map_instance, 'write_region')
        assert hasattr(map_instance, 'read_region')

    def test_pixel_to_tile_conversion(self, map_instance):
        """Test converting pixel coords to tile coords"""
        # Pixel (130, 195) should be in tile (2, 3) at local (2, 3)
        tile_x, tile_y, local_x, local_y = map_instance._pixel_to_tile(130, 195)

        assert tile_x == 2
        assert tile_y == 3
        assert local_x == 2
        assert local_y == 3

    def test_set_and_get_pixel(self, map_instance):
        """Test setting and getting individual pixels"""
        # Set a red pixel
        map_instance.set_pixel(100, 200, (255, 0, 0))

        # Read it back
        rgb = map_instance.get_pixel(100, 200)

        assert rgb == (255, 0, 0)

    def test_get_unset_pixel(self, map_instance):
        """Test reading unset pixels returns black"""
        rgb = map_instance.get_pixel(50000, 50000)

        assert rgb == (0, 0, 0)


class TestInfiniteMapRegions:
    """Tests for region read/write operations"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(tile_size=64, storage_path=temp_dir)

    def test_write_read_bytes(self, map_instance):
        """Test writing and reading raw bytes"""
        test_data = b"Hello, Infinite Map!"

        # Write bytes
        map_instance.write_region(0, 0, test_data)

        # Read back as bytes
        # Calculate read dimensions
        num_pixels = (len(test_data) + 2) // 3
        width = min(64, num_pixels)
        height = (num_pixels + width - 1) // width

        result = map_instance.read_region(0, 0, width, height, as_pixels=False)

        # Should match original (with padding)
        assert result[:len(test_data)] == test_data

    def test_write_read_pixels(self, map_instance):
        """Test writing and reading pixel arrays"""
        # Create test pixel data (red gradient)
        test_pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        test_pixels[:, :, 0] = np.arange(32)[:, np.newaxis]  # Red gradient

        # Write pixels
        map_instance.write_region(100, 100, test_pixels, source_is_pixels=True)

        # Read back
        result = map_instance.read_region(100, 100, 32, 32, as_pixels=True)

        np.testing.assert_array_equal(result, test_pixels)

    def test_sparse_writes(self, map_instance):
        """Test writing to distant locations"""
        # Write at origin
        map_instance.write_region(0, 0, b"Origin")

        # Write far away
        map_instance.write_region(10000, 20000, b"Far")

        # Both should be readable (need to read enough pixels to get all bytes)
        # "Origin" = 6 bytes = 2 pixels (3 bytes each), so read 3x1 to be safe
        origin = map_instance.read_region(0, 0, 3, 1, as_pixels=False)
        far = map_instance.read_region(10000, 20000, 2, 1, as_pixels=False)

        assert origin[:6] == b"Origin"
        assert far[:3] == b"Far"

    def test_large_region_write(self, map_instance):
        """Test writing region larger than one tile"""
        # Create 128x128 region (spans multiple 64x64 tiles)
        large_data = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        # Write it
        map_instance.write_region(0, 0, large_data, source_is_pixels=True)

        # Read it back
        result = map_instance.read_region(0, 0, 128, 128, as_pixels=True)

        np.testing.assert_array_equal(result, large_data)


class TestInfiniteMapNegativeCoords:
    """Tests for negative coordinate support"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(tile_size=64, storage_path=temp_dir)

    def test_negative_pixel_coords(self, map_instance):
        """Test negative pixel coordinates"""
        # Set pixel at negative coords
        map_instance.set_pixel(-100, -200, (100, 150, 200))

        # Read it back
        rgb = map_instance.get_pixel(-100, -200)

        assert rgb == (100, 150, 200)

    def test_negative_region_write(self, map_instance):
        """Test writing region at negative coordinates"""
        test_data = np.ones((16, 16, 3), dtype=np.uint8) * 128

        # Write at negative coords
        map_instance.write_region(-50, -50, test_data, source_is_pixels=True)

        # Read back
        result = map_instance.read_region(-50, -50, 16, 16)

        np.testing.assert_array_equal(result, test_data)

    def test_mixed_positive_negative(self, map_instance):
        """Test regions spanning positive and negative coordinates"""
        test_data = np.ones((64, 64, 3), dtype=np.uint8) * 200

        # Write centered on origin (spans -32 to +32)
        map_instance.write_region(-32, -32, test_data, source_is_pixels=True)

        # Read back
        result = map_instance.read_region(-32, -32, 64, 64)

        np.testing.assert_array_equal(result, test_data)


class TestInfiniteMapSpatialQueries:
    """Tests for spatial query operations"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(tile_size=64, storage_path=temp_dir)

    def test_get_neighbors(self, map_instance):
        """Test getting neighboring pixels"""
        # Set center pixel
        map_instance.set_pixel(100, 100, (255, 0, 0))

        # Set some neighbors
        map_instance.set_pixel(101, 100, (0, 255, 0))
        map_instance.set_pixel(100, 101, (0, 0, 255))

        # Get neighbors
        neighbors = map_instance.get_neighbors(100, 100, radius=1)

        # Should have 8 neighbors (3x3 grid minus center)
        assert len(neighbors) == 8

        # Check some neighbors are correct
        neighbor_dict = {(x, y): rgb for x, y, rgb in neighbors}
        assert neighbor_dict[(101, 100)] == (0, 255, 0)
        assert neighbor_dict[(100, 101)] == (0, 0, 255)

    def test_find_tiles_in_region(self, map_instance):
        """Test finding tiles that intersect a region"""
        # Write to create some tiles
        map_instance.write_region(0, 0, b"tile1")
        map_instance.write_region(128, 0, b"tile2")
        map_instance.write_region(0, 128, b"tile3")

        # Find tiles in large region
        tiles = map_instance.find_tiles_in_region(0, 0, 200, 200)

        # Should find tiles (0,0), (2,0), (0,2)
        # Note: 128 pixels / 64 tile_size = tile index 2
        assert len(tiles) >= 3


class TestInfiniteMapPersistence:
    """Tests for tile persistence and caching"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Don't clean up yet - needed for persistence tests

    def test_tile_save_load(self, temp_dir):
        """Test tiles are saved and loaded from disk"""
        # Create map and write data
        map1 = InfiniteMap(tile_size=64, storage_path=temp_dir)
        test_data = b"Persistent data!"
        map1.write_region(100, 100, test_data)

        # Flush to disk
        map1.flush()

        # Create new map instance
        map2 = InfiniteMap(tile_size=64, storage_path=temp_dir)

        # Read back - should load from disk
        result = map2.read_region(100, 100, 64, 1, as_pixels=False)

        assert result[:len(test_data)] == test_data

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_cache_eviction(self, temp_dir):
        """Test LRU cache eviction"""
        # Create map with small cache
        map_instance = InfiniteMap(tile_size=64, storage_path=temp_dir, cache_size=2)

        # Write to 3 different tiles (will trigger eviction)
        map_instance.write_region(0, 0, b"tile1")
        map_instance.write_region(128, 0, b"tile2")
        map_instance.write_region(256, 0, b"tile3")

        # Cache should only have 2 tiles
        assert len(map_instance.tiles) <= 2

        # But all data should still be accessible (will reload from disk if needed)
        # "tile1" = 5 bytes = 2 pixels, so read 2x1
        result1 = map_instance.read_region(0, 0, 2, 1, as_pixels=False)
        result2 = map_instance.read_region(128, 0, 2, 1, as_pixels=False)
        result3 = map_instance.read_region(256, 0, 2, 1, as_pixels=False)

        assert result1[:5] == b"tile1"
        assert result2[:5] == b"tile2"
        assert result3[:5] == b"tile3"

        # Cleanup
        shutil.rmtree(temp_dir)


class TestInfiniteMapStats:
    """Tests for map statistics and info"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(tile_size=64, storage_path=temp_dir)

    def test_get_stats(self, map_instance):
        """Test getting map statistics"""
        # Initial stats
        stats = map_instance.get_stats()

        assert 'tile_size' in stats
        assert 'tiles_in_memory' in stats
        assert 'tiles_on_disk' in stats

        assert stats['tile_size'] == 64
        assert stats['tiles_in_memory'] == 0

        # Write some data
        map_instance.write_region(0, 0, b"test")

        # Stats should update
        new_stats = map_instance.get_stats()
        assert new_stats['tiles_in_memory'] > 0

    def test_dirty_tiles_tracking(self, map_instance):
        """Test dirty tiles are tracked"""
        # Write data
        map_instance.write_region(0, 0, b"test")

        stats = map_instance.get_stats()

        # Should have dirty tiles
        assert stats['dirty_tiles'] > 0

        # Flush
        map_instance.flush()

        # No more dirty tiles
        new_stats = map_instance.get_stats()
        assert new_stats['dirty_tiles'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
