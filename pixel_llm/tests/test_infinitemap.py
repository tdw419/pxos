#!/usr/bin/env python3
"""
Unit tests for InfiniteMap

Tests the 2D spatial memory system that manages pixel regions.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Import InfiniteMap
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from infinite_map import InfiniteMap, MapRegion


class TestMapRegion:
    """Tests for MapRegion data structure"""

    def test_region_creation(self):
        """Test creating a region with valid coordinates"""
        region = MapRegion(
            x=100, y=200,
            width=64, height=64,
            data=np.zeros((64, 64, 3), dtype=np.uint8)
        )

        assert region.x == 100
        assert region.y == 200
        assert region.width == 64
        assert region.height == 64
        assert region.data.shape == (64, 64, 3)

    def test_region_bounds(self):
        """Test region boundary calculations"""
        region = MapRegion(
            x=100, y=200,
            width=64, height=64,
            data=np.zeros((64, 64, 3), dtype=np.uint8)
        )

        # Right edge should be x + width - 1
        assert region.x + region.width - 1 == 163
        # Bottom edge should be y + height - 1
        assert region.y + region.height - 1 == 263


class TestInfiniteMapCore:
    """Core InfiniteMap functionality tests"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test map"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        """Create a fresh InfiniteMap for each test"""
        return InfiniteMap(storage_dir=temp_dir)

    def test_map_creation(self, map_instance):
        """Test creating an InfiniteMap"""
        assert map_instance is not None
        assert hasattr(map_instance, 'write_region')
        assert hasattr(map_instance, 'read_region')

    def test_write_single_region(self, map_instance):
        """Test writing a single region"""
        # Create test data (64x64 red pixels)
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        data[:, :, 0] = 255  # Red channel

        # Write at origin
        map_instance.write_region(x=0, y=0, data=data)

        # Verify we can read it back
        result = map_instance.read_region(x=0, y=0, width=64, height=64)
        assert result is not None
        assert result.shape == (64, 64, 3)
        np.testing.assert_array_equal(result, data)

    def test_write_multiple_regions(self, map_instance):
        """Test writing multiple non-overlapping regions"""
        # Red region at (0, 0)
        red_data = np.zeros((64, 64, 3), dtype=np.uint8)
        red_data[:, :, 0] = 255

        # Green region at (100, 0)
        green_data = np.zeros((64, 64, 3), dtype=np.uint8)
        green_data[:, :, 1] = 255

        # Blue region at (0, 100)
        blue_data = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_data[:, :, 2] = 255

        # Write all regions
        map_instance.write_region(x=0, y=0, data=red_data)
        map_instance.write_region(x=100, y=0, data=green_data)
        map_instance.write_region(x=0, y=100, data=blue_data)

        # Read back and verify
        red_result = map_instance.read_region(x=0, y=0, width=64, height=64)
        green_result = map_instance.read_region(x=100, y=0, width=64, height=64)
        blue_result = map_instance.read_region(x=0, y=100, width=64, height=64)

        np.testing.assert_array_equal(red_result, red_data)
        np.testing.assert_array_equal(green_result, green_data)
        np.testing.assert_array_equal(blue_result, blue_data)

    def test_read_unwritten_region(self, map_instance):
        """Test reading from unwritten region returns zeros"""
        # Read from region that was never written
        result = map_instance.read_region(x=1000, y=1000, width=64, height=64)

        # Should return zeros (black pixels)
        expected = np.zeros((64, 64, 3), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_overwrite_region(self, map_instance):
        """Test overwriting existing region"""
        # Write red
        red_data = np.zeros((64, 64, 3), dtype=np.uint8)
        red_data[:, :, 0] = 255
        map_instance.write_region(x=0, y=0, data=red_data)

        # Overwrite with blue
        blue_data = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_data[:, :, 2] = 255
        map_instance.write_region(x=0, y=0, data=blue_data)

        # Should read blue
        result = map_instance.read_region(x=0, y=0, width=64, height=64)
        np.testing.assert_array_equal(result, blue_data)

    def test_negative_coordinates(self, map_instance):
        """Test writing/reading with negative coordinates"""
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        data[:, :, 0] = 128

        # Write at negative coordinates
        map_instance.write_region(x=-100, y=-100, data=data)

        # Read back
        result = map_instance.read_region(x=-100, y=-100, width=64, height=64)
        np.testing.assert_array_equal(result, data)


class TestInfiniteMapSpatial:
    """Tests for spatial operations"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(storage_dir=temp_dir)

    def test_get_neighbors(self, map_instance):
        """Test getting neighboring regions"""
        # Write center region
        center = np.zeros((64, 64, 3), dtype=np.uint8)
        center[:, :, 0] = 255  # Red
        map_instance.write_region(x=0, y=0, data=center)

        # Write north neighbor
        north = np.zeros((64, 64, 3), dtype=np.uint8)
        north[:, :, 1] = 255  # Green
        map_instance.write_region(x=0, y=-64, data=north)

        # Get neighbors (if implemented)
        if hasattr(map_instance, 'get_neighbors'):
            neighbors = map_instance.get_neighbors(x=0, y=0, radius=1)
            # Should find at least the north neighbor
            assert len(neighbors) >= 1

    def test_large_sparse_map(self, map_instance):
        """Test map with sparse distant regions"""
        # Write regions far apart
        positions = [
            (0, 0),
            (10000, 0),
            (0, 10000),
            (-5000, -5000)
        ]

        for i, (x, y) in enumerate(positions):
            data = np.zeros((64, 64, 3), dtype=np.uint8)
            data[:, :, i % 3] = (i + 1) * 50  # Unique color per region
            map_instance.write_region(x=x, y=y, data=data)

        # Verify all can be read back
        for i, (x, y) in enumerate(positions):
            result = map_instance.read_region(x=x, y=y, width=64, height=64)
            assert result is not None
            # Check the distinctive color is present
            assert result[:, :, i % 3].max() > 0

    def test_partial_overlap(self, map_instance):
        """Test reading region that partially overlaps written data"""
        # Write a 64x64 region at (0, 0)
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        data[:, :, 0] = 255
        map_instance.write_region(x=0, y=0, data=data)

        # Read a region that overlaps: (32, 32, 96, 96)
        # This spans both written and unwritten areas
        result = map_instance.read_region(x=32, y=32, width=96, height=96)

        assert result.shape == (96, 96, 3)
        # Top-left should be red (from written region)
        assert result[0, 0, 0] > 0  # Some red present
        # Bottom-right should be black (unwritten)
        # (exact behavior depends on implementation)


class TestInfiniteMapEdgeCases:
    """Edge cases and error conditions"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(storage_dir=temp_dir)

    def test_zero_size_read(self, map_instance):
        """Test reading zero-sized region"""
        # Depending on implementation, might raise error or return empty array
        try:
            result = map_instance.read_region(x=0, y=0, width=0, height=0)
            # If it returns, should be empty
            assert result.size == 0
        except ValueError:
            # It's also valid to raise an error
            pass

    def test_very_large_region(self, map_instance):
        """Test with very large region (tests memory handling)"""
        # Create 1024x1024 region (3MB)
        large_data = np.zeros((1024, 1024, 3), dtype=np.uint8)
        large_data[::10, ::10, :] = 255  # Sparse pattern

        # Write
        map_instance.write_region(x=0, y=0, data=large_data)

        # Read back
        result = map_instance.read_region(x=0, y=0, width=1024, height=1024)
        assert result.shape == (1024, 1024, 3)
        # Verify pattern is preserved
        assert result[0, 0, 0] == 255
        assert result[5, 5, 0] == 0

    def test_persistence(self, temp_dir):
        """Test that data persists across map instances"""
        # Create first map and write data
        map1 = InfiniteMap(storage_dir=temp_dir)
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        data[:, :, 0] = 200
        map1.write_region(x=0, y=0, data=data)

        # Create second map pointing to same storage
        map2 = InfiniteMap(storage_dir=temp_dir)
        result = map2.read_region(x=0, y=0, width=64, height=64)

        # Data should persist
        np.testing.assert_array_equal(result, data)

    def test_wrong_data_shape(self, map_instance):
        """Test writing data with wrong shape"""
        # 2D instead of 3D
        bad_data = np.zeros((64, 64), dtype=np.uint8)

        with pytest.raises((ValueError, AttributeError)):
            map_instance.write_region(x=0, y=0, data=bad_data)

    def test_wrong_data_type(self, map_instance):
        """Test writing data with wrong dtype"""
        # Float instead of uint8
        bad_data = np.zeros((64, 64, 3), dtype=np.float32)

        # Should either convert or raise error
        try:
            map_instance.write_region(x=0, y=0, data=bad_data)
            # If it succeeds, verify it was converted
            result = map_instance.read_region(x=0, y=0, width=64, height=64)
            assert result.dtype == np.uint8
        except (ValueError, TypeError):
            # It's also valid to raise an error
            pass


class TestInfiniteMapPerformance:
    """Performance and stress tests"""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def map_instance(self, temp_dir):
        return InfiniteMap(storage_dir=temp_dir)

    def test_many_small_writes(self, map_instance):
        """Test writing many small regions"""
        import time
        start = time.time()

        # Write 100 small regions
        for i in range(100):
            data = np.zeros((16, 16, 3), dtype=np.uint8)
            data[:, :, i % 3] = (i * 2) % 256
            x = (i % 10) * 20
            y = (i // 10) * 20
            map_instance.write_region(x=x, y=y, data=data)

        elapsed = time.time() - start
        print(f"\n  ⏱️  100 writes took {elapsed:.3f}s ({elapsed/100*1000:.1f}ms each)")

        # Should complete in reasonable time (< 5s for 100 writes)
        assert elapsed < 5.0

    def test_repeated_reads(self, map_instance):
        """Test reading same region repeatedly (tests caching)"""
        import time

        # Write once
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        data[:, :, 0] = 255
        map_instance.write_region(x=0, y=0, data=data)

        # Read many times
        start = time.time()
        for _ in range(1000):
            result = map_instance.read_region(x=0, y=0, width=64, height=64)
        elapsed = time.time() - start

        print(f"\n  ⏱️  1000 reads took {elapsed:.3f}s ({elapsed/1000*1000:.2f}ms each)")

        # Should be fast (< 1s for 1000 reads)
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
