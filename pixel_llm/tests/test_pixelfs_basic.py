#!/usr/bin/env python3
"""
Basic unit tests for PixelFS

Tests the core functionality of pixel-based file storage.
This is a demonstration of the test quality we're aiming for in Phase 0.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import hashlib

# Import PixelFS (adjust path as needed)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from pixelfs import PixelFS, PixelFileHeader


class TestPixelFileHeader:
    """Test PixelFileHeader serialization/deserialization"""

    def test_header_pack_unpack(self):
        """Test round-trip serialization"""
        original = PixelFileHeader(
            original_size=1000,
            width=32,
            height=32,
            compression=0,
            checksum=b'a' * 32
        )

        packed = original.pack()
        assert len(packed) == PixelFileHeader.HEADER_SIZE, "Header size must be 64 bytes"

        unpacked = PixelFileHeader.unpack(packed)
        assert unpacked.original_size == original.original_size
        assert unpacked.width == original.width
        assert unpacked.height == original.height
        assert unpacked.compression == original.compression

    def test_header_magic_validation(self):
        """Test that invalid magic bytes are rejected"""
        bad_header = b'XXXX' + b'\x00' * 60

        with pytest.raises(ValueError, match="Invalid magic"):
            PixelFileHeader.unpack(bad_header)

    def test_header_size_validation(self):
        """Test that short headers are rejected"""
        short_header = b'PXIF' + b'\x00' * 50  # Only 54 bytes

        with pytest.raises(ValueError, match="Header too short"):
            PixelFileHeader.unpack(short_header)


class TestPixelFS:
    """Test PixelFS core operations"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def pixelfs(self, temp_storage):
        """Create PixelFS instance"""
        return PixelFS(root=temp_storage)

    def test_write_read_roundtrip(self, pixelfs):
        """Test basic write/read cycle"""
        test_data = b"Hello, Pixel World!" * 100
        filename = "test_roundtrip"

        # Write
        filepath = pixelfs.write(filename, test_data)
        assert filepath.exists(), "File should be created"

        # Read
        read_data = pixelfs.read(filename)
        assert read_data == test_data, "Read data should match written data"

    def test_checksum_verification(self, pixelfs):
        """Test that checksums are validated"""
        test_data = b"Test data for checksum"
        filename = "test_checksum"

        # Write
        pixelfs.write(filename, test_data)

        # Verify
        is_valid = pixelfs.verify(filename)
        assert is_valid, "Checksum should be valid"

    def test_empty_file(self, pixelfs):
        """Test handling of empty files"""
        empty_data = b""
        filename = "test_empty"

        # Should handle empty data gracefully
        filepath = pixelfs.write(filename, empty_data)
        read_data = pixelfs.read(filename)

        assert read_data == empty_data, "Should handle empty files"

    def test_large_data(self, pixelfs):
        """Test handling of larger data (>10KB)"""
        # Generate 50KB of data
        large_data = b"x" * 50000
        filename = "test_large"

        filepath = pixelfs.write(filename, large_data)
        read_data = pixelfs.read(filename)

        assert len(read_data) == len(large_data)
        assert read_data == large_data

    def test_get_info(self, pixelfs):
        """Test metadata retrieval"""
        test_data = b"Metadata test"
        filename = "test_info"

        pixelfs.write(filename, test_data)
        info = pixelfs.get_info(filename)

        assert info.original_size == len(test_data)
        assert info.width > 0
        assert info.height > 0

    def test_list_files(self, pixelfs):
        """Test file listing"""
        # Write multiple files
        pixelfs.write("file1", b"data1")
        pixelfs.write("file2", b"data2")
        pixelfs.write("file3", b"data3")

        files = pixelfs.list_files()
        assert len(files) >= 3, "Should list all written files"

    def test_file_not_found(self, pixelfs):
        """Test error handling for missing files"""
        with pytest.raises(FileNotFoundError):
            pixelfs.read("nonexistent_file")


# Example of a more complex test
class TestPixelFSEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def pixelfs(self):
        temp_dir = tempfile.mkdtemp()
        fs = PixelFS(root=temp_dir)
        yield fs
        shutil.rmtree(temp_dir)

    def test_special_characters_in_data(self, pixelfs):
        """Test handling of binary data with all byte values"""
        # Create data with all possible byte values
        test_data = bytes(range(256)) * 10
        filename = "test_binary"

        pixelfs.write(filename, test_data)
        read_data = pixelfs.read(filename)

        assert read_data == test_data

    def test_repeated_writes(self, pixelfs):
        """Test overwriting files"""
        filename = "test_overwrite"

        # Write first time
        pixelfs.write(filename, b"version1")

        # Overwrite
        pixelfs.write(filename, b"version2 is longer")

        # Read should get latest version
        data = pixelfs.read(filename)
        assert data == b"version2 is longer"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
