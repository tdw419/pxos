"""
Pixel compressor - converts files to pixel representations
"""

import os
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Optional

from .pixel import Pixel, PixelType, Position, PixelVisual, Metadata, DataReference
from .storage import ContentStore, MetadataStore


class PixelCompressor:
    """Compresses files into single-pixel representations"""

    def __init__(self, content_store: ContentStore, metadata_store: MetadataStore):
        self.content_store = content_store
        self.metadata_store = metadata_store
        self.backend = "local"

    def compress_file(
        self,
        file_path: str,
        position: Position,
        parent_id: Optional[str] = None
    ) -> Pixel:
        """
        Compress a file into a single pixel.

        Args:
            file_path: Path to file to compress
            position: Where to place pixel on map
            parent_id: Optional parent directory pixel ID

        Returns:
            Pixel object representing the file
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Read file content
        content = path.read_bytes()

        # Store content and get hash (automatic deduplication)
        content_hash = self.content_store.store(content, compress=True)

        # Extract file metadata
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))

        metadata = Metadata(
            name=path.name,
            path=str(path.absolute()),
            size=stat.st_size,
            permissions=oct(stat.st_mode)[-4:],
            owner=stat.st_uid,
            group=stat.st_gid,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            accessed=datetime.fromtimestamp(stat.st_atime),
            mime_type=mime_type
        )

        # Generate visual representation
        visual = PixelVisual.from_file_properties(
            mime_type=mime_type or "",
            size=stat.st_size,
            access_time=stat.st_atime
        )

        # Create data reference
        data_ref = DataReference(
            hash=f"sha256:{content_hash}",
            backend=self.backend,
            compressed=True,
            compression="zstd"
        )

        # Determine pixel type
        pixel_type = PixelType.EXECUTABLE if self._is_executable(path) else PixelType.FILE

        # Create pixel
        pixel = Pixel(
            type=pixel_type,
            position=position,
            visual=visual,
            metadata=metadata,
            data=data_ref,
            parent_id=parent_id
        )

        # Store pixel metadata
        self.metadata_store.store_pixel(pixel)

        return pixel

    def compress_directory(
        self,
        dir_path: str,
        position: Position,
        parent_id: Optional[str] = None,
        recursive: bool = True
    ) -> Pixel:
        """
        Compress a directory into a directory pixel.

        Args:
            dir_path: Path to directory
            position: Where to place directory pixel
            parent_id: Optional parent directory ID
            recursive: Whether to recursively compress children

        Returns:
            Directory pixel with children
        """
        path = Path(dir_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        # Get directory metadata
        stat = path.stat()

        metadata = Metadata(
            name=path.name or "/",
            path=str(path.absolute()),
            size=0,  # Directories don't have size
            permissions=oct(stat.st_mode)[-4:],
            owner=stat.st_uid,
            group=stat.st_gid,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            accessed=datetime.fromtimestamp(stat.st_atime),
            mime_type="inode/directory"
        )

        # Directory visual (distinct from files)
        visual = PixelVisual(r=100, g=100, b=200, a=255, size=2)

        # Dummy data reference (directories don't have content)
        data_ref = DataReference(
            hash="sha256:0" * 64,
            backend=self.backend,
            compressed=False,
            compression="none"
        )

        # Create directory pixel
        dir_pixel = Pixel(
            type=PixelType.DIRECTORY,
            position=position,
            visual=visual,
            metadata=metadata,
            data=data_ref,
            parent_id=parent_id,
            children=[]
        )

        # Store directory first to get ID
        self.metadata_store.store_pixel(dir_pixel)

        # Recursively compress children if requested
        if recursive:
            child_pixels = []
            grid_x, grid_y = 0, 0
            grid_size = 16  # 16x16 grid layout

            try:
                for idx, child in enumerate(sorted(path.iterdir())):
                    # Calculate relative position within directory
                    child_x = position.x + (grid_x * 2)
                    child_y = position.y + (grid_y * 2)
                    child_pos = Position(child_x, child_y)

                    try:
                        if child.is_file():
                            child_pixel = self.compress_file(
                                str(child),
                                child_pos,
                                parent_id=dir_pixel.id
                            )
                        elif child.is_dir():
                            child_pixel = self.compress_directory(
                                str(child),
                                child_pos,
                                parent_id=dir_pixel.id,
                                recursive=True
                            )
                        else:
                            continue  # Skip symlinks, etc.

                        child_pixels.append(child_pixel.id)

                        # Update grid position
                        grid_x += 1
                        if grid_x >= grid_size:
                            grid_x = 0
                            grid_y += 1

                    except (PermissionError, OSError) as e:
                        print(f"Warning: Skipping {child}: {e}")
                        continue

            except PermissionError as e:
                print(f"Warning: Cannot list directory {dir_path}: {e}")

            # Update directory with children
            dir_pixel.children = child_pixels
            self.metadata_store.store_pixel(dir_pixel)

        return dir_pixel

    def _is_executable(self, path: Path) -> bool:
        """Check if file is executable"""
        if not path.exists():
            return False

        # Check if executable bit is set
        stat = path.stat()
        return bool(stat.st_mode & 0o111)

    def get_compression_stats(self, pixel: Pixel) -> dict:
        """Get compression statistics for a pixel"""
        original_size = pixel.metadata.size
        hash_str = pixel.data.hash.replace("sha256:", "")
        compressed_size = self.content_store.size_on_disk(hash_str)

        ratio = compressed_size / original_size if original_size > 0 else 0

        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": ratio,
            "savings": original_size - compressed_size,
            "savings_percent": (1 - ratio) * 100 if original_size > 0 else 0
        }
