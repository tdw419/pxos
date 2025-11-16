"""
Pixel decompressor - reconstructs files from pixel representations
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from .pixel import Pixel, PixelType, Position, PixelVisual, Metadata, DataReference
from .storage import ContentStore, MetadataStore


class PixelDecompressor:
    """Decompresses pixels back into files"""

    def __init__(self, content_store: ContentStore, metadata_store: MetadataStore):
        self.content_store = content_store
        self.metadata_store = metadata_store

    def decompress_pixel(
        self,
        pixel: Pixel,
        output_path: Optional[str] = None,
        verify_hash: bool = True
    ) -> Path:
        """
        Decompress a pixel back into a file.

        Args:
            pixel: Pixel to decompress
            output_path: Where to write file (defaults to original path)
            verify_hash: Whether to verify content hash after decompression

        Returns:
            Path to decompressed file
        """
        if pixel.type == PixelType.DIRECTORY:
            raise ValueError("Use decompress_directory() for directory pixels")

        # Determine output path
        if output_path is None:
            output_path = pixel.metadata.path

        out_path = Path(output_path)

        # Retrieve content from store
        hash_str = pixel.data.hash.replace("sha256:", "")
        content = self.content_store.retrieve(hash_str, compressed=pixel.data.compressed)

        if content is None:
            raise ValueError(f"Content not found for hash: {pixel.data.hash}")

        # Verify hash if requested
        if verify_hash:
            import hashlib
            computed_hash = hashlib.sha256(content).hexdigest()
            if computed_hash != hash_str:
                raise ValueError(
                    f"Hash mismatch! Expected {hash_str}, got {computed_hash}"
                )

        # Create parent directories
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        out_path.write_bytes(content)

        # Restore metadata
        self._restore_metadata(out_path, pixel)

        return out_path

    def decompress_directory(
        self,
        pixel: Pixel,
        output_path: Optional[str] = None,
        recursive: bool = True
    ) -> Path:
        """
        Decompress a directory pixel.

        Args:
            pixel: Directory pixel
            output_path: Where to create directory
            recursive: Whether to recursively decompress children

        Returns:
            Path to decompressed directory
        """
        if pixel.type != PixelType.DIRECTORY:
            raise ValueError("Not a directory pixel")

        # Determine output path
        if output_path is None:
            output_path = pixel.metadata.path

        out_path = Path(output_path)

        # Create directory
        out_path.mkdir(parents=True, exist_ok=True)

        # Restore directory metadata
        self._restore_metadata(out_path, pixel)

        # Recursively decompress children if requested
        if recursive and pixel.children:
            for child_id in pixel.children:
                child_data = self.metadata_store.get_pixel_by_id(child_id)
                if not child_data:
                    print(f"Warning: Child pixel {child_id} not found")
                    continue

                # Reconstruct child pixel object
                child_pixel = Pixel(
                    type=PixelType(child_data["type"]),
                    id=child_data["id"],
                    position=Position(**child_data["position"]),
                    visual=PixelVisual(**child_data["visual"]),
                    metadata=Metadata(
                        **{k: v if k not in ["created", "modified", "accessed"]
                           else datetime.fromisoformat(v)
                           for k, v in child_data["metadata"].items()}
                    ),
                    data=DataReference(**child_data["data"]),
                    children=self.metadata_store.get_children(child_id),
                    parent_id=child_data["parent_id"]
                )

                # Decompress child
                child_name = child_pixel.metadata.name
                child_output = out_path / child_name

                try:
                    if child_pixel.type == PixelType.DIRECTORY:
                        self.decompress_directory(child_pixel, str(child_output), recursive=True)
                    else:
                        self.decompress_pixel(child_pixel, str(child_output))
                except Exception as e:
                    print(f"Warning: Failed to decompress {child_name}: {e}")

        return out_path

    def decompress_by_path(
        self,
        original_path: str,
        output_path: Optional[str] = None
    ) -> Path:
        """
        Decompress a pixel by its original file path.

        Args:
            original_path: Original path of the file
            output_path: Where to write output (defaults to original path)

        Returns:
            Path to decompressed file
        """
        pixel_data = self.metadata_store.get_pixel_by_path(original_path)
        if not pixel_data:
            raise ValueError(f"No pixel found for path: {original_path}")

        # Reconstruct pixel object
        pixel = Pixel(
            type=PixelType(pixel_data["type"]),
            id=pixel_data["id"],
            position=Position(**pixel_data["position"]),
            visual=PixelVisual(**pixel_data["visual"]),
            metadata=Metadata(
                **{k: v if k not in ["created", "modified", "accessed"]
                   else datetime.fromisoformat(v)
                   for k, v in pixel_data["metadata"].items()}
            ),
            data=DataReference(**pixel_data["data"]),
            children=self.metadata_store.get_children(pixel_data["id"]),
            parent_id=pixel_data["parent_id"]
        )

        if pixel.type == PixelType.DIRECTORY:
            return self.decompress_directory(pixel, output_path)
        else:
            return self.decompress_pixel(pixel, output_path)

    def _restore_metadata(self, path: Path, pixel: Pixel):
        """Restore file/directory metadata (permissions, timestamps)"""
        try:
            # Restore permissions
            mode = int(pixel.metadata.permissions, 8)
            os.chmod(path, mode)

            # Restore timestamps (access and modification)
            os.utime(
                path,
                (
                    pixel.metadata.accessed.timestamp(),
                    pixel.metadata.modified.timestamp()
                )
            )

            # Note: We can't restore creation time on Linux
            # We also can't easily restore owner/group without root

        except Exception as e:
            print(f"Warning: Could not restore all metadata for {path}: {e}")

    def verify_pixel(self, pixel: Pixel) -> bool:
        """
        Verify that a pixel's content can be retrieved and matches its hash.

        Args:
            pixel: Pixel to verify

        Returns:
            True if valid, False otherwise
        """
        if pixel.type == PixelType.DIRECTORY:
            return True  # Directories don't have content to verify

        try:
            hash_str = pixel.data.hash.replace("sha256:", "")
            content = self.content_store.retrieve(hash_str, compressed=pixel.data.compressed)

            if content is None:
                return False

            # Verify hash
            import hashlib
            computed_hash = hashlib.sha256(content).hexdigest()
            return computed_hash == hash_str

        except Exception:
            return False
