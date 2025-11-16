#!/usr/bin/env python3
"""
pxfs CLI - Command-line interface for pixel filesystem operations
"""

import argparse
import sys
from pathlib import Path

from pxfs import (
    ContentStore,
    MetadataStore,
    PixelCompressor,
    PixelDecompressor,
    PixelMapVisualizer,
    Pixel,
    PixelType,
    Position,
    PixelVisual,
    Metadata,
    DataReference,
)
from datetime import datetime


class PXFSCLI:
    def __init__(self, storage_path: str = ".pxfs"):
        self.storage_path = Path(storage_path)
        self.content_store = ContentStore(str(self.storage_path / "content"))
        self.metadata_store = MetadataStore(str(self.storage_path / "metadata.db"))
        self.compressor = PixelCompressor(self.content_store, self.metadata_store)
        self.decompressor = PixelDecompressor(self.content_store, self.metadata_store)
        self.visualizer = PixelMapVisualizer(self.metadata_store)

    def compress_file(self, file_path: str, x: int = 0, y: int = 0):
        """Compress a single file to pixel"""
        print(f"Compressing file: {file_path}")
        position = Position(x, y)
        pixel = self.compressor.compress_file(file_path, position)

        print(f"✓ File compressed to pixel")
        print(f"  Pixel ID: {pixel.id}")
        print(f"  Position: ({pixel.position.x}, {pixel.position.y})")
        print(f"  Visual: RGB({pixel.visual.r}, {pixel.visual.g}, {pixel.visual.b})")
        print(f"  Hash: {pixel.data.hash}")

        stats = self.compressor.get_compression_stats(pixel)
        print(f"\nCompression stats:")
        print(f"  Original size: {stats['original_size']:,} bytes")
        print(f"  Compressed size: {stats['compressed_size']:,} bytes")
        print(f"  Ratio: {stats['ratio']:.2%}")
        print(f"  Savings: {stats['savings']:,} bytes ({stats['savings_percent']:.1f}%)")

        return pixel

    def compress_directory(self, dir_path: str, x: int = 0, y: int = 0, recursive: bool = True):
        """Compress entire directory tree to pixels"""
        print(f"Compressing directory: {dir_path}")
        print(f"Recursive: {recursive}")
        print()

        position = Position(x, y)
        pixel = self.compressor.compress_directory(dir_path, position, recursive=recursive)

        print(f"\n✓ Directory compressed to pixel map")
        print(f"  Root pixel ID: {pixel.id}")
        print(f"  Position: ({pixel.position.x}, {pixel.position.y})")
        print(f"  Children: {len(pixel.children)}")

        # Get stats
        stats = self.metadata_store.get_stats()
        print(f"\nStorage stats:")
        print(f"  Total pixels: {stats['total_pixels']:,}")
        print(f"  Unique files: {stats['unique_files']:,}")
        print(f"  Total size: {stats['total_size']:,} bytes")
        print(f"  Directories: {stats['directories']:,}")
        print(f"  Files: {stats['files']:,}")

        dedup_ratio = stats['total_pixels'] / stats['unique_files'] if stats['unique_files'] > 0 else 1
        print(f"  Deduplication ratio: {dedup_ratio:.2f}x")

        return pixel

    def decompress_file(self, pixel_path: str, output_path: str = None):
        """Decompress pixel back to file"""
        print(f"Decompressing: {pixel_path}")

        output = self.decompressor.decompress_by_path(pixel_path, output_path)

        print(f"✓ File decompressed to: {output}")
        print(f"  Size: {output.stat().st_size:,} bytes")

        return output

    def decompress_directory(self, pixel_path: str, output_path: str = None):
        """Decompress directory pixel back to filesystem"""
        print(f"Decompressing directory: {pixel_path}")

        output = self.decompressor.decompress_by_path(pixel_path, output_path)

        print(f"✓ Directory decompressed to: {output}")

        # Count files
        file_count = sum(1 for _ in output.rglob("*") if _.is_file())
        dir_count = sum(1 for _ in output.rglob("*") if _.is_dir())

        print(f"  Files: {file_count}")
        print(f"  Directories: {dir_count}")

        return output

    def list_pixels(self, chunk_x: int = 0, chunk_y: int = 0):
        """List pixels in a chunk"""
        print(f"Pixels in chunk ({chunk_x}, {chunk_y}):")

        pixels = self.metadata_store.get_pixels_in_chunk(chunk_x, chunk_y)

        if not pixels:
            print("  (empty)")
            return

        for px in pixels:
            print(f"\n  {px['metadata']['name']}")
            print(f"    Type: {px['type']}")
            print(f"    Position: ({px['position']['x']}, {px['position']['y']})")
            print(f"    Size: {px['metadata']['size']:,} bytes")
            print(f"    Visual: RGB({px['visual']['r']}, {px['visual']['g']}, {px['visual']['b']})")

    def show_stats(self):
        """Show storage statistics"""
        stats = self.metadata_store.get_stats()

        print("pxFS Storage Statistics")
        print("=" * 40)
        print(f"Total pixels: {stats['total_pixels']:,}")
        print(f"Unique files: {stats['unique_files']:,}")
        print(f"Total size: {stats['total_size']:,} bytes ({stats['total_size'] / 1024 / 1024:.2f} MB)")
        print(f"Directories: {stats['directories']:,}")
        print(f"Files: {stats['files']:,}")

        if stats['unique_files'] > 0:
            dedup = stats['total_pixels'] / stats['unique_files']
            print(f"\nDeduplication ratio: {dedup:.2f}x")
            print(f"Storage savings: {(1 - 1/dedup) * 100:.1f}%")

    def verify_integrity(self):
        """Verify integrity of all pixels"""
        print("Verifying pixel integrity...")

        stats = self.metadata_store.get_stats()
        total = stats['files']

        if total == 0:
            print("No files to verify")
            return

        # Get all file pixels
        all_pixels = []
        for chunk_data in self.metadata_store.conn.execute("SELECT DISTINCT chunk_x, chunk_y FROM pixels"):
            chunk_pixels = self.metadata_store.get_pixels_in_chunk(chunk_data[0], chunk_data[1])
            all_pixels.extend(chunk_pixels)

        verified = 0
        failed = 0

        for px_data in all_pixels:
            if px_data['type'] == 'directory':
                continue

            pixel = Pixel(
                type=PixelType(px_data["type"]),
                id=px_data["id"],
                position=Position(**px_data["position"]),
                visual=PixelVisual(**px_data["visual"]),
                metadata=Metadata(
                    **{k: v if k not in ["created", "modified", "accessed"]
                       else datetime.fromisoformat(v)
                       for k, v in px_data["metadata"].items()}
                ),
                data=DataReference(**px_data["data"]),
                children=[],
                parent_id=px_data["parent_id"]
            )

            if self.decompressor.verify_pixel(pixel):
                verified += 1
            else:
                failed += 1
                print(f"✗ Failed: {pixel.metadata.path}")

        print(f"\nVerification complete:")
        print(f"  Verified: {verified}/{total}")
        print(f"  Failed: {failed}/{total}")


def main():
    parser = argparse.ArgumentParser(
        description="pxFS - Pixel Filesystem CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress a file
  pxfs_cli.py compress-file /path/to/file.txt

  # Compress entire directory
  pxfs_cli.py compress-dir /path/to/directory

  # Decompress a file
  pxfs_cli.py decompress-file /path/to/file.txt -o /output/path

  # Show statistics
  pxfs_cli.py stats

  # Verify integrity
  pxfs_cli.py verify
        """
    )

    parser.add_argument(
        "--storage",
        default=".pxfs",
        help="Path to pxFS storage directory (default: .pxfs)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Compress file
    compress_file = subparsers.add_parser("compress-file", help="Compress a file to pixel")
    compress_file.add_argument("path", help="Path to file")
    compress_file.add_argument("-x", type=int, default=0, help="X coordinate")
    compress_file.add_argument("-y", type=int, default=0, help="Y coordinate")

    # Compress directory
    compress_dir = subparsers.add_parser("compress-dir", help="Compress directory to pixels")
    compress_dir.add_argument("path", help="Path to directory")
    compress_dir.add_argument("-x", type=int, default=0, help="X coordinate")
    compress_dir.add_argument("-y", type=int, default=0, help="Y coordinate")
    compress_dir.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirectories")

    # Decompress file
    decompress_file = subparsers.add_parser("decompress-file", help="Decompress pixel to file")
    decompress_file.add_argument("path", help="Original file path")
    decompress_file.add_argument("-o", "--output", help="Output path")

    # Decompress directory
    decompress_dir = subparsers.add_parser("decompress-dir", help="Decompress directory pixels")
    decompress_dir.add_argument("path", help="Original directory path")
    decompress_dir.add_argument("-o", "--output", help="Output path")

    # List pixels
    list_cmd = subparsers.add_parser("list", help="List pixels in chunk")
    list_cmd.add_argument("-x", type=int, default=0, help="Chunk X coordinate")
    list_cmd.add_argument("-y", type=int, default=0, help="Chunk Y coordinate")

    # Stats
    subparsers.add_parser("stats", help="Show storage statistics")

    # Verify
    subparsers.add_parser("verify", help="Verify integrity of all pixels")

    # Visualize (HTML)
    visualize_html = subparsers.add_parser("visualize", help="Generate HTML visualization of pixel map")
    visualize_html.add_argument("-o", "--output", default="pixel_map.html", help="Output HTML file")
    visualize_html.add_argument("-x", type=int, default=0, help="Chunk X coordinate")
    visualize_html.add_argument("-y", type=int, default=0, help="Chunk Y coordinate")
    visualize_html.add_argument("-s", "--size", type=int, default=10, help="Pixel size in pixels")

    # Visualize (ASCII)
    visualize_ascii = subparsers.add_parser("visualize-ascii", help="Show ASCII visualization of pixel map")
    visualize_ascii.add_argument("-x", type=int, default=0, help="Chunk X coordinate")
    visualize_ascii.add_argument("-y", type=int, default=0, help="Chunk Y coordinate")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize CLI
    cli = PXFSCLI(args.storage)

    try:
        if args.command == "compress-file":
            cli.compress_file(args.path, args.x, args.y)

        elif args.command == "compress-dir":
            cli.compress_directory(args.path, args.x, args.y, recursive=not args.no_recursive)

        elif args.command == "decompress-file":
            cli.decompress_file(args.path, args.output)

        elif args.command == "decompress-dir":
            cli.decompress_directory(args.path, args.output)

        elif args.command == "list":
            cli.list_pixels(args.x, args.y)

        elif args.command == "stats":
            cli.show_stats()

        elif args.command == "verify":
            cli.verify_integrity()

        elif args.command == "visualize":
            cli.visualizer.render_html(args.output, args.x, args.y, args.size)

        elif args.command == "visualize-ascii":
            ascii_map = cli.visualizer.render_ascii(args.x, args.y)
            print(ascii_map)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        cli.metadata_store.close()


if __name__ == "__main__":
    sys.exit(main())
