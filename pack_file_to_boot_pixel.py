#!/usr/bin/env python3
"""
pack_file_to_boot_pixel.py - Turn individual files into sub-boot pixels

Each file becomes:
  1. A 1×1 RGBA pixel (visual handle)
  2. A registry entry with file metadata
  3. A compressed blob in project_files.bin

Usage:
  python3 pack_file_to_boot_pixel.py add pxi_cpu.py --type py
  python3 pack_file_to_boot_pixel.py add hello_world.pxi.png --type pxi_module
  python3 pack_file_to_boot_pixel.py list
  python3 pack_file_to_boot_pixel.py show 0x41E2939A
  python3 pack_file_to_boot_pixel.py pack  # pack all into project_files.bin
"""

import json
import zlib
import hashlib
from pathlib import Path
from PIL import Image
import argparse
import struct

class FileBootPixelPacker:
    def __init__(self):
        self.registry_path = Path("file_boot_registry.json")
        self.blobs_dir = Path("file_blobs")
        self.blobs_dir.mkdir(exist_ok=True)

        # Load or create registry
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def add_file(self, file_path: str, file_type: str, name: str = None):
        """
        Pack a file into a sub-boot pixel

        Args:
            file_path: Path to file to pack
            file_type: Type of file (py, pxi_module, world, model, data, etc.)
            name: Optional friendly name (defaults to filename)
        """
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return None

        # Read and compress file
        data = path.read_bytes()
        compressed = zlib.compress(data, level=9)

        # Generate stable 32-bit ID from content hash
        hash_obj = hashlib.sha256(data)
        hash_bytes = hash_obj.digest()
        file_id = struct.unpack('>I', hash_bytes[:4])[0]  # 32-bit big-endian

        # Derive RGBA from file_id
        R = (file_id >> 24) & 0xFF
        G = (file_id >> 16) & 0xFF
        B = (file_id >> 8) & 0xFF
        A = file_id & 0xFF

        # Friendly name
        display_name = name or path.stem

        # Save compressed blob
        blob_path = self.blobs_dir / f"file_{file_id:08X}.bin"
        blob_path.write_bytes(compressed)

        # Create 1×1 sub-boot pixel
        pixel_img = Image.new("RGBA", (1, 1), (R, G, B, A))
        pixel_name = f"{path.stem}.filepx.png"
        pixel_img.save(pixel_name)

        # Add to registry
        self.registry[str(file_id)] = {
            "file_id": f"0x{file_id:08X}",
            "name": display_name,
            "type": file_type,
            "original_path": str(path),
            "original_size": len(data),
            "compressed_size": len(compressed),
            "compression_ratio": f"{(1 - len(compressed)/len(data))*100:.1f}%",
            "pixel": [R, G, B, A],
            "pixel_file": pixel_name,
            "blob_file": str(blob_path),
            "hash": hash_obj.hexdigest()
        }

        self._save_registry()

        print(f"✅ Created sub-boot pixel: {pixel_name}")
        print(f"   File ID:    0x{file_id:08X}")
        print(f"   RGBA:       ({R}, {G}, {B}, {A})")
        print(f"   Type:       {file_type}")
        print(f"   Size:       {len(data)} → {len(compressed)} bytes")
        print(f"   Ratio:      {(1 - len(compressed)/len(data))*100:.1f}% compression")

        return file_id

    def list_files(self):
        """List all registered sub-boot pixels"""
        if not self.registry:
            print("No sub-boot pixels registered yet.")
            return

        print(f"\n{'File ID':<12} {'Name':<25} {'Type':<12} {'Size':<15} {'Compression':<12}")
        print("=" * 90)

        for file_id_str, entry in sorted(self.registry.items()):
            name = entry['name'][:24]
            ftype = entry['type']
            size = f"{entry['original_size']:,}"
            ratio = entry['compression_ratio']

            print(f"{entry['file_id']:<12} {name:<25} {ftype:<12} {size:<15} {ratio:<12}")

        print(f"\nTotal: {len(self.registry)} sub-boot pixels")

    def show_file(self, file_id: int):
        """Show detailed info for a specific file"""
        entry = self.registry.get(str(file_id))
        if not entry:
            print(f"❌ File ID {file_id:08X} not found in registry")
            return

        print(f"\n{'='*60}")
        print(f"Sub-Boot Pixel: {entry['name']}")
        print(f"{'='*60}")
        print(f"File ID:          {entry['file_id']}")
        print(f"Type:             {entry['type']}")
        print(f"RGBA Pixel:       ({entry['pixel'][0]}, {entry['pixel'][1]}, {entry['pixel'][2]}, {entry['pixel'][3]})")
        print(f"Pixel File:       {entry['pixel_file']}")
        print(f"Original Path:    {entry['original_path']}")
        print(f"Original Size:    {entry['original_size']:,} bytes")
        print(f"Compressed Size:  {entry['compressed_size']:,} bytes")
        print(f"Compression:      {entry['compression_ratio']}")
        print(f"Content Hash:     {entry['hash'][:16]}...")
        print(f"Blob Location:    {entry['blob_file']}")
        print(f"{'='*60}\n")

    def pack_all(self):
        """
        Pack all registered files into a single project_files.bin

        Creates:
          - project_files.bin: Sequential compressed blobs
          - Updates registry with blob offsets
        """
        if not self.registry:
            print("No files to pack.")
            return

        print(f"Packing {len(self.registry)} files into project_files.bin...")

        output_path = Path("project_files.bin")

        with open(output_path, 'wb') as out:
            current_offset = 0

            for file_id_str, entry in self.registry.items():
                blob_path = Path(entry['blob_file'])
                compressed = blob_path.read_bytes()

                # Write to output
                out.write(compressed)

                # Update registry with offset
                entry['blob_offset'] = current_offset
                entry['blob_len'] = len(compressed)

                current_offset += len(compressed)

        self._save_registry()

        total_size = output_path.stat().st_size
        total_original = sum(e['original_size'] for e in self.registry.values())

        print(f"✅ Packed {len(self.registry)} files")
        print(f"   Total original:   {total_original:,} bytes")
        print(f"   Total packed:     {total_size:,} bytes")
        print(f"   Overall ratio:    {(1 - total_size/total_original)*100:.1f}% compression")
        print(f"   Output:           project_files.bin")

    def extract_file(self, file_id: int, output_path: str = None):
        """Extract a file from its blob"""
        entry = self.registry.get(str(file_id))
        if not entry:
            print(f"❌ File ID {file_id:08X} not found")
            return None

        # If packed, read from project_files.bin
        if 'blob_offset' in entry:
            project_files = Path("project_files.bin")
            if not project_files.exists():
                print(f"❌ project_files.bin not found. Run 'pack' first.")
                return None

            with open(project_files, 'rb') as f:
                f.seek(entry['blob_offset'])
                compressed = f.read(entry['blob_len'])
        else:
            # Read from individual blob
            blob_path = Path(entry['blob_file'])
            compressed = blob_path.read_bytes()

        # Decompress
        data = zlib.decompress(compressed)

        # Write output
        if output_path:
            out = Path(output_path)
            out.write_bytes(data)
            print(f"✅ Extracted to {output_path}")

        return data


def main():
    parser = argparse.ArgumentParser(description="Pack files into sub-boot pixels")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a file as sub-boot pixel')
    add_parser.add_argument('file', help='File to pack')
    add_parser.add_argument('--type', required=True,
                           choices=['py', 'pxi_module', 'world', 'model', 'data', 'config'],
                           help='Type of file')
    add_parser.add_argument('--name', help='Optional friendly name')

    # List command
    subparsers.add_parser('list', help='List all sub-boot pixels')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show details for a file')
    show_parser.add_argument('file_id', help='File ID (hex, e.g., 0x41E2939A)')

    # Pack command
    subparsers.add_parser('pack', help='Pack all files into project_files.bin')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract a file')
    extract_parser.add_argument('file_id', help='File ID (hex)')
    extract_parser.add_argument('--output', help='Output path')

    args = parser.parse_args()

    packer = FileBootPixelPacker()

    if args.command == 'add':
        packer.add_file(args.file, args.type, args.name)

    elif args.command == 'list':
        packer.list_files()

    elif args.command == 'show':
        file_id = int(args.file_id, 16)
        packer.show_file(file_id)

    elif args.command == 'pack':
        packer.pack_all()

    elif args.command == 'extract':
        file_id = int(args.file_id, 16)
        packer.extract_file(file_id, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
