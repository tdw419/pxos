#!/usr/bin/env python3
"""
INTERACTIVE PIXEL NAVIGATOR
Navigate your codebase as an intelligent pixel map

Commands:
  - expand <filename>     - Expand a pixel to see content
  - find <concept>        - Find all files about concept
  - related <filename>    - Find related files
  - cluster <concept>     - Show semantic cluster
  - map                   - Show pixel map
  - stats                 - Show database statistics
  - exit                  - Exit navigator
"""

import pickle
import os
from fractal_pixel_db import FractalPixelDatabase, PixelQueryEngine

class PixelNavigator:
    """Interactive navigator for the fractal pixel database"""

    def __init__(self, db_file: str = 'pxos_fractal_pixels.db'):
        print("📂 Loading fractal pixel database...")
        with open(db_file, 'rb') as f:
            save_data = pickle.load(f)

        # Reconstruct database
        from fractal_pixel_db import FractalPixel, PixelIntelligence
        self.db = FractalPixelDatabase(save_data['root_dir'])

        for pixel_data in save_data['pixels_data']:
            pixel = FractalPixel(
                file_path=pixel_data['file_path'],
                collapsed_pixel=pixel_data['collapsed_pixel'],
                content_hash=pixel_data['content_hash'],
                metadata=pixel_data['metadata']
            )
            pixel.intelligence.semantic_understanding = pixel_data['semantic_understanding']

            self.db.pixels[pixel_data['file_path']] = pixel
            self.db.pixel_map[pixel_data['collapsed_pixel']] = pixel

            # Rebuild concept index
            concepts = pixel_data['semantic_understanding'].get('semantic_concepts', [])
            for concept in concepts:
                if concept not in self.db.concept_index:
                    self.db.concept_index[concept] = []
                self.db.concept_index[concept].append(pixel)

        self.db.initialized = True
        self.query_engine = PixelQueryEngine(self.db)

        print(f"✅ Loaded {len(self.db.pixels)} intelligent pixels")
        print(f"🧠 {len(self.db.concept_index)} concepts indexed")

    def run(self):
        """Run interactive navigator"""
        print("\n" + "="*60)
        print("🎯 PIXEL NAVIGATOR - Interactive Codebase Exploration")
        print("="*60)
        print("\nType 'help' for commands, 'exit' to quit\n")

        while True:
            try:
                cmd = input("pixel> ").strip()

                if not cmd:
                    continue

                parts = cmd.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == 'exit' or command == 'quit':
                    print("👋 Goodbye!")
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'map':
                    self.show_map()
                elif command == 'stats':
                    self.show_stats()
                elif command == 'expand':
                    self.expand_pixel(args)
                elif command == 'find':
                    self.find_concept(args)
                elif command == 'related':
                    self.find_related(args)
                elif command == 'cluster':
                    self.show_cluster(args)
                elif command == 'concepts':
                    self.list_concepts()
                elif command == 'files':
                    self.list_files()
                else:
                    print(f"❌ Unknown command: {command}")
                    print("   Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_help(self):
        """Show available commands"""
        print("\n📖 AVAILABLE COMMANDS:")
        print("  map                 - Show pixel visualization")
        print("  stats               - Show database statistics")
        print("  concepts            - List all concepts")
        print("  files               - List all files")
        print("  find <concept>      - Find files about concept")
        print("  expand <file>       - Expand pixel to see content")
        print("  related <file>      - Find related files")
        print("  cluster <concept>   - Show semantic cluster")
        print("  exit                - Exit navigator")
        print()

    def show_map(self):
        """Show pixel map"""
        print("\n🎨 PIXEL MAP:")
        grid = self.db.get_pixel_grid(width=12)

        for row in grid:
            for pixel in row:
                r, g, b = pixel
                block = f"\033[48;2;{r};{g};{b}m  \033[0m"
                print(block, end="")
            print()
        print()

    def show_stats(self):
        """Show database statistics"""
        print("\n📊 DATABASE STATISTICS:")
        print(f"  Total pixels: {len(self.db.pixels)}")
        print(f"  Concepts indexed: {len(self.db.concept_index)}")

        print("\n📈 CONCEPT DISTRIBUTION:")
        for concept in sorted(self.db.concept_index.keys()):
            count = len(self.db.concept_index[concept])
            bar = '█' * min(count, 40)
            print(f"  {concept:15s} | {count:2d} | {bar}")
        print()

    def list_concepts(self):
        """List all concepts"""
        print("\n🧠 ALL CONCEPTS:")
        concepts = sorted(self.db.concept_index.keys())
        for i, concept in enumerate(concepts):
            count = len(self.db.concept_index[concept])
            print(f"  {i+1:2d}. {concept:15s} ({count} files)")
        print()

    def list_files(self):
        """List all files"""
        print("\n📁 ALL FILES:")
        files = sorted(self.db.pixels.values(), key=lambda p: p.metadata['file_name'])
        for i, pixel in enumerate(files):
            rgb = pixel.collapsed_pixel
            summary = pixel.get_intelligent_summary()
            print(f"  {i+1:2d}. RGB{rgb} | {pixel.metadata['file_name']:40s} | {summary}")
        print()

    def find_concept(self, concept: str):
        """Find files about a concept"""
        if not concept:
            print("❌ Usage: find <concept>")
            return

        results = self.query_engine.query_by_concept(concept)

        if not results:
            print(f"❌ No files found for concept: {concept}")
            return

        print(f"\n🔎 Files about '{concept}' ({len(results)} found):")
        for i, pixel in enumerate(results):
            rgb = pixel.collapsed_pixel
            summary = pixel.get_intelligent_summary()
            print(f"  {i+1:2d}. RGB{rgb} | {pixel.metadata['file_name']:40s}")
            print(f"      {summary}")
        print()

    def expand_pixel(self, filename: str):
        """Expand a pixel to show content"""
        if not filename:
            print("❌ Usage: expand <filename>")
            return

        # Find pixel by filename
        pixel = None
        for p in self.db.pixels.values():
            if p.metadata['file_name'] == filename or p.file_path.endswith(filename):
                pixel = p
                break

        if not pixel:
            print(f"❌ File not found: {filename}")
            return

        print(f"\n📄 {pixel.metadata['file_name']}")
        print(f"   Path: {pixel.file_path}")
        print(f"   Pixel: RGB{pixel.collapsed_pixel}")
        print(f"   Size: {pixel.metadata['file_size']} bytes")
        print(f"   Type: {pixel.intelligence.semantic_understanding.get('file_type', 'unknown')}")

        concepts = pixel.intelligence.semantic_understanding.get('semantic_concepts', [])
        if concepts:
            print(f"   Concepts: {', '.join(concepts)}")

        # Show content preview
        content = pixel.expand()
        lines = content.split('\n')

        print(f"\n📝 CONTENT PREVIEW (first 20 lines):")
        print("   " + "-"*57)
        for i, line in enumerate(lines[:20], 1):
            print(f"   {i:3d} | {line[:50]}")

        if len(lines) > 20:
            print(f"   ... ({len(lines) - 20} more lines)")

        print("   " + "-"*57)
        print()

    def find_related(self, filename: str):
        """Find files related to a file"""
        if not filename:
            print("❌ Usage: related <filename>")
            return

        # Find pixel by filename
        pixel = None
        for p in self.db.pixels.values():
            if p.metadata['file_name'] == filename or p.file_path.endswith(filename):
                pixel = p
                break

        if not pixel:
            print(f"❌ File not found: {filename}")
            return

        # Get expansion map
        expansion = self.db.expand_to_map(pixel)
        neighbors = expansion['semantic_neighbors']

        print(f"\n🔗 Files related to '{pixel.metadata['file_name']}':")
        print(f"   Found {len(neighbors)} related files")
        print()

        for i, neighbor in enumerate(neighbors[:10], 1):
            neighbor_pixel = neighbor['pixel']
            similarity = neighbor['similarity']
            shared = neighbor['shared_concepts']

            print(f"  {i:2d}. {neighbor_pixel.metadata['file_name']:40s}")
            print(f"      Similarity: {similarity} shared concepts")
            print(f"      Via: {', '.join(shared)}")

        if len(neighbors) > 10:
            print(f"\n  ... and {len(neighbors) - 10} more")
        print()

    def show_cluster(self, concept: str):
        """Show semantic cluster for a concept"""
        if not concept:
            print("❌ Usage: cluster <concept>")
            return

        results = self.query_engine.query_by_concept(concept)

        if not results:
            print(f"❌ No files found for concept: {concept}")
            return

        print(f"\n🧩 SEMANTIC CLUSTER: '{concept}'")
        print(f"   {len(results)} files in cluster")
        print()

        # Group by file type
        from collections import defaultdict
        by_type = defaultdict(list)

        for pixel in results:
            file_type = pixel.intelligence.semantic_understanding.get('file_type', 'unknown')
            by_type[file_type].append(pixel)

        for file_type, pixels in sorted(by_type.items()):
            print(f"  📦 {file_type} ({len(pixels)} files):")
            for pixel in pixels[:5]:
                print(f"     • {pixel.metadata['file_name']}")
            if len(pixels) > 5:
                print(f"     ... and {len(pixels) - 5} more")
            print()

def main():
    """Main entry point"""
    nav = PixelNavigator('pxos_fractal_pixels.db')
    nav.run()

if __name__ == "__main__":
    main()
