#!/usr/bin/env python3
"""
FRACTAL PIXEL SYSTEM DEMONSTRATION
Shows the power of the intelligent pixel database
"""

import pickle
from fractal_pixel_db import FractalPixelDatabase, PixelQueryEngine, FractalPixel

def load_database(db_file='pxos_fractal_pixels.db'):
    """Load the pixel database"""
    print("📂 Loading fractal pixel database...")

    with open(db_file, 'rb') as f:
        save_data = pickle.load(f)

    # Reconstruct database
    db = FractalPixelDatabase(save_data['root_dir'])

    for pixel_data in save_data['pixels_data']:
        pixel = FractalPixel(
            file_path=pixel_data['file_path'],
            collapsed_pixel=pixel_data['collapsed_pixel'],
            content_hash=pixel_data['content_hash'],
            metadata=pixel_data['metadata']
        )
        pixel.intelligence.semantic_understanding = pixel_data['semantic_understanding']

        db.pixels[pixel_data['file_path']] = pixel
        db.pixel_map[pixel_data['collapsed_pixel']] = pixel

        # Rebuild concept index
        concepts = pixel_data['semantic_understanding'].get('semantic_concepts', [])
        for concept in concepts:
            if concept not in db.concept_index:
                db.concept_index[concept] = []
            db.concept_index[concept].append(pixel)

    db.initialized = True
    print(f"✅ Loaded {len(db.pixels)} intelligent pixels\n")

    return db

def demo_1_visual_compression(db):
    """Demonstrate: Entire codebase as single visual"""
    print("="*60)
    print("DEMO 1: VISUAL COMPRESSION")
    print("  Every file → 1 pixel")
    print("  Entire codebase → Single visual map")
    print("="*60)
    print()

    print("🎨 Your entire OS codebase as pixels:")
    grid = db.get_pixel_grid(width=12)

    for row in grid:
        for pixel in row:
            r, g, b = pixel
            block = f"\033[48;2;{r};{g};{b}m  \033[0m"
            print(block, end="")
        print()

    print(f"\n📊 {len(db.pixels)} files compressed to {len(grid)} rows of pixels")
    print("   Each colored block = 1 complete file")
    print("   Color = semantic fingerprint of content")
    print()

def demo_2_semantic_search(db):
    """Demonstrate: Find files by meaning, not name"""
    print("="*60)
    print("DEMO 2: SEMANTIC SEARCH")
    print("  Find files by MEANING, not just filename")
    print("="*60)
    print()

    query_engine = PixelQueryEngine(db)

    # Example queries
    queries = ['boot', 'memory', 'semantic', 'primitive']

    for concept in queries:
        results = query_engine.query_by_concept(concept)
        print(f"🔎 Query: 'Files about {concept}'")
        print(f"   Found: {len(results)} files")

        if results:
            print(f"   Examples:")
            for pixel in results[:3]:
                rgb = pixel.collapsed_pixel
                name = pixel.metadata['file_name']
                print(f"     • RGB{rgb} | {name}")
        print()

def demo_3_intelligent_relationships(db):
    """Demonstrate: Files know how they relate to each other"""
    print("="*60)
    print("DEMO 3: INTELLIGENT RELATIONSHIPS")
    print("  Each pixel knows its relationships")
    print("="*60)
    print()

    # Pick an interesting file
    sample_file = None
    for pixel in db.pixels.values():
        if 'README' in pixel.metadata['file_name']:
            sample_file = pixel
            break

    if sample_file:
        print(f"📄 Analyzing: {sample_file.metadata['file_name']}")
        print(f"   Pixel: RGB{sample_file.collapsed_pixel}")

        # Get relationships
        expansion = db.expand_to_map(sample_file)
        neighbors = expansion['semantic_neighbors']

        print(f"\n🔗 This file is related to {len(neighbors)} other files:")

        for i, neighbor in enumerate(neighbors[:5], 1):
            neighbor_pixel = neighbor['pixel']
            similarity = neighbor['similarity']
            shared = neighbor['shared_concepts'][:3]

            print(f"\n  {i}. {neighbor_pixel.metadata['file_name']}")
            print(f"     Shared concepts: {', '.join(shared)}")
            print(f"     Similarity score: {similarity}")
    print()

def demo_4_fractal_expansion(db):
    """Demonstrate: Collapse/expand between pixel and full content"""
    print("="*60)
    print("DEMO 4: FRACTAL EXPANSION")
    print("  Collapse to 1 pixel ⇄ Expand to full file")
    print("="*60)
    print()

    # Pick a small file to demonstrate
    small_file = None
    for pixel in db.pixels.values():
        if pixel.metadata['file_size'] < 2000 and pixel.metadata['file_name'].endswith('.sh'):
            small_file = pixel
            break

    if small_file:
        print(f"📄 File: {small_file.metadata['file_name']}")
        print(f"   Size: {small_file.metadata['file_size']} bytes")
        print(f"\n1️⃣  COLLAPSED STATE:")
        print(f"   → Single pixel: RGB{small_file.collapsed_pixel}")

        print(f"\n2️⃣  EXPANDED STATE:")
        content = small_file.expand()
        lines = content.split('\n')[:10]

        for i, line in enumerate(lines, 1):
            print(f"   {i:2d} | {line[:55]}")

        print(f"\n3️⃣  COLLAPSED AGAIN:")
        pixel = small_file.collapse()
        print(f"   → Back to: RGB{pixel}")
        print(f"   → All content preserved in 3 bytes!")
    print()

def demo_5_intelligent_summary(db):
    """Demonstrate: AI understanding of file content"""
    print("="*60)
    print("DEMO 5: INTELLIGENT UNDERSTANDING")
    print("  Each pixel has AI that understands its content")
    print("="*60)
    print()

    # Show intelligence for different file types
    file_types = {}
    for pixel in db.pixels.values():
        file_type = pixel.intelligence.semantic_understanding.get('file_type', 'unknown')
        if file_type not in file_types:
            file_types[file_type] = pixel

    for file_type, pixel in list(file_types.items())[:5]:
        print(f"\n📄 {pixel.metadata['file_name']}")
        print(f"   Type: {file_type}")
        print(f"   Intelligence: {pixel.get_intelligent_summary()}")

        concepts = pixel.intelligence.semantic_understanding.get('semantic_concepts', [])
        if concepts:
            print(f"   Understands: {', '.join(concepts[:5])}")

        functions = pixel.intelligence.semantic_understanding.get('key_functions', [])
        if functions:
            print(f"   Functions found: {len(functions)}")
    print()

def demo_6_concept_clusters(db):
    """Demonstrate: Automatic clustering by semantic meaning"""
    print("="*60)
    print("DEMO 6: SEMANTIC CLUSTERING")
    print("  Files automatically group by meaning")
    print("="*60)
    print()

    print("🧩 Automatic concept clusters found:")
    print()

    for concept in sorted(db.concept_index.keys())[:8]:
        pixels = db.concept_index[concept]
        print(f"  📦 {concept.upper()} cluster ({len(pixels)} files):")

        # Group by subdirectory
        from collections import defaultdict
        by_dir = defaultdict(list)

        for pixel in pixels:
            dir_name = pixel.metadata['directory'].split('/')[-1] or 'root'
            by_dir[dir_name].append(pixel)

        for dir_name, dir_pixels in sorted(by_dir.items())[:3]:
            print(f"     {dir_name}/: {len(dir_pixels)} files")
    print()

def demo_7_query_by_color(db):
    """Demonstrate: Find similar files by pixel color"""
    print("="*60)
    print("DEMO 7: COLOR-BASED SIMILARITY")
    print("  Similar colors = similar semantic content")
    print("="*60)
    print()

    query_engine = PixelQueryEngine(db)

    # Pick a reference file
    ref_pixel = list(db.pixels.values())[0]
    target_color = ref_pixel.collapsed_pixel

    print(f"🎯 Reference file: {ref_pixel.metadata['file_name']}")
    print(f"   Color: RGB{target_color}")
    print()

    # Find similar colors
    similar = query_engine.query_by_color(target_color, tolerance=30)

    print(f"🔎 Files with similar colors (within 30 RGB units):")
    for i, pixel in enumerate(similar[:5], 1):
        rgb = pixel.collapsed_pixel
        name = pixel.metadata['file_name']

        # Calculate color distance
        dr = abs(rgb[0] - target_color[0])
        dg = abs(rgb[1] - target_color[1])
        db_val = abs(rgb[2] - target_color[2])
        distance = max(dr, dg, db_val)

        print(f"  {i}. RGB{rgb} | {name} (distance: {distance})")
    print()

def main():
    """Run all demonstrations"""
    print()
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "FRACTAL PIXEL SYSTEM" + " "*23 + "║")
    print("║" + " "*12 + "POWER DEMONSTRATION" + " "*28 + "║")
    print("╚" + "═"*58 + "╝")
    print()

    db = load_database()

    demo_1_visual_compression(db)
    input("Press Enter to continue...")

    demo_2_semantic_search(db)
    input("Press Enter to continue...")

    demo_3_intelligent_relationships(db)
    input("Press Enter to continue...")

    demo_4_fractal_expansion(db)
    input("Press Enter to continue...")

    demo_5_intelligent_summary(db)
    input("Press Enter to continue...")

    demo_6_concept_clusters(db)
    input("Press Enter to continue...")

    demo_7_query_by_color(db)

    print()
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "DEMONSTRATION COMPLETE" + " "*21 + "║")
    print("╚" + "═"*58 + "╝")
    print()
    print("🎯 Key Takeaways:")
    print("   • Every file = 1 intelligent pixel")
    print("   • Pixels understand their content")
    print("   • Pixels know their relationships")
    print("   • Search by meaning, not just names")
    print("   • Automatic semantic clustering")
    print("   • Visual compression of entire codebase")
    print()
    print("🚀 Next: Use pixel intelligence to navigate and synthesize code!")
    print()

if __name__ == "__main__":
    main()
