#!/usr/bin/env python3
"""
DIRECTORY-PRESERVING FRACTAL PIXEL SYSTEM
Maintains original file structure while adding pixel intelligence

Key innovation: Preserves your familiar directory structure while adding
semantic pixel intelligence as metadata alongside each file.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import Counter

@dataclass
class DirectoryNode:
    """Represents a directory in the preserved structure"""
    path: str
    name: str
    child_dirs: List['DirectoryNode']
    child_files: List[str]
    pixel_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.pixel_metadata is None:
            self.pixel_metadata = {
                'directory_color': None,
                'semantic_summary': '',
                'file_count': len(self.child_files),
                'concept_clusters': []
            }

class StructurePreservingPixelDB:
    """Maintains original directory structure while adding pixel intelligence"""

    def __init__(self, source_root: str):
        self.source_root = Path(source_root).resolve()
        self.structure_map: Dict[str, DirectoryNode] = {}
        self.file_pixels: Dict[str, Dict] = {}  # file_path -> pixel_data
        self.root_node = None

    def build_preserved_structure(self):
        """Build the directory tree structure with pixel metadata"""
        print("🌳 Building directory structure with pixel intelligence...")
        print("-" * 60)

        def build_node(current_path: Path) -> DirectoryNode:
            """Recursively build directory tree"""
            node = DirectoryNode(
                path=str(current_path),
                name=current_path.name if current_path != self.source_root else current_path.name,
                child_dirs=[],
                child_files=[]
            )

            # Skip certain directories
            skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv'}

            try:
                for item in current_path.iterdir():
                    if item.is_dir() and item.name not in skip_dirs and not item.name.startswith('.'):
                        # Recursively process subdirectory
                        child_node = build_node(item)
                        node.child_dirs.append(child_node)
                    elif item.is_file() and not item.name.startswith('.') and not item.name.endswith('.pyc'):
                        # Add file relative to source root
                        try:
                            rel_path = str(item.relative_to(self.source_root))
                            node.child_files.append(rel_path)

                            # Initialize pixel data for this file
                            self.file_pixels[rel_path] = self._create_pixel_for_file(item, rel_path)
                        except Exception as e:
                            print(f"   ⚠️  Skipping {item.name}: {e}")
            except PermissionError:
                print(f"   ⚠️  Permission denied: {current_path}")

            self.structure_map[str(current_path)] = node
            return node

        self.root_node = build_node(self.source_root)
        print(f"✅ Structure built: {len(self.structure_map)} directories, {len(self.file_pixels)} files")

    def _create_pixel_for_file(self, file_path: Path, rel_path: str) -> Dict:
        """Create pixel metadata for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Generate unique pixel RGB
            pixel_rgb = self._content_to_pixel(content, rel_path)

            # Extract semantic concepts
            concepts = self._extract_semantic_concepts(content, file_path.name)

            return {
                'original_path': rel_path,
                'pixel_rgb': pixel_rgb,
                'semantic_concepts': concepts,
                'file_type': file_path.suffix.lower(),
                'size_bytes': file_path.stat().st_size,
                'line_count': len(content.splitlines()),
                'content_preview': content[:150] + "..." if len(content) > 150 else content
            }
        except Exception as e:
            # For binary or unreadable files
            return {
                'original_path': rel_path,
                'pixel_rgb': (128, 128, 128),
                'semantic_concepts': ['binary_or_unreadable'],
                'file_type': file_path.suffix.lower(),
                'size_bytes': file_path.stat().st_size,
                'error': str(e)
            }

    def _content_to_pixel(self, content: str, file_path: str) -> tuple:
        """Convert file content to unique RGB pixel"""
        # Use content-based hashing for deterministic pixels
        combined = file_path + content[:1000]
        content_hash = hashlib.md5(combined.encode()).hexdigest()

        r = int(content_hash[0:2], 16)
        g = int(content_hash[2:4], 16)
        b = int(content_hash[4:6], 16)

        return (r, g, b)

    def _extract_semantic_concepts(self, content: str, filename: str) -> List[str]:
        """Extract semantic concepts from file content"""
        concepts = []
        content_lower = content.lower()

        # File type concepts
        if any(ext in filename for ext in ['.py', '.python']):
            concepts.append('python_code')
        elif any(ext in filename for ext in ['.c', '.h', '.cpp']):
            concepts.append('c_code')
        elif any(ext in filename for ext in ['.md', '.txt', '.rst']):
            concepts.append('documentation')
        elif any(ext in filename for ext in ['.asm', '.s']):
            concepts.append('assembly_code')
        elif any(ext in filename for ext in ['.sh', '.bash']):
            concepts.append('shell_script')
        elif any(ext in filename for ext in ['.json', '.yaml', '.yml']):
            concepts.append('configuration')

        # Semantic concept indicators
        concept_indicators = {
            'memory_management': ['malloc', 'free', 'page', 'virtual', 'mmu', 'alloc', 'heap', 'stack'],
            'boot_system': ['boot', 'grub', 'bios', 'uefi', 'multiboot', 'bootloader', 'sector'],
            'scheduling': ['schedule', 'sched', 'process', 'thread', 'priority', 'scheduler', 'yield'],
            'filesystem': ['file', 'inode', 'block', 'directory', 'vfs', 'ext', 'fat', 'filesystem'],
            'drivers': ['driver', 'device', 'interrupt', 'pci', 'usb', 'hardware'],
            'networking': ['tcp', 'ip', 'socket', 'packet', 'protocol', 'ethernet', 'network'],
            'kernel': ['kernel', 'syscall', 'interrupt', 'handler', 'kernel_space'],
            'concurrency': ['lock', 'mutex', 'semaphore', 'atomic', 'critical_section'],
            'architecture': ['x86', 'arm', 'riscv', 'assembly', 'instructions', 'cpu', 'register'],
            'primitives': ['WRITE', 'DEFINE', 'CALL', 'primitive', 'semantic'],
            'pixel_system': ['pixel', 'fractal', 'semantic', 'synthesis', 'intent']
        }

        for concept, indicators in concept_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                concepts.append(concept)

        return concepts

    def save_pixel_metadata(self):
        """Save pixel metadata alongside original files"""
        print("\n💾 Saving pixel metadata...")

        metadata_dir = self.source_root / ".pixel_metadata"
        metadata_dir.mkdir(exist_ok=True)

        # Save individual file pixel data
        for rel_path, pixel_data in self.file_pixels.items():
            # Create metadata filename
            file_path = Path(rel_path)
            metadata_filename = f"{file_path.name}.pixel.json"

            # Preserve directory structure in metadata
            metadata_file_dir = metadata_dir / file_path.parent
            metadata_file_dir.mkdir(parents=True, exist_ok=True)

            metadata_file = metadata_file_dir / metadata_filename

            with open(metadata_file, 'w') as f:
                json.dump(pixel_data, f, indent=2)

        print(f"   ✅ Saved {len(self.file_pixels)} pixel metadata files to .pixel_metadata/")

    def save_structure_overview(self):
        """Save overview files showing structure and indices"""
        print("   📊 Creating structure overview...")

        metadata_dir = self.source_root / ".pixel_metadata"

        # 1. Directory structure JSON
        structure_file = metadata_dir / "directory_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(self._serialize_structure(), f, indent=2)

        # 2. Pixel map (file -> pixel mapping)
        pixel_map_file = metadata_dir / "pixel_map.json"
        pixel_map = {
            file_path: data['pixel_rgb']
            for file_path, data in self.file_pixels.items()
        }
        with open(pixel_map_file, 'w') as f:
            json.dump(pixel_map, f, indent=2)

        # 3. Semantic index (concept -> files)
        semantic_index = self._build_semantic_index()
        semantic_file = metadata_dir / "semantic_index.json"
        with open(semantic_file, 'w') as f:
            json.dump(semantic_index, f, indent=2)

        # 4. Directory summaries
        directory_summaries = self._build_directory_summaries()
        summaries_file = metadata_dir / "directory_summaries.json"
        with open(summaries_file, 'w') as f:
            json.dump(directory_summaries, f, indent=2)

        print(f"   ✅ Saved structure overview files")

    def _serialize_structure(self) -> Dict:
        """Serialize directory structure for JSON"""
        def serialize_node(node: DirectoryNode) -> Dict:
            return {
                'path': node.path,
                'name': node.name,
                'child_dirs': [serialize_node(child) for child in node.child_dirs],
                'child_files': node.child_files,
                'file_count': len(node.child_files),
                'total_files': len(node.child_files) + sum(len(c.child_files) for c in node.child_dirs),
                'representative_color': self._get_directory_color(node)
            }

        return serialize_node(self.root_node)

    def _build_semantic_index(self) -> Dict:
        """Build index of concepts -> files"""
        semantic_index = {}

        for file_path, pixel_data in self.file_pixels.items():
            for concept in pixel_data.get('semantic_concepts', []):
                if concept not in semantic_index:
                    semantic_index[concept] = []
                semantic_index[concept].append({
                    'file': file_path,
                    'pixel_rgb': pixel_data['pixel_rgb'],
                    'file_type': pixel_data['file_type']
                })

        # Sort files in each concept by path
        for concept in semantic_index:
            semantic_index[concept] = sorted(semantic_index[concept], key=lambda x: x['file'])

        return semantic_index

    def _build_directory_summaries(self) -> Dict:
        """Build summaries for each directory"""
        summaries = {}

        for dir_path, node in self.structure_map.items():
            # Collect all concepts from files in this directory
            all_concepts = []
            total_size = 0

            for file_rel in node.child_files:
                pixel_data = self.file_pixels.get(file_rel, {})
                all_concepts.extend(pixel_data.get('semantic_concepts', []))
                total_size += pixel_data.get('size_bytes', 0)

            # Count concept frequency
            concept_counts = Counter(all_concepts)

            rel_path = str(Path(dir_path).relative_to(self.source_root))

            summaries[rel_path] = {
                'file_count': len(node.child_files),
                'subdirectory_count': len(node.child_dirs),
                'total_size_bytes': total_size,
                'top_concepts': dict(concept_counts.most_common(5)),
                'representative_color': self._get_directory_color(node),
                'all_concepts': list(set(all_concepts))
            }

        return summaries

    def _get_directory_color(self, node: DirectoryNode) -> List[int]:
        """Calculate representative color for directory based on contained files"""
        if not node.child_files:
            return [128, 128, 128]  # Gray for empty dirs

        # Average colors of files in this directory
        total_r, total_g, total_b = 0, 0, 0
        file_count = 0

        for file_rel in node.child_files:
            pixel_data = self.file_pixels.get(file_rel, {})
            if pixel_data.get('pixel_rgb'):
                r, g, b = pixel_data['pixel_rgb']
                total_r += r
                total_g += g
                total_b += b
                file_count += 1

        if file_count > 0:
            avg_r = total_r // file_count
            avg_g = total_g // file_count
            avg_b = total_b // file_count
            return [avg_r, avg_g, avg_b]
        else:
            return [128, 128, 128]

    def visualize_structure(self):
        """Create ASCII visualization of the preserved structure"""
        print("\n" + "=" * 60)
        print("🌳 DIRECTORY STRUCTURE WITH FRACTAL PIXELS")
        print("=" * 60)

        def visualize_node(node: DirectoryNode, level: int = 0):
            indent = "  " * level
            dir_color = self._get_directory_color(node)

            # Print directory with its representative color
            r, g, b = dir_color
            dir_block = f"\033[48;2;{r};{g};{b}m  \033[0m"

            print(f"\n{indent}{dir_block} 📁 {node.name}/")

            # Print files in this directory
            for file_rel in sorted(node.child_files)[:10]:  # Show first 10 files
                pixel_data = self.file_pixels.get(file_rel, {})
                pixel_rgb = pixel_data.get('pixel_rgb', (128, 128, 128))
                concepts = pixel_data.get('semantic_concepts', [])

                # Create colored block for file
                r, g, b = pixel_rgb
                pixel_block = f"\033[48;2;{r};{g};{b}m  \033[0m"

                file_name = Path(file_rel).name
                concepts_str = ", ".join(concepts[:2]) if concepts else "no concepts"
                size_kb = pixel_data.get('size_bytes', 0) / 1024

                print(f"{indent}  {pixel_block} {file_name:40s} | {size_kb:6.1f}KB | {concepts_str}")

            if len(node.child_files) > 10:
                print(f"{indent}  ... and {len(node.child_files) - 10} more files")

            # Recursively process subdirectories
            for child_dir in sorted(node.child_dirs, key=lambda x: x.name):
                visualize_node(child_dir, level + 1)

        visualize_node(self.root_node)

class StructuredPixelQuery:
    """Query system that understands directory structure"""

    def __init__(self, pixel_db: StructurePreservingPixelDB):
        self.pixel_db = pixel_db

    def find_files_by_concept(self, concept: str, directory: str = None) -> List[Dict]:
        """Find files by concept, optionally filtered by directory"""
        results = []

        for file_rel, pixel_data in self.pixel_db.file_pixels.items():
            if concept in pixel_data.get('semantic_concepts', []):
                # Apply directory filter if specified
                if directory is None or file_rel.startswith(directory):
                    results.append({
                        'file': file_rel,
                        'pixel_rgb': pixel_data['pixel_rgb'],
                        'full_path': str(self.pixel_db.source_root / file_rel),
                        'concepts': pixel_data['semantic_concepts'],
                        'size': pixel_data.get('size_bytes', 0)
                    })

        return results

    def get_directory_summary(self, directory_path: str) -> Dict:
        """Get semantic summary of a directory"""
        dir_node = self.pixel_db.structure_map.get(directory_path)
        if not dir_node:
            return {'error': 'Directory not found'}

        # Collect all concepts from files in this directory
        all_concepts = []
        total_size = 0

        for file_rel in dir_node.child_files:
            pixel_data = self.pixel_db.file_pixels.get(file_rel, {})
            all_concepts.extend(pixel_data.get('semantic_concepts', []))
            total_size += pixel_data.get('size_bytes', 0)

        # Count concept frequency
        concept_counts = Counter(all_concepts)

        return {
            'directory': directory_path,
            'file_count': len(dir_node.child_files),
            'subdirectory_count': len(dir_node.child_dirs),
            'total_size_kb': total_size / 1024,
            'top_concepts': concept_counts.most_common(5),
            'all_concepts': list(set(all_concepts)),
            'representative_color': self.pixel_db._get_directory_color(dir_node)
        }

    def find_related_directories(self, concept: str) -> List[Dict]:
        """Find directories that contain files about a concept"""
        dir_concept_count = {}

        for file_rel, pixel_data in self.pixel_db.file_pixels.items():
            if concept in pixel_data.get('semantic_concepts', []):
                file_dir = str(Path(file_rel).parent)
                if file_dir not in dir_concept_count:
                    dir_concept_count[file_dir] = {'count': 0, 'files': []}
                dir_concept_count[file_dir]['count'] += 1
                dir_concept_count[file_dir]['files'].append(Path(file_rel).name)

        return [
            {
                'directory': dir_path,
                'file_count': data['count'],
                'sample_files': data['files'][:3]
            }
            for dir_path, data in sorted(dir_concept_count.items(),
                                        key=lambda x: x[1]['count'], reverse=True)
        ]

def main():
    """Main execution"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "STRUCTURE-PRESERVING FRACTAL PIXELS" + " " * 15 + "║")
    print("║" + " " * 10 + "Directory Structure + Pixel Intelligence" + " " * 8 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Initialize with pxOS directory
    source_dir = '/home/user/pxos/pxos-v1.0'
    pixel_db = StructurePreservingPixelDB(source_dir)

    # Build the structure with pixels
    pixel_db.build_preserved_structure()

    # Save pixel metadata
    pixel_db.save_pixel_metadata()
    pixel_db.save_structure_overview()

    # Visualize the result
    pixel_db.visualize_structure()

    # Demonstrate queries
    print("\n" + "=" * 60)
    print("🔍 STRUCTURE-AWARE QUERIES")
    print("=" * 60)

    query_engine = StructuredPixelQuery(pixel_db)

    # Query 1: Find all boot files
    print("\n1️⃣  Files about 'boot_system':")
    boot_files = query_engine.find_files_by_concept('boot_system')
    for i, result in enumerate(boot_files[:5], 1):
        rgb = result['pixel_rgb']
        print(f"   {i}. RGB{rgb} | {result['file']}")
    if len(boot_files) > 5:
        print(f"   ... and {len(boot_files) - 5} more")

    # Query 2: Directory summary
    print("\n2️⃣  Directory summary:")
    summary = query_engine.get_directory_summary(source_dir)
    print(f"   Files: {summary['file_count']}")
    print(f"   Subdirectories: {summary['subdirectory_count']}")
    print(f"   Total size: {summary['total_size_kb']:.1f} KB")
    print(f"   Top concepts: {dict(summary['top_concepts'])}")

    # Query 3: Find related directories
    print("\n3️⃣  Directories with 'primitives' concept:")
    prim_dirs = query_engine.find_related_directories('primitives')
    for i, result in enumerate(prim_dirs[:5], 1):
        print(f"   {i}. {result['directory']:40s} | {result['file_count']} files")

    print("\n" + "=" * 60)
    print("✅ PIXEL METADATA SAVED")
    print("=" * 60)
    print(f"📁 Location: {source_dir}/.pixel_metadata/")
    print()
    print("Files created:")
    print("  • .pixel_metadata/*/          - Individual file pixels")
    print("  • directory_structure.json    - Complete structure")
    print("  • pixel_map.json             - File -> Pixel mapping")
    print("  • semantic_index.json        - Concept -> Files index")
    print("  • directory_summaries.json   - Per-directory analysis")
    print()
    print("🎯 Original structure preserved + Pixel intelligence added!")
    print()

if __name__ == "__main__":
    main()
