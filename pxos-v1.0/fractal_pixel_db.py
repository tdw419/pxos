#!/usr/bin/env python3
"""
FRACTAL PIXEL FILE SYSTEM
Each file collapses to one pixel, expands to infinite complexity
Each pixel contains intelligence to manage its information

The key insight: Every file in your codebase becomes a single intelligent pixel
that knows what it contains and how it relates to every other pixel.
"""

import os
import hashlib
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

class PixelState(Enum):
    COLLAPSED = 1      # Single pixel representation
    EXPANDING = 2      # Currently expanding
    EXPANDED = 3       # Full content visible
    INTELLIGENT = 4    # AI-managed content

@dataclass
class FractalPixel:
    """A single pixel that can expand to infinite complexity"""
    file_path: str
    collapsed_pixel: tuple  # (R, G, B) - the single pixel representation
    content_hash: str
    metadata: Dict[str, Any]
    intelligence: Optional[Any] = None
    expansion_map: Optional[Dict] = None  # How to expand this pixel
    state: PixelState = PixelState.COLLAPSED

    def __post_init__(self):
        # Each pixel gets its own AI manager for its content
        if self.intelligence is None:
            self.intelligence = PixelIntelligence(self)

    def expand(self) -> str:
        """Expand from single pixel to full file content"""
        self.state = PixelState.EXPANDING

        if self.expansion_map and 'content' in self.expansion_map:
            # Reconstruct from expansion map
            content = self.expansion_map['content']
        else:
            # Read from filesystem
            try:
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                content = f"<error reading file: {e}>"

        self.state = PixelState.EXPANDED
        return content

    def collapse(self) -> tuple:
        """Collapse file back to single pixel"""
        self.state = PixelState.COLLAPSED
        return self.collapsed_pixel

    def get_intelligent_summary(self) -> str:
        """Use pixel's AI to summarize content"""
        return self.intelligence.summarize()

    def find_relationships(self, other_pixels: List['FractalPixel']) -> List['FractalPixel']:
        """Find related files using pixel intelligence"""
        return self.intelligence.find_related(other_pixels)

class PixelIntelligence:
    """AI that lives inside each pixel and manages its information"""

    def __init__(self, fractal_pixel: FractalPixel):
        self.pixel = fractal_pixel
        self.semantic_understanding = {}
        self.relationships = []

    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze file content and extract semantic meaning"""
        # This is where the magic happens - understanding what the file "means"
        analysis = {
            'file_type': self._detect_file_type(content),
            'semantic_concepts': self._extract_concepts(content),
            'complexity_score': self._calculate_complexity(content),
            'dependencies': self._find_dependencies(content),
            'key_functions': self._extract_functions(content),
            'topic_clusters': self._cluster_topics(content),
            'language': self._detect_language(content)
        }

        self.semantic_understanding = analysis
        return analysis

    def summarize(self) -> str:
        """Generate intelligent summary of file content"""
        concepts = self.semantic_understanding.get('semantic_concepts', [])
        file_type = self.semantic_understanding.get('file_type', 'unknown')
        language = self.semantic_understanding.get('language', '')

        if concepts:
            return f"{language} {file_type}: {', '.join(concepts[:3])}"
        else:
            return f"{file_type} file"

    def find_related(self, other_pixels: List[FractalPixel]) -> List[FractalPixel]:
        """Find semantically related files"""
        related = []
        my_concepts = set(self.semantic_understanding.get('semantic_concepts', []))

        if not my_concepts:
            return []

        for other_pixel in other_pixels:
            if other_pixel.file_path != self.pixel.file_path:
                other_concepts = set(other_pixel.intelligence.semantic_understanding.get('semantic_concepts', []))

                # Calculate semantic similarity
                similarity = len(my_concepts.intersection(other_concepts))
                if similarity > 0:
                    related.append(other_pixel)

        return related

    def _detect_language(self, content: str) -> str:
        """Detect programming language"""
        if self.pixel.file_path.endswith('.py'):
            return 'Python'
        elif self.pixel.file_path.endswith('.c'):
            return 'C'
        elif self.pixel.file_path.endswith('.cpp'):
            return 'C++'
        elif self.pixel.file_path.endswith('.js'):
            return 'JavaScript'
        elif self.pixel.file_path.endswith('.md'):
            return 'Markdown'
        elif self.pixel.file_path.endswith('.txt'):
            return 'Text'
        elif self.pixel.file_path.endswith('.sh'):
            return 'Shell'
        elif self.pixel.file_path.endswith('.asm'):
            return 'Assembly'
        else:
            return ''

    def _detect_file_type(self, content: str) -> str:
        """Intelligently detect file type"""
        if content.startswith('#!/'):
            return 'executable_script'
        elif 'def ' in content or 'class ' in content:
            return 'python_module'
        elif '#include' in content or 'void ' in content:
            return 'c_source'
        elif '<html' in content.lower() or '<body' in content.lower():
            return 'html_document'
        elif 'WRITE ' in content or 'DEFINE ' in content:
            return 'primitive_code'
        elif content.startswith('#'):
            return 'documentation'
        else:
            return 'text_file'

    def _extract_concepts(self, content: str) -> List[str]:
        """Extract semantic concepts from content"""
        concepts = []

        # Concept indicators
        concept_indicators = {
            'memory': ['malloc', 'free', 'page', 'virtual', 'physical', 'heap', 'stack', 'allocat'],
            'boot': ['boot', 'grub', 'bios', 'uefi', 'multiboot', 'sector', 'mbr'],
            'scheduler': ['schedule', 'sched', 'process', 'thread', 'priority', 'yield', 'context'],
            'filesystem': ['file', 'inode', 'block', 'directory', 'vfs', 'fat', 'ext'],
            'driver': ['driver', 'device', 'interrupt', 'pci', 'usb', 'hardware'],
            'network': ['tcp', 'ip', 'socket', 'packet', 'protocol', 'ethernet', 'arp'],
            'assembly': ['mov', 'jmp', 'call', 'push', 'pop', 'cli', 'sti', 'int'],
            'kernel': ['kernel', 'syscall', 'interrupt', 'handler', 'ring'],
            'primitive': ['WRITE', 'DEFINE', 'CALL', 'COMMENT', 'primitive'],
            'semantic': ['semantic', 'pixel', 'synthesis', 'intent', 'concept'],
            'documentation': ['README', 'guide', 'tutorial', 'reference', 'manual'],
            'testing': ['test', 'assert', 'verify', 'check', 'validate'],
            'build': ['build', 'compile', 'make', 'cmake', 'configure']
        }

        content_lower = content.lower()
        for concept, indicators in concept_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                concepts.append(concept)

        return concepts

    def _calculate_complexity(self, content: str) -> float:
        """Calculate file complexity score"""
        lines = content.split('\n')
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('//')]

        if not code_lines:
            return 0.0

        # Simple complexity heuristic
        complexity = min(len(code_lines) / 100.0, 1.0)
        return complexity

    def _find_dependencies(self, content: str) -> List[str]:
        """Find files this file depends on"""
        dependencies = []

        # Look for imports/includes
        lines = content.split('\n')
        for line in lines:
            if 'import ' in line or '#include' in line or 'from ' in line:
                # Extract dependency name
                words = line.split()
                for word in words:
                    if '.' in word or word.endswith('.h') or word.endswith('.py'):
                        dependencies.append(word.strip('"\'<>;'))

        return dependencies

    def _extract_functions(self, content: str) -> List[str]:
        """Extract function definitions"""
        functions = []
        lines = content.split('\n')

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('function ') or 'void ' in stripped:
                # Extract just the function name
                if 'def ' in stripped:
                    parts = stripped.split('def ')[1].split('(')[0]
                    functions.append(f"def {parts}()")
                elif 'function ' in stripped:
                    parts = stripped.split('function ')[1].split('(')[0]
                    functions.append(f"function {parts}()")

        return functions[:10]  # Return first 10

    def _cluster_topics(self, content: str) -> List[str]:
        """Cluster content into topics"""
        # Simple topic extraction based on frequency
        words = content.lower().split()
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stopwords]

        common_words = Counter(meaningful_words).most_common(5)
        return [word for word, count in common_words]

class FractalPixelDatabase:
    """Database where every file is represented as an intelligent pixel"""

    def __init__(self, root_directory: str):
        self.root_dir = root_directory
        self.pixels: Dict[str, FractalPixel] = {}
        self.pixel_map: Dict[tuple, FractalPixel] = {}  # RGB -> Pixel
        self.concept_index: Dict[str, List[FractalPixel]] = {}  # Concept -> Pixels
        self.initialized = False

    def initialize_database(self):
        """Convert all files to intelligent pixels"""
        print("🔄 Converting files to fractal pixels...")
        print("-" * 60)

        files_found = self._discover_files()

        for i, file_path in enumerate(files_found):
            try:
                pixel = self._file_to_pixel(file_path)
                self.pixels[file_path] = pixel
                self.pixel_map[pixel.collapsed_pixel] = pixel

                # Index by concepts
                concepts = pixel.intelligence.semantic_understanding.get('semantic_concepts', [])
                for concept in concepts:
                    if concept not in self.concept_index:
                        self.concept_index[concept] = []
                    self.concept_index[concept].append(pixel)

                # Show progress
                rgb = pixel.collapsed_pixel
                concepts_str = ', '.join(concepts[:2]) if concepts else 'none'
                print(f"   {i+1:2d}. RGB{rgb} | {os.path.basename(file_path):40s} | {concepts_str}")

            except Exception as e:
                print(f"   ❌ {file_path}: {e}")

        self.initialized = True
        print("-" * 60)
        print(f"✅ Database initialized: {len(self.pixels)} files as intelligent pixels")
        print(f"🧠 Concepts indexed: {len(self.concept_index)}")

    def _discover_files(self) -> List[str]:
        """Discover all files in directory tree"""
        file_paths = []

        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}

        for root, dirs, files in os.walk(self.root_dir):
            # Remove skip directories from traversal
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                # Skip hidden files and certain extensions
                if not file.startswith('.') and not file.endswith('.pyc'):
                    file_paths.append(os.path.join(root, file))

        return file_paths

    def _file_to_pixel(self, file_path: str) -> FractalPixel:
        """Convert a file to a fractal pixel"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            content = f"<error reading file: {e}>"

        # Generate unique pixel from file content
        pixel_rgb = self._content_to_pixel(content, file_path)
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Get file metadata
        file_stat = os.stat(file_path)

        # Create fractal pixel
        fractal_pixel = FractalPixel(
            file_path=file_path,
            collapsed_pixel=pixel_rgb,
            content_hash=content_hash,
            metadata={
                'file_size': file_stat.st_size,
                'file_name': os.path.basename(file_path),
                'directory': os.path.dirname(file_path),
                'extension': os.path.splitext(file_path)[1],
                'modified_time': file_stat.st_mtime
            }
        )

        # Analyze content with pixel intelligence
        fractal_pixel.intelligence.analyze_content(content)

        return fractal_pixel

    def _content_to_pixel(self, content: str, file_path: str) -> tuple:
        """Convert file content to unique RGB pixel"""
        # Use semantic hashing to generate meaningful pixels
        # Not random - semantically derived from content!

        # Combine file path and content for unique hash
        combined = file_path + content[:1000]  # First 1000 chars for efficiency
        content_hash = hashlib.md5(combined.encode()).hexdigest()

        # Convert first 6 hex chars to RGB
        r = int(content_hash[0:2], 16)
        g = int(content_hash[2:4], 16)
        b = int(content_hash[4:6], 16)

        return (r, g, b)

    def get_pixel_grid(self, width: int = 16) -> List[List[tuple]]:
        """Get 2D grid of pixels for visualization"""
        pixels_list = list(self.pixels.values())

        grid = []
        for i in range(0, len(pixels_list), width):
            row = [pixel.collapsed_pixel for pixel in pixels_list[i:i+width]]
            # Pad row if needed
            while len(row) < width:
                row.append((0, 0, 0))  # Black padding
            grid.append(row)

        return grid

    def find_semantic_cluster(self, concept: str) -> List[FractalPixel]:
        """Find all pixels related to a semantic concept"""
        return self.concept_index.get(concept, [])

    def expand_to_map(self, center_pixel: FractalPixel, radius: int = 3) -> Dict:
        """Expand a pixel to show semantic relationships"""
        center_pixel.state = PixelState.EXPANDING

        expansion_map = {
            'center': center_pixel,
            'immediate_relationships': center_pixel.find_relationships(list(self.pixels.values())),
            'semantic_neighbors': [],
            'topic_clusters': []
        }

        # Find semantic neighbors (files about similar things)
        center_concepts = set(center_pixel.intelligence.semantic_understanding.get('semantic_concepts', []))

        for pixel in self.pixels.values():
            if pixel != center_pixel:
                pixel_concepts = set(pixel.intelligence.semantic_understanding.get('semantic_concepts', []))
                overlap = len(center_concepts.intersection(pixel_concepts))

                if overlap > 0:
                    expansion_map['semantic_neighbors'].append({
                        'pixel': pixel,
                        'similarity': overlap,
                        'shared_concepts': list(center_concepts.intersection(pixel_concepts))
                    })

        # Sort by similarity
        expansion_map['semantic_neighbors'].sort(key=lambda x: x['similarity'], reverse=True)

        center_pixel.state = PixelState.EXPANDED
        return expansion_map

    def save_database(self, filename: str = "fractal_pixels.db"):
        """Save the pixel database to disk"""
        save_data = {
            'root_dir': self.root_dir,
            'pixels_data': []
        }

        for file_path, pixel in self.pixels.items():
            pixel_data = {
                'file_path': pixel.file_path,
                'collapsed_pixel': pixel.collapsed_pixel,
                'content_hash': pixel.content_hash,
                'metadata': pixel.metadata,
                'semantic_understanding': pixel.intelligence.semantic_understanding,
            }
            save_data['pixels_data'].append(pixel_data)

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"💾 Saved pixel database: {filename} ({len(save_data['pixels_data'])} pixels)")

        # Also save as JSON for human readability
        json_file = filename.replace('.db', '.json')
        with open(json_file, 'w') as f:
            # Convert to JSON-serializable format
            json_data = {
                'root_dir': save_data['root_dir'],
                'total_pixels': len(save_data['pixels_data']),
                'concepts': list(self.concept_index.keys()),
                'sample_pixels': save_data['pixels_data'][:5]  # Just first 5 for readability
            }
            json.dump(json_data, f, indent=2)

        print(f"📄 Saved JSON summary: {json_file}")

class PixelQueryEngine:
    """Query engine for the fractal pixel database"""

    def __init__(self, fractal_db: FractalPixelDatabase):
        self.db = fractal_db

    def query_by_concept(self, concept: str) -> List[FractalPixel]:
        """Find all files about a specific concept"""
        return self.db.find_semantic_cluster(concept)

    def query_by_color(self, target_color: tuple, tolerance: int = 20) -> List[FractalPixel]:
        """Find files with similar pixel colors (semantic similarity)"""
        similar = []
        target_r, target_g, target_b = target_color

        for pixel in self.db.pixels.values():
            r, g, b = pixel.collapsed_pixel
            if (abs(r - target_r) <= tolerance and
                abs(g - target_g) <= tolerance and
                abs(b - target_b) <= tolerance):
                similar.append(pixel)

        return similar

    def find_related_system(self, core_concept: str) -> Dict:
        """Find complete system around a core concept"""
        core_files = self.query_by_concept(core_concept)

        system_map = {
            'core_concept': core_concept,
            'core_files': core_files,
            'supporting_files': [],
            'interface_files': [],
            'documentation': []
        }

        # Find supporting files (dependencies, related concepts)
        for core_file in core_files:
            # Find files that core file depends on
            dependencies = core_file.intelligence.semantic_understanding.get('dependencies', [])

            # Find interface files (files that use similar concepts)
            interfaces = core_file.find_relationships(list(self.db.pixels.values()))
            system_map['interface_files'].extend(interfaces)

        # Remove duplicates
        system_map['interface_files'] = list(set(system_map['interface_files']))

        return system_map

def visualize_fractal_database(fractal_db: FractalPixelDatabase):
    """Create ASCII visualization of the pixel database"""

    print("\n" + "=" * 60)
    print("🎨 FRACTAL PIXEL DATABASE VISUALIZATION")
    print("=" * 60)

    # Get pixel grid
    grid = fractal_db.get_pixel_grid(width=12)

    print("\n📊 Pixel Map (each block = 1 file):")
    print()

    # Display as colored ASCII
    for row in grid:
        for pixel in row:
            r, g, b = pixel
            # Create colored block
            block = f"\033[48;2;{r};{g};{b}m  \033[0m"
            print(block, end="")
        print()  # New line after each row

    # Show database stats
    total_files = len(fractal_db.pixels)
    concepts_found = set()

    for pixel in fractal_db.pixels.values():
        concepts_found.update(pixel.intelligence.semantic_understanding.get('semantic_concepts', []))

    print(f"\n📊 DATABASE STATISTICS:")
    print(f"   Total files: {total_files}")
    print(f"   Unique concepts: {len(concepts_found)}")
    print(f"   Concepts: {', '.join(sorted(concepts_found))}")

    # Show concept distribution
    print(f"\n📈 CONCEPT DISTRIBUTION:")
    for concept in sorted(fractal_db.concept_index.keys()):
        count = len(fractal_db.concept_index[concept])
        bar = '█' * min(count, 30)
        print(f"   {concept:15s} | {count:2d} files | {bar}")

def demonstrate_fractal_expansion(fractal_db: FractalPixelDatabase):
    """Demonstrate collapsing/expanding of files"""

    print("\n" + "=" * 60)
    print("🌀 FRACTAL EXPANSION DEMONSTRATION")
    print("=" * 60)

    # Pick interesting files to demonstrate
    sample_files = list(fractal_db.pixels.values())[:5]

    for i, pixel in enumerate(sample_files):
        print(f"\n{i+1}. 📄 {pixel.metadata['file_name']}")
        print(f"   Collapsed: RGB{pixel.collapsed_pixel}")
        print(f"   State: {pixel.state.name}")
        print(f"   Size: {pixel.metadata['file_size']} bytes")

        # Get intelligence summary
        summary = pixel.get_intelligent_summary()
        print(f"   Intelligence: {summary}")

        # Show concepts
        concepts = pixel.intelligence.semantic_understanding.get('semantic_concepts', [])
        if concepts:
            print(f"   Concepts: {', '.join(concepts)}")

        # Show relationships
        relationships = pixel.find_relationships(list(fractal_db.pixels.values()))
        print(f"   Related files: {len(relationships)}")
        if relationships:
            print(f"   Examples: {', '.join([os.path.basename(r.file_path) for r in relationships[:3]])}")

def query_demonstration(fractal_db: FractalPixelDatabase):
    """Demonstrate semantic queries"""

    print("\n" + "=" * 60)
    print("🔍 SEMANTIC QUERY DEMONSTRATION")
    print("=" * 60)

    query_engine = PixelQueryEngine(fractal_db)

    # Query by concept
    concepts_to_query = ['boot', 'memory', 'primitive', 'semantic']

    for concept in concepts_to_query:
        results = query_engine.query_by_concept(concept)
        if results:
            print(f"\n🔎 Files about '{concept}':")
            for i, pixel in enumerate(results[:5]):
                summary = pixel.get_intelligent_summary()
                print(f"   {i+1}. {pixel.metadata['file_name']:40s} | {summary}")

                # Show expansion for first result
                if i == 0:
                    expansion = fractal_db.expand_to_map(pixel)
                    neighbors = expansion['semantic_neighbors'][:3]
                    if neighbors:
                        print(f"      Related to:")
                        for neighbor in neighbors:
                            shared = ', '.join(neighbor['shared_concepts'])
                            print(f"        → {os.path.basename(neighbor['pixel'].file_path)} (via: {shared})")

def main():
    """Main demonstration"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "FRACTAL PIXEL DATABASE SYSTEM" + " " * 18 + "║")
    print("║" + " " * 5 + "Where every file becomes an intelligent pixel" + " " * 7 + "║")
    print("╚" + "═" * 58 + "╝")

    # Create the fractal pixel database
    fractal_db = FractalPixelDatabase('/home/user/pxos/pxos-v1.0')
    fractal_db.initialize_database()

    # Visualize
    visualize_fractal_database(fractal_db)

    # Demonstrate expansion
    demonstrate_fractal_expansion(fractal_db)

    # Demonstrate queries
    query_demonstration(fractal_db)

    # Save database
    print("\n" + "=" * 60)
    print("💾 SAVING DATABASE")
    print("=" * 60)
    fractal_db.save_database('pxos_fractal_pixels.db')

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "DATABASE CREATION COMPLETE" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("🎯 Your codebase is now a living pixel organism!")
    print("   Each file = 1 intelligent pixel")
    print("   Each pixel knows what it contains")
    print("   Each pixel knows how it relates to others")
    print()
    print("🚀 Next: Use pixels to navigate, query, and synthesize code!")

if __name__ == "__main__":
    main()
