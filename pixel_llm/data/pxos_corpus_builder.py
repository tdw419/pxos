#!/usr/bin/env python3
"""
pxos_corpus_builder.py - Build Training Corpus from pxOS Repository

Collects all code, documentation, and specifications into a single text corpus
for training Pixel-LLM.

The corpus includes:
- Python source files (.py)
- Documentation (.md)
- Configuration files (.json, .yaml)
- Genesis specification
- Evolution guides

Usage:
    python3 pixel_llm/data/pxos_corpus_builder.py
"""

from pathlib import Path
from typing import List, Set
import json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "pixel_llm" / "data" / "pxos_corpus.txt"

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".py",    # Python code
    ".md",    # Documentation
    ".json",  # Configuration
    ".yaml",  # Templates
    ".txt",   # Text files
}

# Directories to skip
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
}

# Files to skip
SKIP_FILES = {
    ".gitignore",
    ".DS_Store",
    "pxos_corpus.txt",  # Don't include the corpus itself
    "pixellm_v0.npz",   # Don't include binary weights
}


def should_include_file(path: Path) -> bool:
    """Check if file should be included in corpus."""
    # Skip if extension not in whitelist
    if path.suffix not in INCLUDE_EXTENSIONS:
        return False

    # Skip if filename in skip list
    if path.name in SKIP_FILES:
        return False

    # Skip if any parent directory in skip list
    for part in path.parts:
        if part in SKIP_DIRS:
            return False

    return True


def collect_files(root: Path) -> List[Path]:
    """Collect all files for corpus."""
    files = []

    for path in root.rglob("*"):
        if path.is_file() and should_include_file(path):
            files.append(path)

    # Sort for determinism
    files.sort()

    return files


def build_corpus(root: Path, output: Path) -> dict:
    """
    Build the corpus file.

    Returns statistics dict.
    """
    print("="*60)
    print("BUILDING pxOS TRAINING CORPUS")
    print("="*60)
    print(f"Root: {root}")
    print(f"Output: {output}")
    print()

    # Collect files
    print("Collecting files...")
    files = collect_files(root)
    print(f"Found {len(files)} files")
    print()

    # Stats
    stats = {
        "total_files": 0,
        "total_chars": 0,
        "total_lines": 0,
        "by_extension": {},
        "files": [],
    }

    # Write corpus
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        # Header
        f.write("# pxOS Training Corpus\n")
        f.write("# Auto-generated from repository\n")
        f.write("# This corpus contains all code, docs, and specs for training Pixel-LLM\n")
        f.write("\n" + "="*60 + "\n\n")

        # Process each file
        for i, path in enumerate(files, 1):
            rel_path = path.relative_to(root)
            ext = path.suffix

            try:
                text = path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"⚠️  Skipping {rel_path}: {e}")
                continue

            # Write to corpus
            f.write(f"\n{'='*60}\n")
            f.write(f"FILE: {rel_path}\n")
            f.write(f"{'='*60}\n\n")
            f.write(text)
            f.write("\n\n")

            # Update stats
            num_chars = len(text)
            num_lines = text.count("\n") + 1

            stats["total_files"] += 1
            stats["total_chars"] += num_chars
            stats["total_lines"] += num_lines

            if ext not in stats["by_extension"]:
                stats["by_extension"][ext] = {"files": 0, "chars": 0, "lines": 0}

            stats["by_extension"][ext]["files"] += 1
            stats["by_extension"][ext]["chars"] += num_chars
            stats["by_extension"][ext]["lines"] += num_lines

            stats["files"].append({
                "path": str(rel_path),
                "extension": ext,
                "chars": num_chars,
                "lines": num_lines,
            })

            # Progress
            if i % 10 == 0:
                print(f"  Processed {i}/{len(files)} files...")

    print(f"✅ Processed all {len(files)} files")
    print()

    return stats


def print_stats(stats: dict):
    """Print corpus statistics."""
    print("="*60)
    print("CORPUS STATISTICS")
    print("="*60)
    print(f"Total files: {stats['total_files']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Total lines: {stats['total_lines']:,}")
    print()

    print("By file type:")
    print("-"*60)
    for ext, data in sorted(stats["by_extension"].items()):
        print(f"{ext:8s}: {data['files']:4d} files, "
              f"{data['chars']:8,} chars, {data['lines']:6,} lines")
    print("-"*60)
    print()

    # Estimate tokens (rough: ~4 chars per token)
    est_tokens = stats['total_chars'] // 4
    print(f"Estimated tokens: ~{est_tokens:,}")
    print(f"Estimated training batches (64 tokens/batch): ~{est_tokens // 64:,}")
    print()


def main():
    """Build the pxOS corpus."""
    # Build corpus
    stats = build_corpus(ROOT, OUTPUT_PATH)

    # Print stats
    print_stats(stats)

    # Save stats as JSON
    stats_path = OUTPUT_PATH.parent / "pxos_corpus_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Corpus saved to: {OUTPUT_PATH}")
    print(f"   Size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"✅ Stats saved to: {stats_path}")
    print()
    print("="*60)
    print("Ready for training!")
    print("Next step: python3 pixel_llm/models/pixellm_v0_train.py")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
