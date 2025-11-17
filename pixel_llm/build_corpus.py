#!/usr/bin/env python3
"""
pixel_llm/build_corpus.py

Build training corpus for Pixel-LLM from multiple sources.

This script merges text files from raw/ directory into a single
clean corpus for training.

Usage:
    # Place .txt files in pixel_llm/data/raw/
    python3 pixel_llm/build_corpus.py

    # Result: pixel_llm/data/pxos_corpus.txt
"""

from pathlib import Path
import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    - Normalize line endings
    - Collapse multiple blank lines
    - Strip trailing whitespace
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n')

    # Collapse multiple blank lines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip() + '\n\n'


def build_corpus(
    raw_dir: Path = Path("pixel_llm/data/raw"),
    output_path: Path = Path("pixel_llm/data/pxos_corpus.txt")
):
    """
    Build corpus from raw text files.

    Args:
        raw_dir: Directory containing .txt files
        output_path: Output corpus file
    """
    print("=" * 70)
    print(" BUILDING TRAINING CORPUS")
    print("=" * 70)
    print()

    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all .txt files
    txt_files = sorted(raw_dir.glob("*.txt"))

    if not txt_files:
        print(f"WARNING: No .txt files found in {raw_dir}")
        print("Creating a minimal corpus from defaults...")
        # We'll create a default corpus below
        txt_files = []

    # Merge files
    parts = []
    total_chars = 0

    for txt_file in txt_files:
        print(f"Reading: {txt_file.name}")
        try:
            content = txt_file.read_text(encoding='utf-8', errors='ignore')
            cleaned = clean_text(content)
            parts.append(cleaned)
            total_chars += len(cleaned)
            print(f"  Added {len(cleaned):,} chars")
        except Exception as e:
            print(f"  ERROR: {e}")

    # If no files, create default corpus from pxOS concepts
    if not parts:
        print("\nCreating default pxOS corpus...")
        default_corpus = create_default_corpus()
        parts.append(default_corpus)
        total_chars = len(default_corpus)

    # Write combined corpus
    final_corpus = ''.join(parts)
    output_path.write_text(final_corpus, encoding='utf-8')

    print()
    print(f"Corpus written: {output_path}")
    print(f"  Total size: {len(final_corpus):,} characters")
    print(f"  Total files: {len(txt_files)} source files")
    print()
    print("=" * 70)
    print()


def create_default_corpus() -> str:
    """Create a default corpus about pxOS and computing concepts."""
    return """
Pixel Operating System

pxOS is a pixel-native operating system where programs are stored as PNG images.
Each program is a pixel image containing both code and data.

Neural networks execute natively as pixels. A neural network can be stored in a PNG
file and executed directly without unpacking or conversion. The weights, biases,
and architecture are all encoded in the pixel values.

The system uses quantization to store floating point weights as unsigned bytes.
Each matrix has a scale and offset stored in metadata pixels. This allows lossless
round-trip conversion between float32 and uint8 representations.

Programs use ASCII opcodes for self-documentation. Each operation is represented
by a printable character. For example, M is MatMul, A is Add, R is ReLU, and H is Halt.
This makes programs human-readable when viewed in a hex editor.

The visual layer and machine layer are unified. Text rendering uses bitmap fonts
stored as pixel atlases. Programs can render their own output using glyph blitting
operations. This enables self-visualizing code where the program draws its own state.

The executor is agnostic to backend. The same pixel program can run on CPU or GPU
without modification. CPU execution uses NumPy for operations. GPU execution uses
WebGPU compute shaders for parallel processing. Both backends produce identical results.

Training and inference happen in the same format. Models are trained using standard
techniques, then exported to pixel format. The exported program contains the full
forward pass including embeddings, hidden layers, and output projections.

Autoregressive text generation works by running the pixel program in a loop. Each
iteration produces logits for the next token. Sampling from the logits generates
the next character or word. The generated sequence builds up token by token.

The quantization protocol uses per-matrix scaling. Each matrix stores its own scale
and offset parameters. This allows different matrices to use their full dynamic range.
The result is high numerical accuracy despite using only eight bits per weight.

Instruction format uses four bytes per operation. The first byte is the opcode which
is an ASCII character. The remaining bytes are arguments such as row addresses for
matrix locations. This keeps the instruction encoding compact and readable.

Memory layout follows a grid structure. Matrices are stored row-major in the pixel
image. Each matrix starts with a header containing dimensions. Data follows in
subsequent pixels wrapping at the image width. This enables efficient access patterns.

The development workflow has three phases. First, write programs symbolically using
the assembler. Second, compile to pixel format using quantization. Third, execute
the pixel program and inspect results. The assembler and inspector are lenses to
view pixels, not replacements for them.

Font rendering enables visual output. Bitmap font atlases map characters to glyphs.
Text can be rendered into pixel images using glyph blitting. This creates a visual
representation of generated text. The output is another PNG image showing the text.

Matrix operations form the computational core. Matrix multiplication implements
neural network layers. Element-wise addition handles biases. ReLU activation
introduces nonlinearity. These three operations suffice for most neural architectures.

The system validates numerical correctness. Test programs compare pixel execution
against NumPy reference implementations. Correlation scores measure accuracy.
Values above point nine nine nine indicate faithful execution. Maximum element-wise
error stays below point zero one.

Self-hosting is the long-term vision. The system should compile and run itself
entirely in pixels. Programs that modify programs. A pixel operating system that
manages pixel applications. Complete closure in the visual domain.

Content addressing enables deduplication. Matrices and programs can be stored by
hash. Identical content shares storage. This builds toward a Merkle DAG file system.
Version control becomes native to the architecture.

The pixel format is the canonical representation. Source code is symbolic and
human-friendly. But the truth lives in pixels. Programs are images. Data is pixels.
The visual form is primary, not derived.

Transparency promotes understanding. Every operation can be inspected visually.
Programs render their own state. Debugging tools show pixel contents. The system
resists black boxes. Comprehensibility is architectural.

Small models prove the concept. A tiny character-level model demonstrates end-to-end
functionality. Training, export, execution, and generation all work. Quality improves
with better training data and longer optimization. The architecture scales.

GPU acceleration multiplies throughput. WebGPU shaders parallelize matrix operations.
Workgroups tile across output elements. Each thread computes one result. This achieves
fifty to one hundred times speedup over CPU for large matrices.

The design philosophy emphasizes elegance. Programs should be both executable and
beautiful. Machine code can be human-readable. Visual forms can be computational.
These properties reinforce rather than conflict.

Training happens outside the pixel domain initially. Standard frameworks like PyTorch
or NumPy train the model. Gradients flow through symbolic computation graphs.
Optimization updates parameters in floating point. Then export quantizes and stores
in pixels. Runtime execution uses only the pixel program.

Future directions include learned visual representations. Models could operate directly
on glyph bitmaps. Text becomes images throughout processing. The boundary between
text and vision dissolves. Everything is pixels.

Metadata enriches programs with semantics. Comments, types, and documentation can
attach to pixel programs. These annotations help humans and tools understand intent.
The core execution ignores metadata but development tools consume it.

Composability enables building larger systems. Small programs connect via standard
interfaces. Pixel outputs become pixel inputs. Programs chain into pipelines.
The system grows organically through composition.

Error handling needs graceful degradation. Programs should detect invalid states.
Fallback to safe defaults when possible. Visualize error conditions. Make failure
modes transparent. Debug information stays accessible.

Performance tuning balances multiple constraints. Memory layout affects cache behavior.
Quantization trades precision for size. GPU offload trades latency for throughput.
These tradeoffs require measurement and iteration. Profiling guides optimization.

Documentation serves multiple audiences. End users need usage examples. Developers
need architecture explanation. Contributors need implementation details. Each level
builds on the previous. Clear documentation multiplies impact.

Testing validates correctness at multiple scales. Unit tests check individual operations.
Integration tests verify pipeline stages. End-to-end tests demonstrate full workflows.
Visual tests capture image outputs. Comprehensive testing builds confidence.

The community participates through open development. Source code lives in public
repositories. Issues track bugs and features. Pull requests propose changes.
Discussion happens in the open. Transparency enables contribution.

Versioning tracks evolution over time. Semantic versions mark compatibility boundaries.
Tags capture milestone releases. Branches explore alternatives. Git history preserves
development narrative. Version control is infrastructure.

""".strip()


def main():
    """Build corpus from command line."""
    build_corpus()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
