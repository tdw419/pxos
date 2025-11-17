#!/usr/bin/env python3
"""
pxvm/dev/inspector.py

Pixel Program Inspector - View and understand .pxi programs.

Philosophy:
- This is a LENS to view pixels, not a replacement for pixels
- Read-only operations - never modifies the program
- Helps humans and LLMs understand what's in the pixels

Usage:
    from pxvm.dev import PixelInspector

    inspector = PixelInspector("program.pxi")
    inspector.summary()           # Program overview
    inspector.instructions()      # List all instructions
    inspector.matrix_info("W_hidden")  # Show matrix details
    inspector.health_check()      # Detect issues
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from pxvm.core.opcodes import (
    OP_HALT, OP_ADD, OP_RELU, OP_MATMUL, OP_DOT,
    opcode_to_char, opcode_name, format_instruction,
    is_legacy_opcode, migrate_legacy_opcode
)

# Legacy alias
OP_DOT_RGB = OP_DOT
from pxvm.core.interpreter import _read_shape
from pxvm.visual.text_render import FontAtlas, create_text_image


class PixelInspector:
    """
    Inspector for .pxi pixel programs.

    This is a development tool that helps you understand what's in a pixel
    program without modifying it. Think of it as a specialized viewer for
    the pixel format.
    """

    OPCODE_NAMES = {
        OP_HALT: "HALT",
        OP_DOT_RGB: "DOT_RGB",
        OP_ADD: "ADD",
        OP_RELU: "RELU",
        OP_MATMUL: "MATMUL",
    }

    def __init__(self, program_path: Path | str):
        """
        Load a pixel program for inspection.

        Args:
            program_path: Path to .pxi file
        """
        self.path = Path(program_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Program not found: {self.path}")

        # Load as RGBA numpy array
        self.img = np.array(Image.open(self.path).convert("RGBA"), dtype=np.uint8)
        self.height, self.width, self.channels = self.img.shape

        # Cache discovered structures
        self._instructions_cache = None
        self._matrices_cache = None

    def summary(self) -> None:
        """Print high-level program summary."""
        print("=" * 70)
        print(f" PIXEL PROGRAM: {self.path.name}")
        print("=" * 70)
        print()
        print(f"Image Dimensions: {self.width}×{self.height} RGBA")
        print(f"File Size: {self.path.stat().st_size:,} bytes")
        print()

        instructions = self._get_instructions()
        print(f"Instructions: {len(instructions)} opcodes")

        matrices = self._discover_matrices()
        print(f"Matrices/Vectors: {len(matrices)} detected")
        print()

        # Opcode breakdown
        opcode_counts = {}
        for instr in instructions:
            opcode_name = self.OPCODE_NAMES.get(instr[0], f"UNKNOWN({instr[0]})")
            opcode_counts[opcode_name] = opcode_counts.get(opcode_name, 0) + 1

        print("Opcode Distribution:")
        for opcode, count in opcode_counts.items():
            print(f"  {opcode:10s}: {count}")
        print()
        print("=" * 70)

    def instructions(self, verbose: bool = True) -> List[Tuple]:
        """
        List all instructions in the program.

        Args:
            verbose: If True, print formatted output

        Returns:
            List of (opcode, arg0, arg1, arg2) tuples
        """
        instructions = self._get_instructions()

        if verbose:
            print("=" * 70)
            print(" INSTRUCTIONS")
            print("=" * 70)
            print()

            for i, instr in enumerate(instructions):
                opcode, arg0, arg1, arg2 = instr
                opcode_name = self.OPCODE_NAMES.get(opcode, f"UNKNOWN({opcode})")

                # Format instruction with semantics
                if opcode == OP_MATMUL:
                    desc = f"C[row{arg2}] = A[row{arg0}] @ B[row{arg1}]"
                elif opcode == OP_ADD:
                    desc = f"C[row{arg2}] = A[row{arg0}] + B[row{arg1}]"
                elif opcode == OP_RELU:
                    desc = f"row{arg0} = relu(row{arg0})"
                elif opcode == OP_DOT_RGB:
                    desc = f"row{arg2} = dot(row{arg0}, row{arg1})"
                elif opcode == OP_HALT:
                    desc = "Stop execution"
                else:
                    desc = f"args=({arg0}, {arg1}, {arg2})"

                print(f"  [{i}] {opcode_name:10s}  {desc}")

            print()
            print("=" * 70)

        return instructions

    def visualize_instructions(self, output_path: Optional[Path] = None) -> None:
        """
        Render instruction row as visual text using font atlas.

        This demonstrates the ASCII opcode design: instructions are both
        executable machine code AND human-readable text.

        Args:
            output_path: Optional path to save visualization (defaults to program_visual.png)
        """
        print("=" * 70)
        print(" VISUAL INSTRUCTION RENDERING (ASCII Opcodes)")
        print("=" * 70)
        print()

        # Extract instruction row (row 0, R channel)
        instr_row = self.img[0, :, 0]

        # Convert to text, handling legacy opcodes
        text_parts = []
        readable_parts = []

        for i in range(self.width):
            opcode = int(instr_row[i])

            if opcode == 0:  # End of instructions
                break

            # Migrate legacy opcodes if needed
            if is_legacy_opcode(opcode):
                print(f"  Note: Found legacy opcode {opcode}, migrating to ASCII")
                opcode = migrate_legacy_opcode(opcode)

            # Get character representation
            char = opcode_to_char(opcode)
            name = opcode_name(opcode)

            text_parts.append(char)
            readable_parts.append(name)

        # Create visual text
        visual_text = ''.join(text_parts)
        readable_text = ' → '.join(readable_parts)

        print(f"Visual (ASCII):  {visual_text}")
        print(f"Readable:        {readable_text}")
        print()

        # Load font atlas
        root = Path(__file__).resolve().parents[2]
        font_png = root / "fonts" / "ascii_16x16.png"
        font_json = root / "fonts" / "ascii_16x16.json"

        if not font_png.exists():
            print("WARNING: Font atlas not found, skipping image output")
            print(f"  Run: python3 -m pxvm.visual.font_atlas")
            print()
            print("=" * 70)
            return

        # Render as image
        font = FontAtlas(font_png, font_json)

        # Create title + instructions image
        full_text = f"PIXEL PROGRAM: {self.path.name}\n\nInstructions (ASCII):\n{visual_text}\n\nReadable:\n{readable_text}"

        visual_img = create_text_image(
            full_text,
            font,
            padding=20,
            background=(20, 25, 35, 255),
            text_color=(180, 220, 255)
        )

        # Save
        if output_path is None:
            output_path = self.path.parent / f"{self.path.stem}_visual.png"

        Image.fromarray(visual_img).save(output_path)

        print(f"Saved visual rendering: {output_path}")
        print(f"  Image size: {visual_img.shape[1]}×{visual_img.shape[0]} RGBA")
        print()
        print("=" * 70)

    def matrix_info(self, row: int, name: Optional[str] = None) -> None:
        """
        Show detailed information about a matrix at given row.

        Args:
            row: Row number where matrix starts
            name: Optional symbolic name for display
        """
        try:
            cols, rows = _read_shape(self.img, row)
        except:
            print(f"❌ No valid matrix header at row {row}")
            return

        # Calculate how many image rows this matrix spans
        total_elements = cols * rows
        stride = self.width - 1  # Column 0 is header
        data_rows = (total_elements + stride - 1) // stride
        total_rows = 1 + data_rows  # +1 for header

        # Read data statistics
        data_values = []
        for i in range(total_elements):
            x = 1 + (i % stride)
            y = row + (i // stride)
            if y < self.height:
                data_values.append(int(self.img[y, x, 0]))

        data_values = np.array(data_values)

        print("=" * 70)
        if name:
            print(f" MATRIX: {name} (row {row})")
        else:
            print(f" MATRIX at row {row}")
        print("=" * 70)
        print()
        print(f"Shape: {cols}×{rows} (cols × rows)")
        print(f"Total Elements: {total_elements:,}")
        print(f"Image Rows Used: {total_rows} (rows {row}-{row+total_rows-1})")
        print()
        print("Data Statistics:")
        print(f"  Min: {data_values.min()}")
        print(f"  Max: {data_values.max()}")
        print(f"  Mean: {data_values.mean():.2f}")
        print(f"  Std Dev: {data_values.std():.2f}")
        print(f"  Non-zero: {np.count_nonzero(data_values)} / {len(data_values)}")
        print()

        # Show first few values
        print("First 20 values:")
        print(f"  {data_values[:20].tolist()}")
        print()
        print("=" * 70)

    def matrices(self) -> Dict[int, Tuple[int, int]]:
        """
        Discover all matrices in the program.

        Returns:
            Dict mapping row_start → (cols, rows)
        """
        matrices = self._discover_matrices()

        print("=" * 70)
        print(" MATRICES/VECTORS")
        print("=" * 70)
        print()

        for row_start, (cols, rows) in sorted(matrices.items()):
            total_elements = cols * rows
            stride = self.width - 1
            total_rows = 1 + ((total_elements + stride - 1) // stride)

            print(f"Row {row_start:4d}: {cols}×{rows:4d}  "
                  f"({total_elements:6,} elements, {total_rows:3d} rows)")

        print()
        print(f"Total: {len(matrices)} matrices/vectors")
        print("=" * 70)

        return matrices

    def health_check(self) -> Dict[str, List[str]]:
        """
        Check for common issues in the program.

        Returns:
            Dict with 'errors', 'warnings', 'info' lists
        """
        issues = {
            'errors': [],
            'warnings': [],
            'info': [],
        }

        # Check 1: Valid HALT instruction
        instructions = self._get_instructions()
        if not instructions:
            issues['errors'].append("No instructions found (row 0 empty)")
        elif instructions[-1][0] != OP_HALT:
            issues['errors'].append("Program does not end with HALT")

        # Check 2: Instruction arguments within image bounds
        for i, instr in enumerate(instructions):
            opcode, arg0, arg1, arg2 = instr

            if opcode != OP_HALT:
                for arg_num, arg_val in enumerate([arg0, arg1, arg2]):
                    if arg_val > 0 and arg_val >= self.height:
                        issues['errors'].append(
                            f"Instruction {i}: arg{arg_num}={arg_val} exceeds height {self.height}"
                        )

        # Check 3: Matrix header validity
        matrices = self._discover_matrices()
        for row_start, (cols, rows) in matrices.items():
            if cols == 0 or rows == 0:
                issues['warnings'].append(f"Matrix at row {row_start} has zero dimension: {cols}×{rows}")

            # Check if matrix data fits in image
            total_elements = cols * rows
            stride = self.width - 1
            data_rows_needed = (total_elements + stride - 1) // stride
            last_row = row_start + data_rows_needed

            if last_row >= self.height:
                issues['errors'].append(
                    f"Matrix at row {row_start} ({cols}×{rows}) exceeds image height "
                    f"(needs rows {row_start}-{last_row}, have {self.height})"
                )

        # Check 4: MATMUL shape compatibility
        for i, instr in enumerate(instructions):
            if instr[0] == OP_MATMUL:
                row_a, row_b, row_c = instr[1], instr[2], instr[3]

                try:
                    cols_a, rows_a = _read_shape(self.img, row_a)
                    cols_b, rows_b = _read_shape(self.img, row_b)

                    # For A @ B, need: cols_a == rows_b
                    if cols_a != rows_b:
                        issues['errors'].append(
                            f"Instruction {i} (MATMUL): Shape mismatch - "
                            f"A[row{row_a}]={cols_a}×{rows_a}, B[row{row_b}]={cols_b}×{rows_b}, "
                            f"incompatible (need cols_a={cols_a} == rows_b={rows_b})"
                        )
                except:
                    issues['warnings'].append(
                        f"Instruction {i} (MATMUL): Could not read matrix shapes"
                    )

        # Check 5: uint8 addressing constraints
        for i, instr in enumerate(instructions):
            opcode, arg0, arg1, arg2 = instr
            if opcode in [OP_MATMUL, OP_ADD, OP_DOT_RGB]:
                if arg0 > 255 or arg1 > 255 or arg2 > 255:
                    issues['errors'].append(
                        f"Instruction {i}: Arguments exceed uint8 range (max 255)"
                    )

        # Print health report
        print("=" * 70)
        print(" HEALTH CHECK")
        print("=" * 70)
        print()

        if not issues['errors'] and not issues['warnings']:
            print("✅ No issues detected")
            issues['info'].append("Program appears healthy")
        else:
            if issues['errors']:
                print(f"❌ ERRORS ({len(issues['errors'])}):")
                for error in issues['errors']:
                    print(f"  • {error}")
                print()

            if issues['warnings']:
                print(f"⚠️  WARNINGS ({len(issues['warnings'])}):")
                for warning in issues['warnings']:
                    print(f"  • {warning}")
                print()

        print("=" * 70)

        return issues

    def flow_graph(self) -> None:
        """
        Show data flow through the program.

        Traces which rows are read/written by each instruction.
        """
        instructions = self._get_instructions()

        print("=" * 70)
        print(" DATA FLOW")
        print("=" * 70)
        print()

        row_first_write = {}  # Track first write to each row
        row_reads = {}  # Track reads from each row

        for i, instr in enumerate(instructions):
            opcode, arg0, arg1, arg2 = instr
            opcode_name = self.OPCODE_NAMES.get(opcode, f"UNKNOWN({opcode})")

            if opcode == OP_MATMUL:
                # Reads: arg0, arg1; Writes: arg2
                reads = [arg0, arg1]
                writes = [arg2]
            elif opcode == OP_ADD:
                # Reads: arg0, arg1; Writes: arg2
                reads = [arg0, arg1]
                writes = [arg2]
            elif opcode == OP_RELU:
                # Reads and writes: arg0
                reads = [arg0]
                writes = [arg0]
            elif opcode == OP_DOT_RGB:
                # Reads: arg0, arg1; Writes: arg2
                reads = [arg0, arg1]
                writes = [arg2]
            else:
                reads = []
                writes = []

            # Track first write
            for row in writes:
                if row not in row_first_write:
                    row_first_write[row] = i

            # Track reads
            for row in reads:
                if row not in row_reads:
                    row_reads[row] = []
                row_reads[row].append(i)

            # Print flow
            if reads or writes:
                read_str = ", ".join(f"row{r}" for r in reads) if reads else "-"
                write_str = ", ".join(f"row{r}" for r in writes) if writes else "-"
                print(f"  [{i}] {opcode_name:10s}  reads: {read_str:20s} writes: {write_str}")

        print()
        print("Row Usage Summary:")

        all_rows = set(row_first_write.keys()) | set(row_reads.keys())
        for row in sorted(all_rows):
            first_write_at = row_first_write.get(row, None)
            read_at = row_reads.get(row, [])

            status = []
            if first_write_at is not None:
                status.append(f"written at instr {first_write_at}")
            if read_at:
                status.append(f"read at instr {read_at}")

            print(f"  row {row:3d}: {', '.join(status)}")

        print()
        print("=" * 70)

    # Private helper methods

    def _get_instructions(self) -> List[Tuple[int, int, int, int]]:
        """Extract instructions from row 0."""
        if self._instructions_cache is not None:
            return self._instructions_cache

        instructions = []
        for x in range(self.width):
            instr = self.img[0, x]
            opcode = int(instr[0])

            if opcode == OP_HALT:
                instructions.append((opcode, int(instr[1]), int(instr[2]), int(instr[3])))
                break
            elif opcode in self.OPCODE_NAMES:
                instructions.append((opcode, int(instr[1]), int(instr[2]), int(instr[3])))
            else:
                # Unknown opcode or end of instructions
                if opcode != 0:  # Don't stop on zeros (could be padding)
                    break

        self._instructions_cache = instructions
        return instructions

    def _discover_matrices(self) -> Dict[int, Tuple[int, int]]:
        """
        Scan all rows looking for valid matrix headers.

        Returns:
            Dict mapping row_start → (cols, rows)
        """
        if self._matrices_cache is not None:
            return self._matrices_cache

        matrices = {}

        for row in range(1, self.height):  # Skip row 0 (instructions)
            try:
                cols, rows = _read_shape(self.img, row)

                # Validate: non-zero dimensions, reasonable values
                if cols > 0 and rows > 0 and cols < 65536 and rows < 65536:
                    matrices[row] = (cols, rows)
            except:
                pass

        self._matrices_cache = matrices
        return matrices


def main():
    """CLI interface for pixel inspector."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 -m pxvm.dev.inspector <program.pxi>")
        return

    program_path = sys.argv[1]
    inspector = PixelInspector(program_path)

    # Run full inspection
    inspector.summary()
    print()
    inspector.instructions()
    print()
    inspector.matrices()
    print()
    inspector.flow_graph()
    print()
    inspector.health_check()


if __name__ == "__main__":
    main()
