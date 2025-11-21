"""
pxos/eval/vram_os_evaluator.py

Evaluation framework for VRAM OS quality.

This runs checks on a built VRAM snapshot to measure:
- Structural integrity (are regions properly initialized?)
- Feature completeness (is the opcode palette populated?)
- Program correctness (can we decode the hello program?)

The evaluator emits metrics that the improvement loop uses to guide changes.
"""

from pathlib import Path
import json
from typing import Dict, Any
from pxos.vram_sim import SimulatedVRAM
from pxos.layout import constants as L


class VRAMOSEvaluator:
    """Evaluator for VRAM OS snapshots."""

    def __init__(self, vram: SimulatedVRAM):
        self.vram = vram
        self.metrics: Dict[str, Any] = {}

    def evaluate_all(self) -> Dict[str, Any]:
        """Run all evaluation checks and return metrics."""
        self.check_metadata_region()
        self.check_opcode_palette()
        self.check_program_area()
        self.check_region_colors()
        self.compute_overall_score()
        return self.metrics

    def check_metadata_region(self):
        """Check that metadata region is initialized."""
        meta = L.REGION_METADATA
        sample_pixel = self.vram.read_pixel(meta["x"] + 10, meta["y"] + 2)

        # Check if it's non-black (indicates initialization)
        is_nonblack = any(channel > 0 for channel in sample_pixel[:3])
        self.metrics["metadata_initialized"] = is_nonblack

    def check_opcode_palette(self):
        """Check opcode palette diversity."""
        palette = L.REGION_OPCODE_PALETTE
        palette_y = palette["y"] + (palette["h"] // 2)

        # Sample colors across the palette band
        colors = set()
        sample_points = 16
        for i in range(sample_points):
            x = (i * self.vram.width) // sample_points
            color = self.vram.read_pixel(x, palette_y)
            colors.add(color)

        self.metrics["opcode_palette_unique_colors"] = len(colors)
        self.metrics["opcode_palette_sufficient"] = len(colors) >= 4

    def check_program_area(self):
        """Check that program area has been written to."""
        prog = L.REGION_PROGRAM_AREA
        bg_color = (20, 20, 20, 255)

        # Sample a small region where we expect the hello program
        sample_x = 32
        sample_y = prog["y"] + 8
        sample_w = 32
        sample_h = 8

        non_bg_pixels = 0
        for dy in range(sample_h):
            for dx in range(sample_w):
                pixel = self.vram.read_pixel(sample_x + dx, sample_y + dy)
                if pixel[:3] != bg_color[:3]:
                    non_bg_pixels += 1

        self.metrics["hello_program_pixels"] = non_bg_pixels
        self.metrics["hello_program_exists"] = non_bg_pixels > 0

    def check_region_colors(self):
        """Verify that each region has its expected base color."""
        regions_ok = 0
        regions_checked = 0

        for name, region in [
            ("metadata", L.REGION_METADATA),
            ("kernel", L.REGION_KERNEL),
            ("syscall_table", L.REGION_SYSCALL_TABLE),
            ("process_table", L.REGION_PROCESS_TABLE),
        ]:
            if "color" not in region:
                continue

            regions_checked += 1
            expected_color = region["color"]

            # Sample center of region
            sample_x = region["x"] + region["w"] // 2
            sample_y = region["y"] + region["h"] // 2
            actual_color = self.vram.read_pixel(sample_x, sample_y)

            # Check if colors roughly match (allow some tolerance)
            if self._colors_match(expected_color, actual_color):
                regions_ok += 1

        self.metrics["regions_checked"] = regions_checked
        self.metrics["regions_ok"] = regions_ok
        self.metrics["all_regions_ok"] = regions_ok == regions_checked

    def _colors_match(self, c1, c2, tolerance=10):
        """Check if two colors are within tolerance."""
        return all(abs(a - b) <= tolerance for a, b in zip(c1[:3], c2[:3]))

    def compute_overall_score(self):
        """Compute a simple overall quality score."""
        score = 0
        max_score = 0

        # Metadata region (1 point)
        max_score += 1
        if self.metrics.get("metadata_initialized", False):
            score += 1

        # Opcode palette (1 point)
        max_score += 1
        if self.metrics.get("opcode_palette_sufficient", False):
            score += 1

        # Program area (1 point)
        max_score += 1
        if self.metrics.get("hello_program_exists", False):
            score += 1

        # Region colors (1 point)
        max_score += 1
        if self.metrics.get("all_regions_ok", False):
            score += 1

        self.metrics["score"] = score
        self.metrics["max_score"] = max_score
        self.metrics["score_pct"] = (score / max_score * 100) if max_score > 0 else 0


def evaluate_vram_os_from_png(png_path: str) -> Dict[str, Any]:
    """Load a VRAM PNG and evaluate it."""
    vram = SimulatedVRAM.load_png(png_path)
    evaluator = VRAMOSEvaluator(vram)
    return evaluator.evaluate_all()


def evaluate_and_save(png_path: str, output_json: str, verbose: bool = True):
    """Evaluate a VRAM PNG and save metrics to JSON."""
    if verbose:
        print(f"[evaluator] Loading VRAM from {png_path}")

    metrics = evaluate_vram_os_from_png(png_path)

    # Save metrics
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"[evaluator] Saved metrics to {output_json}")
        print(f"[evaluator] Score: {metrics['score']}/{metrics['max_score']} ({metrics['score_pct']:.1f}%)")
        print()
        print("Metrics:")
        for key, value in metrics.items():
            if key not in ["score", "max_score", "score_pct"]:
                print(f"  {key}: {value}")

    return metrics
