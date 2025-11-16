#!/usr/bin/env python3
"""
Compare Design Options Using Pixel-LLM

Use Pixel-LLM as a pixel coprocessor to rank design alternatives.
All interaction happens through pixel files - completely pixel-native.

Usage:
    python3 tools/compare_designs.py

Or with custom options:
    python3 tools/compare_designs.py \
        "simple modular design" \
        "complex monolithic architecture" \
        "distributed microservices"
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add pxOS root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pixellm_helper import ask_pixellm


def compare_designs(options: List[str], verbose: bool = False) -> List[Tuple[str, float]]:
    """
    Compare design options using Pixel-LLM.

    Uses Pixel-LLM as a pixel coprocessor to score each option.
    Higher scores suggest the model "prefers" that option based on
    patterns learned from the pxOS corpus.

    Args:
        options: List of design option descriptions
        verbose: Print detailed progress

    Returns:
        List of (option, score) tuples sorted by score (best first)

    Example:
        options = ["simple design", "complex design"]
        ranked = compare_designs(options)
        best_option, best_score = ranked[0]
    """
    scores = []

    print("Consulting Pixel-LLM coprocessor...")
    print()

    for i, option in enumerate(options, 1):
        if verbose:
            print(f"[{i}/{len(options)}] Querying: '{option}'")

        # Query Pixel-LLM (pixel-only API)
        result = ask_pixellm(option, verbose=False)

        score = result['top_prob']
        scores.append((option, score))

        status = "█" * int(score * 40)
        print(f"  {option:40s} [{status:40s}] {score:.3f}")

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare design options using Pixel-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default options
  python3 tools/compare_designs.py

  # Custom options
  python3 tools/compare_designs.py \
      "simple modular design" \
      "complex integrated system" \
      "pixel-based architecture"

  # Verbose mode
  python3 tools/compare_designs.py --verbose
        """
    )

    parser.add_argument(
        "options",
        nargs="*",
        help="Design options to compare (or use defaults)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )

    args = parser.parse_args()

    # Use provided options or defaults
    if args.options:
        options = args.options
    else:
        # Default options
        options = [
            "simple architecture with clear separation",
            "complex architecture with many dependencies",
            "modular pixel-based design",
            "distributed event-driven system",
            "centralized coordinator pattern",
        ]

    print("="*60)
    print("PIXEL-LLM DESIGN COMPARISON")
    print("="*60)
    print()
    print(f"Comparing {len(options)} design options...")
    print()

    # Compare using Pixel-LLM
    ranked = compare_designs(options, verbose=args.verbose)

    # Show results
    print()
    print("="*60)
    print("PIXEL-LLM RANKING")
    print("="*60)
    print()

    for i, (option, score) in enumerate(ranked, 1):
        # Medal for top 3
        if i == 1:
            marker = "🥇"
        elif i == 2:
            marker = "🥈"
        elif i == 3:
            marker = "🥉"
        else:
            marker = "  "

        print(f"  {marker} {i}. {option}")
        print(f"       Score: {score:.3f}")

    print()
    print("="*60)
    print("✅ Ranking complete (pixel coprocessor API used)")
    print()
    print("Note: Scores reflect patterns learned from pxOS corpus.")
    print("      Use as one input among many for design decisions.")
    print("="*60)


if __name__ == "__main__":
    main()
