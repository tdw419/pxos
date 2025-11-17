#!/usr/bin/env python3
"""
pxvm/examples/quick_start.py

Quick start example for LM Studio + Pixel Learning Loop

This example shows the basic workflow:
1. Initialize a pixel network
2. Query LM Studio
3. Append results to network
4. Watch the network grow!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from pxvm.integration.lm_studio_bridge import LMStudioPixelBridge


def basic_example():
    """Basic usage example."""
    print("\n" + "="*70)
    print("📚 BASIC EXAMPLE: Growing a Pixel Network")
    print("="*70)

    # 1. Create a new pixel network
    bridge = LMStudioPixelBridge(
        network_path="pxvm/networks/example_network.png"
    )

    # 2. Ask a question (without accumulated context)
    print("\n1. First query (no context):")
    query = "What is machine learning?"
    print(f"   Q: {query}")
    answer = bridge.ask_lm_studio(query, use_context=False)
    print(f"   A: {answer[:100]}...")

    # 3. Append to network
    bridge.append_interaction(query, answer)
    print("\n   ✅ Appended to network!")

    # 4. Ask another question (with context from first Q&A)
    print("\n2. Second query (with context from previous conversation):")
    query2 = "How does it relate to neural networks?"
    print(f"   Q: {query2}")
    answer2 = bridge.ask_lm_studio(query2, use_context=True)
    print(f"   A: {answer2[:100]}...")

    # 5. Append again
    bridge.append_interaction(query2, answer2)
    print("\n   ✅ Appended to network!")

    # 6. Show final stats
    print("\n3. Final network statistics:")
    bridge._show_network_stats()

    print("\n" + "="*70)
    print("🎉 The network has grown and accumulated knowledge!")
    print("   Try running this script multiple times to see it expand.")
    print("="*70)


def teaching_example():
    """Example of teaching the network specific knowledge."""
    print("\n" + "="*70)
    print("🎓 TEACHING EXAMPLE: Specialized Knowledge")
    print("="*70)

    bridge = LMStudioPixelBridge(
        network_path="pxvm/networks/specialized_network.png"
    )

    # Teach it about a specific domain
    knowledge = [
        ("What is pxOS?", "pxOS is a GPU-native OS where pixels are computational primitives."),
        ("How does quantization work?", "pxOS uses per-matrix scale/offset quantization for uint8 compression."),
        ("What is font-code?", "Font-code is ASCII-based opcodes where M=MatMul, H=Halt, etc.")
    ]

    print("\n📝 Teaching the network...")
    for i, (q, a) in enumerate(knowledge, 1):
        print(f"\n{i}. Teaching: {q}")
        bridge.append_interaction(q, a)
        print(f"   ✅ Taught!")

    print("\n✅ Network now has specialized pxOS knowledge!")
    print("\nNow when you query about pxOS topics, the LLM will have")
    print("access to this accumulated context!")

    # Test it
    print("\n🧪 Testing learned knowledge...")
    test_query = "Explain the pxOS architecture"
    print(f"\nQ: {test_query}")
    answer = bridge.ask_lm_studio(test_query, use_context=True)
    print(f"A: {answer}")

    print("\n" + "="*70)


def interactive_example():
    """Launch interactive mode."""
    print("\n" + "="*70)
    print("💬 INTERACTIVE EXAMPLE")
    print("="*70)

    bridge = LMStudioPixelBridge(
        network_path="pxvm/networks/interactive_network.png"
    )

    bridge.conversational_loop()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick start examples for LM Studio + Pixel Networks"
    )

    parser.add_argument(
        "example",
        choices=["basic", "teaching", "interactive"],
        nargs="?",
        default="basic",
        help="Which example to run"
    )

    args = parser.parse_args()

    if args.example == "basic":
        basic_example()
    elif args.example == "teaching":
        teaching_example()
    elif args.example == "interactive":
        interactive_example()


if __name__ == "__main__":
    main()
