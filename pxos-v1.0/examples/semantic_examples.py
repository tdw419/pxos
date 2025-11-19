#!/usr/bin/env python3
"""
Semantic Abstraction Layer Examples
====================================

This file demonstrates how different OS operations are processed through
the semantic pipeline: Intent → Concepts → Pixels → Code
"""

import sys
sys.path.insert(0, '..')

from semantic_layer import SemanticPipeline
from pprint import pprint


def print_separator(title: str):
    """Print a section separator"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def show_result(result: dict):
    """Pretty-print pipeline result"""
    print(f"\nPlatform: {result['platform'].upper()}")
    print("-" * 80)

    print("\nIntent:")
    pprint(result['intent'], indent=2)

    print("\nSemantic Concepts:")
    for i, concept in enumerate(result['concepts'], 1):
        print(f"  [{i}] {concept['operation'].upper()}")
        print(f"      Scope:     {concept['scope']}")
        print(f"      Duration:  {concept['duration']}")
        print(f"      Atomicity: {concept['atomicity']}")
        print(f"      Safety:    {concept['safety']}")
        if concept['metadata']:
            print(f"      Metadata:  {concept['metadata']}")

    print("\nPixel Encoding (Intermediate Representation):")
    for i, pixel in enumerate(result['pixels'], 1):
        print(f"  [{i}] {pixel}")

    print("\nGenerated Code:")
    for instruction in result['code']:
        print(f"  {instruction}")
    print()


# =============================================================================
# EXAMPLE 1: Critical Section
# =============================================================================

def example_critical_section():
    """Example: Entering/exiting a critical section"""
    print_separator("EXAMPLE 1: Critical Section (Disable Interrupts)")

    intent = {
        'goal': 'critical_section',
        'context': {
            'reason': 'modifying shared kernel data structure',
            'data_structure': 'process_table'
        },
        'constraints': {
            'max_duration_us': 100,
            'must_be_atomic': True
        }
    }

    print("\nThis demonstrates the CORE VALUE of semantic abstraction:")
    print("  - Same high-level intent across all platforms")
    print("  - Same semantic meaning (isolation, atomic, critical)")
    print("  - Same pixel encoding RGB(255, 32, 255)")
    print("  - Different platform-specific implementations")

    for platform in ['x86_64', 'arm64', 'riscv']:
        pipeline = SemanticPipeline(target_platform=platform)
        result = pipeline.process(intent)
        show_result(result)


# =============================================================================
# EXAMPLE 2: Memory Allocation
# =============================================================================

def example_memory_allocation():
    """Example: Allocating kernel memory"""
    print_separator("EXAMPLE 2: Memory Allocation (4KB Kernel Stack)")

    intent = {
        'goal': 'memory_allocation',
        'context': {
            'size': 4096,
            'alignment': 'page_boundary',
            'purpose': 'kernel'
        },
        'constraints': {
            'must_succeed': True,
            'zero_fill': True
        }
    }

    print("\nThis shows how memory operations are understood semantically:")
    print("  - Operation: MEMORY")
    print("  - Scope: SYSTEM_WIDE (kernel allocation)")
    print("  - Duration: PERSISTENT (not freed immediately)")
    print("  - Safety: CRITICAL (kernel memory)")

    pipeline = SemanticPipeline(target_platform='x86_64')
    result = pipeline.process(intent)
    show_result(result)


# =============================================================================
# EXAMPLE 3: Interrupt Handling
# =============================================================================

def example_interrupt_handling():
    """Example: Handling a hardware interrupt"""
    print_separator("EXAMPLE 3: Interrupt Handling (Timer IRQ)")

    intent = {
        'goal': 'handle_interrupt',
        'context': {
            'irq': 0,  # Timer interrupt
            'device': 'PIT',
            'frequency_hz': 1000
        },
        'constraints': {
            'max_latency_us': 10,
            'must_preserve_state': True
        }
    }

    print("\nInterrupt handling requires MULTIPLE semantic concepts:")
    print("  1. STATE_MANAGEMENT: Save processor state")
    print("  2. INTERRUPT: Handle the actual interrupt")
    print("  3. STATE_MANAGEMENT: Restore processor state")
    print("\nEach concept gets its own pixel encoding!")

    pipeline = SemanticPipeline(target_platform='x86_64')
    result = pipeline.process(intent)
    show_result(result)


# =============================================================================
# EXAMPLE 4: Context Switch
# =============================================================================

def example_context_switch():
    """Example: Switching between processes"""
    print_separator("EXAMPLE 4: Context Switch (Process Scheduler)")

    intent = {
        'goal': 'context_switch',
        'context': {
            'from_pid': 42,
            'to_pid': 137,
            'scheduler': 'round_robin'
        },
        'constraints': {
            'max_latency_us': 50,
            'preserve_all_state': True
        }
    }

    print("\nContext switching is COMPLEX - requires multiple operations:")
    print("  1. Save current process state")
    print("  2. Disable interrupts (isolation)")
    print("  3. Load new process state")
    print("\nThe semantic layer breaks this into atomic concepts!")

    pipeline = SemanticPipeline(target_platform='x86_64')
    result = pipeline.process(intent)
    show_result(result)


# =============================================================================
# EXAMPLE 5: I/O Operation
# =============================================================================

def example_io_operation():
    """Example: Reading from disk"""
    print_separator("EXAMPLE 5: I/O Operation (Disk Read)")

    intent = {
        'goal': 'io_operation',
        'context': {
            'device': 'ATA0',
            'operation': 'read',
            'sector': 42,
            'buffer_addr': 0x100000
        },
        'constraints': {
            'timeout_ms': 1000,
            'retry_on_error': True
        }
    }

    print("\nI/O operations have different characteristics:")
    print("  - Scope: SYSTEM_WIDE (affects hardware)")
    print("  - Duration: TRANSIENT (milliseconds)")
    print("  - Atomicity: BEST_EFFORT (can be interrupted)")
    print("  - Safety: IMPORTANT (data corruption risk)")

    pipeline = SemanticPipeline(target_platform='x86_64')
    result = pipeline.process(intent)
    show_result(result)


# =============================================================================
# EXAMPLE 6: Multi-Platform Comparison
# =============================================================================

def example_multi_platform_comparison():
    """Example: Same intent across all platforms"""
    print_separator("EXAMPLE 6: Multi-Platform Comparison")

    intent = {
        'goal': 'critical_section',
        'context': {
            'reason': 'atomic counter increment'
        }
    }

    print("\nDemonstrating PLATFORM INDEPENDENCE:")
    print("  - Intent is platform-agnostic")
    print("  - Semantic analysis is platform-agnostic")
    print("  - Pixel encoding is platform-agnostic")
    print("  - Only code generation is platform-specific")
    print("\n  Pixels act as UNIVERSAL INTERMEDIATE REPRESENTATION")

    results = {}
    for platform in ['x86_64', 'arm64', 'riscv']:
        pipeline = SemanticPipeline(target_platform=platform)
        results[platform] = pipeline.process(intent)

    # Show that pixels are identical
    print("\n" + "-" * 80)
    print("PIXEL ENCODING (same for all platforms):")
    print(f"  {results['x86_64']['pixels']}")

    print("\nPLATFORM-SPECIFIC CODE:")
    for platform, result in results.items():
        print(f"\n  {platform.upper()}:")
        for instruction in result['code']:
            print(f"    {instruction}")


# =============================================================================
# EXAMPLE 7: Pixel Decoding
# =============================================================================

def example_pixel_decoding():
    """Example: Decoding pixels back to semantic properties"""
    print_separator("EXAMPLE 7: Pixel Decoding (Understanding RGB Values)")

    from semantic_layer import PixelEncoder

    print("\nPixels are NOT arbitrary! Each RGB value has semantic meaning:")
    print("  - Red channel:   Operation type + Safety level")
    print("  - Green channel: Scope + Duration")
    print("  - Blue channel:  Atomicity requirements")

    test_pixels = [
        (255, 32, 255),   # Critical section (isolation)
        (128, 224, 255),  # System-wide memory operation
        (192, 224, 128),  # I/O operation
        (160, 32, 255),   # State management
    ]

    print("\nDecoding example pixels:")
    print("-" * 80)

    for rgb in test_pixels:
        decoded = PixelEncoder.decode(rgb)
        print(f"\n{decoded['rgb']}:")
        print(f"  Operation:  {decoded['operation']}")
        print(f"  Scope:      {decoded['scope']}")
        print(f"  Atomicity:  {decoded['atomicity']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all examples"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               SEMANTIC ABSTRACTION LAYER - COMPREHENSIVE EXAMPLES            ║
║                                                                              ║
║  This demonstrates the semantic-first approach to OS code generation:       ║
║                                                                              ║
║    1. OS Intent     → What is the OS trying to accomplish?                  ║
║    2. Semantic      → What does this mean conceptually?                     ║
║    3. Pixels        → Visual/universal intermediate representation          ║
║    4. Code          → Platform-specific implementation                      ║
║                                                                              ║
║  KEY INSIGHT: Pixels represent CONCEPTS, not arbitrary instructions!        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    examples = [
        ("1", "Critical Section", example_critical_section),
        ("2", "Memory Allocation", example_memory_allocation),
        ("3", "Interrupt Handling", example_interrupt_handling),
        ("4", "Context Switch", example_context_switch),
        ("5", "I/O Operation", example_io_operation),
        ("6", "Multi-Platform Comparison", example_multi_platform_comparison),
        ("7", "Pixel Decoding", example_pixel_decoding),
    ]

    print("\nAvailable examples:")
    for num, name, _ in examples:
        print(f"  [{num}] {name}")
    print("  [all] Run all examples")
    print("  [q] Quit")

    choice = input("\nSelect example (or 'all'): ").strip().lower()

    if choice == 'q':
        return

    if choice == 'all':
        for _, _, func in examples:
            func()
    else:
        for num, _, func in examples:
            if choice == num:
                func()
                break
        else:
            print(f"Invalid choice: {choice}")

    print("\n" + "=" * 80)
    print(" Examples complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
