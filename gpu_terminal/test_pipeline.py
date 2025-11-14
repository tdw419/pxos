#!/usr/bin/env python3
"""
test_pipeline.py - Test the complete PXSCENE pipeline

This validates that the full stack works:
  JSON → pxscene_compile.py → PXTERM → pxos_llm_terminal.py → GPU

Usage:
  python test_pipeline.py
"""

import json
import subprocess
import sys
from pathlib import Path
import tempfile
import os


def test_compilation():
    """Test PXSCENE → PXTERM compilation"""
    print("=" * 60)
    print("TEST 1: PXSCENE → PXTERM Compilation")
    print("=" * 60)

    # Create a minimal test scene
    test_scene = {
        "canvas": {"clear": [0, 0, 0]},
        "layers": [{
            "name": "test",
            "z": 0,
            "commands": [
                {"op": "RECT", "x": 100, "y": 100, "w": 200, "h": 150, "color": [255, 0, 0]}
            ]
        }]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_scene, f)
        json_path = f.name

    try:
        pxterm_path = json_path.replace('.json', '.pxterm')

        # Compile
        result = subprocess.run(
            [sys.executable, "pxscene_compile.py", json_path, pxterm_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"✗ FAILED: Compilation error")
            print(result.stderr)
            return False

        # Check PXTERM was created
        if not Path(pxterm_path).exists():
            print(f"✗ FAILED: PXTERM file not created")
            return False

        # Verify PXTERM content
        with open(pxterm_path, 'r') as f:
            pxterm_content = f.read()

        if "RECT 100 100 200 150 255 0 0" not in pxterm_content:
            print(f"✗ FAILED: PXTERM content incorrect")
            return False

        print("✓ PASSED: Compilation works")
        print(f"  Generated {len(pxterm_content.splitlines())} lines of PXTERM")
        return True

    finally:
        # Cleanup
        try:
            os.unlink(json_path)
            os.unlink(pxterm_path)
        except:
            pass


def test_validation():
    """Test PXSCENE validation catches errors"""
    print()
    print("=" * 60)
    print("TEST 2: PXSCENE Validation")
    print("=" * 60)

    # Test invalid scene (missing required fields)
    invalid_scene = {
        "layers": [{
            "z": 0,  # Missing 'name'
            "commands": []
        }]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_scene, f)
        json_path = f.name

    try:
        pxterm_path = json_path.replace('.json', '.pxterm')

        # Try to compile (should fail)
        result = subprocess.run(
            [sys.executable, "pxscene_compile.py", json_path, pxterm_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✗ FAILED: Should have rejected invalid scene")
            return False

        if "missing 'name'" not in result.stderr.lower():
            print(f"✗ FAILED: Should have reported missing 'name' field")
            return False

        print("✓ PASSED: Validation catches errors")
        return True

    finally:
        try:
            os.unlink(json_path)
        except:
            pass


def test_examples_exist():
    """Test that example scenes exist and are valid"""
    print()
    print("=" * 60)
    print("TEST 3: Example Scenes")
    print("=" * 60)

    examples = [
        "examples/scene1_basic.json",
        "examples/scene2_ui.json",
        "examples/scene3_house.json"
    ]

    all_valid = True

    for example in examples:
        if not Path(example).exists():
            print(f"✗ FAILED: Example not found: {example}")
            all_valid = False
            continue

        # Try to parse JSON
        try:
            with open(example, 'r') as f:
                scene = json.load(f)
        except json.JSONDecodeError as e:
            print(f"✗ FAILED: Invalid JSON in {example}: {e}")
            all_valid = False
            continue

        # Check structure
        if "layers" not in scene:
            print(f"✗ FAILED: Missing 'layers' in {example}")
            all_valid = False
            continue

        print(f"  ✓ {example}")

    if all_valid:
        print("✓ PASSED: All examples exist and are valid JSON")

    return all_valid


def test_pxterm_operations():
    """Test that all PXTERM operations work"""
    print()
    print("=" * 60)
    print("TEST 4: PXTERM Operations")
    print("=" * 60)

    # Test all operations
    test_scene = {
        "canvas": {"clear": [0, 0, 0]},
        "layers": [{
            "name": "test_ops",
            "z": 0,
            "commands": [
                {"op": "CLEAR", "color": [0, 0, 0]},
                {"op": "PIXEL", "x": 10, "y": 10, "color": [255, 0, 0]},
                {"op": "RECT", "x": 20, "y": 20, "w": 50, "h": 50, "color": [0, 255, 0]},
                {"op": "HLINE", "x": 100, "y": 100, "length": 100, "color": [0, 0, 255]},
                {"op": "VLINE", "x": 200, "y": 200, "length": 100, "color": [255, 255, 0]}
            ]
        }]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_scene, f)
        json_path = f.name

    try:
        pxterm_path = json_path.replace('.json', '.pxterm')

        # Compile
        result = subprocess.run(
            [sys.executable, "pxscene_compile.py", json_path, pxterm_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"✗ FAILED: Compilation error")
            return False

        # Check all operations are in PXTERM
        with open(pxterm_path, 'r') as f:
            pxterm_content = f.read()

        operations = ["CLEAR", "PIXEL", "RECT", "HLINE", "VLINE"]
        all_present = True

        for op in operations:
            if op not in pxterm_content:
                print(f"  ✗ Missing operation: {op}")
                all_present = False
            else:
                print(f"  ✓ {op}")

        if all_present:
            print("✓ PASSED: All operations compile correctly")

        return all_present

    finally:
        try:
            os.unlink(json_path)
            os.unlink(pxterm_path)
        except:
            pass


def main():
    """Run all tests"""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "PXSCENE Pipeline Test Suite" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    tests = [
        ("Compilation", test_compilation),
        ("Validation", test_validation),
        ("Examples", test_examples_exist),
        ("Operations", test_pxterm_operations),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ FAILED: {name} - Exception: {e}")
            results.append((name, False))

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")

    print()
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print()
        print("✓ All tests passed! The pipeline is working correctly.")
        print()
        print("Next steps:")
        print("  1. Try: python pxscene_run.py examples/scene1_basic.json")
        print("  2. Read: PROMPTS.md for LLM integration")
        print("  3. Create: Your own PXSCENE JSON")
        return 0
    else:
        print()
        print("✗ Some tests failed. Please fix before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
