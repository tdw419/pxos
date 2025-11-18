#!/usr/bin/env bash
# run_ai_build.sh

# This script demonstrates the new two-stage compilation process:
# 1. Python -> IR (using pxpyc.py)
# 2. IR -> Primitives (using ir_compiler.py)

# --- Configuration ---
PYTHON_FILE="pxos/examples/hello.py"
IR_FILE="build/hello.ir.json"
PRIMITIVES_FILE="build/hello.primitives.json"

# --- Build ---
mkdir -p build

echo "--- 1. Compiling Python to IR ---"
python3 tools/pxpyc.py "$PYTHON_FILE" --output_file "$IR_FILE"

echo "--- 2. Compiling IR to Primitives ---"
python3 tools/ir_compiler.py "$IR_FILE" --output_file "$PRIMITIVES_FILE"

echo "--- ✅ Build Complete ---"
echo "IR saved to: $IR_FILE"
echo "Primitives saved to: $PRIMITIVES_FILE"
