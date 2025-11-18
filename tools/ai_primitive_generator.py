#!/usr/bin/env python3
"""
tools/ai_primitive_generator.py

AI-powered code generator that converts feature descriptions to pxOS primitives
using a strict JSON schema.
"""
import json
import argparse
from pathlib import Path
import sys

# Add project root to path to allow importing lm_studio_bridge
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pxvm.integration.lm_studio_bridge import LMStudioBridge

def main():
    parser = argparse.ArgumentParser(description="Generate pxOS primitives from a feature description.")
    parser.add_argument("--feature", type=str, required=True, help="A description of the feature to implement.")
    parser.add_argument("--schema", type=Path, default="tools/ai_primitives.schema.json", help="Path to the JSON schema for primitives.")
    args = parser.parse_args()

    # --- Load Schema ---
    with open(args.schema, 'r') as f:
        schema = json.load(f)

    # --- Construct Prompt ---
    prompt = f"""
You are an expert x86 assembly programmer for a primitive-based bootloader OS called pxOS.
Your task is to convert a feature description into a sequence of pxOS primitives, adhering to a strict JSON schema.

Feature Description: "{args.feature}"

You MUST respond with a single JSON object that validates against the following schema:
{json.dumps(schema, indent=2)}
"""

    # --- Call LLM with retry logic ---
    bridge = LMStudioBridge()
    max_retries = 3
    for i in range(max_retries):
        print(f"--- Attempt {i+1}/{max_retries}: Generating primitives for '{args.feature}' ---")
        response_text = bridge.query(prompt)

        try:
            response_json = json.loads(response_text)
            # Basic validation (a proper implementation would use jsonschema)
            if "feature" in response_json and "primitives" in response_json:
                print("--- ✅ Primitives Generated Successfully ---")
                print(json.dumps(response_json, indent=2))
                return
            else:
                print("--- ❌ Invalid JSON structure. Retrying... ---")
        except json.JSONDecodeError:
            print("--- ❌ Invalid JSON response. Retrying... ---")

    print(f"--- ❌ Failed to generate valid primitives after {max_retries} attempts. ---")

if __name__ == "__main__":
    # This is a placeholder for the lm_studio_bridge, which I will create next.
    # For now, I will create a dummy class to allow the script to be created.
    class LMStudioBridge:
        def query(self, prompt):
            # In a real implementation, this would call the LM Studio API.
            # For now, it returns a dummy JSON response.
            return json.dumps({
                "feature": "Dummy feature",
                "primitives": [
                    {
                        "type": "COMMENT",
                        "comment": "This is a dummy response."
                    }
                ]
            })
    main()
