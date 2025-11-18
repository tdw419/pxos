#!/usr/bin/env python3
"""
AI-Powered Primitive Generator for pxOS (JSON Schema-Based)

This tool uses LM Studio with STRICT JSON contracts to generate primitives.

Key improvements over v1:
- All LLM responses must conform to schemas/ai_primitives.schema.json
- JSON validation with retry on failure
- No free-form text parsing (eliminates hallucination issues)
- Machine-first interface (designed for agent chaining)

The system:
1. Takes a feature description
2. Queries LM Studio with schema-enforced prompt
3. Validates JSON response against schema
4. Converts valid JSON → primitive commands
5. Appends to pxos_commands.txt
6. Learns from successful builds
"""

from pathlib import Path
import sys
import json
import requests
from typing import List, Dict, Optional, Tuple
import jsonschema

# Add pxOS root to path
pxos_root = Path(__file__).resolve().parents[1]
sys.path.append(str(pxos_root))

from pxvm.integration.lm_studio_bridge import LMStudioPixelBridge


class PrimitiveGenerator:
    """
    Generates pxOS primitive commands using LM Studio with JSON contracts.

    This is the AI brain that writes assembly code as primitives!
    Machine-first design: all responses are JSON, validated against schema.
    """

    def __init__(
        self,
        network_path: Path,
        lm_studio_url: str = "http://localhost:1234/v1",
        schema_path: Optional[Path] = None,
        primitive_library_path: Optional[Path] = None
    ):
        self.bridge = LMStudioPixelBridge(network_path, lm_studio_url)

        # Load JSON schema
        if schema_path is None:
            schema_path = pxos_root / "schemas" / "ai_primitives.schema.json"

        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

        # Load primitive library if available
        self.primitive_library = {}
        if primitive_library_path and primitive_library_path.exists():
            self._load_primitive_library(primitive_library_path)

        # Seed knowledge
        self._seed_knowledge()

    def _load_primitive_library(self, library_path: Path):
        """Load pre-built primitive templates."""
        for prim_file in library_path.glob("*.json"):
            with open(prim_file, 'r') as f:
                template = json.load(f)
                self.primitive_library[template['name']] = template

    def _seed_knowledge(self):
        """Seed the pixel network with pxOS + JSON contract knowledge."""
        knowledge = """
=== pxOS PRIMITIVE SYSTEM WITH JSON CONTRACTS ===

## CRITICAL: ALL RESPONSES MUST BE VALID JSON

You MUST respond with JSON conforming to this schema:

{
  "feature": "string describing what this implements",
  "start_address": "0xHHHH",  // optional
  "primitives": [
    {
      "type": "WRITE",
      "addr": "0xHHHH",
      "byte": "0xBB",
      "comment": "optional explanation"
    },
    {
      "type": "DEFINE",
      "label": "symbol_name",
      "addr": "0xHHHH"
    },
    {
      "type": "COMMENT",
      "text": "comment text"
    }
  ]
}

## MEMORY MAP (STRICT CONSTRAINTS)

0x7C00-0x7DFF : Boot sector (512 bytes) - HUMAN REVIEW REQUIRED
0x7E00-0x7FFF : Extended boot code - safe for AI changes
0x0050        : Cursor position storage
0x1FE-0x1FF   : Boot signature 0x55AA - NEVER CHANGE

## x86 OPCODES (VALIDATED SUBSET)

# BIOS Interrupts
0xCD 0x10 : INT 0x10 (video services)
0xCD 0x16 : INT 0x16 (keyboard services)
0xCD 0x13 : INT 0x13 (disk services)

# Video (INT 0x10)
AH=0x0E : Teletype output (AL=char)
AH=0x00 : Set video mode
AH=0x02 : Set cursor position

# Keyboard (INT 0x16)
AH=0x00 : Wait for keypress (returns AL=ASCII)
AH=0x01 : Check for keypress

# Common sequences
Print char 'X':
  0xB4 0x0E      # mov ah, 0x0E
  0xB0 0x58      # mov al, 'X'
  0xCD 0x10      # int 0x10

Wait for key:
  0xB4 0x00      # mov ah, 0x00
  0xCD 0x16      # int 0x16

Backspace:
  0xB4 0x0E      # mov ah, 0x0E
  0xB0 0x08      # mov al, 0x08 (backspace)
  0xCD 0x10      # int 0x10
  0xB0 0x20      # mov al, 0x20 (space to erase)
  0xCD 0x10      # int 0x10
  0xB0 0x08      # mov al, 0x08 (backspace again)
  0xCD 0x10      # int 0x10

## RESPONSE FORMAT (NON-NEGOTIABLE)

✅ CORRECT:
{
  "feature": "Add backspace support",
  "primitives": [...]
}

❌ INCORRECT:
Here's the code for backspace:
WRITE 0x7E00 0xB4
...

Free-form text will be REJECTED. Only JSON is accepted.
"""

        self.bridge.append_interaction(
            "What is the pxOS primitive JSON contract?",
            knowledge
        )

    def generate_primitives(
        self,
        feature_description: str,
        start_address: Optional[int] = None,
        max_attempts: int = 3,
        constraints: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Dict], str]:
        """
        Generate primitive commands for a feature (JSON-validated).

        Args:
            feature_description: What to implement
            start_address: Optional starting address
            max_attempts: Max retry attempts for invalid JSON
            constraints: Additional constraints (e.g., max_address)

        Returns:
            (success, validated_json_dict, raw_response)
        """
        print(f"\n🤖 Generating primitives for: {feature_description}")

        for attempt in range(max_attempts):
            # Build prompt
            prompt = self._build_json_prompt(feature_description, start_address, constraints)

            # Query LM Studio
            response = self.bridge.ask_lm_studio(prompt, use_context=True)

            # Try to extract and validate JSON
            success, validated, error = self._validate_json_response(response)

            if success:
                print(f"✅ Generated valid JSON with {len(validated['primitives'])} primitives")

                # Learn from successful generation
                self.bridge.append_interaction(
                    f"Generate primitives for: {feature_description}",
                    json.dumps(validated, indent=2)
                )

                return True, validated, response

            else:
                print(f"⚠️  Attempt {attempt + 1}/{max_attempts} failed: {error}")
                if attempt < max_attempts - 1:
                    print("   Retrying with stricter prompt...")

        print(f"❌ All {max_attempts} attempts failed validation")
        return False, None, response

    def _build_json_prompt(
        self,
        feature: str,
        start_addr: Optional[int],
        constraints: Optional[Dict]
    ) -> str:
        """Build a strict JSON-enforcing prompt."""

        addr_hint = f"Start at address 0x{start_addr:04X}." if start_addr else "Use addresses in 0x7E00-0x7FFF range."

        constraint_text = ""
        if constraints:
            constraint_text = "\nADDITIONAL CONSTRAINTS:\n"
            for key, val in constraints.items():
                constraint_text += f"- {key}: {val}\n"

        prompt = f"""Generate pxOS primitive commands for this feature:

FEATURE: {feature}

REQUIREMENTS:
1. Respond with ONLY valid JSON (no explanation text before/after)
2. JSON must conform to the pxOS primitive schema
3. {addr_hint}
4. Use known-good x86 opcodes from the knowledge base
5. Include comments explaining each instruction{constraint_text}

RESPONSE FORMAT (MANDATORY):
{{
  "feature": "{feature}",
  "primitives": [
    {{"type": "COMMENT", "text": "Implementation of {feature}"}},
    {{"type": "DEFINE", "label": "feature_label", "addr": "0xHHHH"}},
    {{"type": "WRITE", "addr": "0xHHHH", "byte": "0xBB", "comment": "explanation"}}
  ]
}}

OUTPUT ONLY THE JSON, NOTHING ELSE:
"""
        return prompt

    def _validate_json_response(self, response: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Extract and validate JSON from LLM response.

        Returns:
            (success, validated_dict, error_message)
        """
        # Try to extract JSON from response
        json_str = self._extract_json(response)

        if not json_str:
            return False, None, "No JSON found in response"

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON syntax: {e}"

        # Validate against schema
        try:
            jsonschema.validate(instance=data, schema=self.schema)
        except jsonschema.ValidationError as e:
            return False, None, f"Schema validation failed: {e.message}"

        # Additional pxOS-specific validation
        valid, errors = self._validate_pxos_constraints(data)
        if not valid:
            return False, None, f"pxOS constraint violation: {errors}"

        return True, data, ""

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text (handles markdown code blocks, etc.)."""
        import re

        # Try to find JSON in markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find raw JSON object
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1)

        return None

    def _validate_pxos_constraints(self, data: Dict) -> Tuple[bool, List[str]]:
        """Validate pxOS-specific constraints beyond schema."""
        errors = []

        for prim in data.get('primitives', []):
            if prim['type'] == 'WRITE':
                addr_str = prim['addr']
                addr = int(addr_str, 16)

                # Check boot sector writes (should require human review)
                if 0x7C00 <= addr <= 0x7DFF:
                    errors.append(f"Boot sector write at {addr_str} requires --allow-boot-edit")

                # Check boot signature
                if addr in [0x1FE, 0x1FF]:
                    errors.append(f"Boot signature write at {addr_str} is forbidden")

                # Check valid range
                if addr >= 0x10000:
                    errors.append(f"Address {addr_str} out of range")

        return len(errors) == 0, errors

    def json_to_primitive_lines(self, data: Dict) -> List[str]:
        """Convert validated JSON to primitive command lines."""
        lines = []

        # Header comment
        lines.append(f"COMMENT {'=' * 70}")
        lines.append(f"COMMENT AI-GENERATED: {data['feature']}")
        lines.append(f"COMMENT {'=' * 70}")
        lines.append("")

        # Convert each primitive
        for prim in data['primitives']:
            ptype = prim['type']

            if ptype == 'WRITE':
                addr = prim['addr']
                byte = prim['byte']
                comment = prim.get('comment', '')
                if comment:
                    lines.append(f"WRITE {addr} {byte}    COMMENT {comment}")
                else:
                    lines.append(f"WRITE {addr} {byte}")

            elif ptype == 'DEFINE':
                label = prim['label']
                addr = prim['addr']
                lines.append(f"DEFINE {label} {addr}")

            elif ptype == 'COMMENT':
                text = prim['text']
                lines.append(f"COMMENT {text}")

            elif ptype == 'CALL':
                label = prim['label']
                comment = prim.get('comment', '')
                if comment:
                    lines.append(f"CALL {label}    COMMENT {comment}")
                else:
                    lines.append(f"CALL {label}")

        return lines

    def append_to_commands_file(
        self,
        data: Dict,
        commands_file: Path
    ):
        """Append validated JSON primitives to pxos_commands.txt."""
        lines = self.json_to_primitive_lines(data)

        with open(commands_file, 'a') as f:
            f.write('\n\n')
            for line in lines:
                f.write(line + '\n')

        print(f"✅ Appended {len(data['primitives'])} primitives to {commands_file}")


def main():
    """Interactive AI primitive generator with JSON validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Powered pxOS Primitive Generator (JSON Schema-Based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate primitives for a feature
  python3 tools/ai_primitive_generator.py --feature "Add backspace support"

  # Interactive mode
  python3 tools/ai_primitive_generator.py --interactive

  # Specify starting address
  python3 tools/ai_primitive_generator.py --feature "Help command" --addr 0x7E00

  # Machine mode (JSON output only, for agent chaining)
  python3 tools/ai_primitive_generator.py --feature "clear screen" --machine
"""
    )

    parser.add_argument(
        "--network",
        default="pxvm/networks/pxos_dev.png",
        help="Pixel network for AI learning"
    )
    parser.add_argument(
        "--feature",
        help="Feature description to generate"
    )
    parser.add_argument(
        "--addr",
        help="Starting address (hex, e.g., 0x7E00)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive generation mode"
    )
    parser.add_argument(
        "--machine",
        action="store_true",
        help="Machine mode: output only JSON, no human-friendly messages"
    )
    parser.add_argument(
        "--output",
        default="pxos-v1.0/pxos_commands.txt",
        help="Commands file to append to"
    )

    args = parser.parse_args()

    # Parse start address if provided
    start_addr = None
    if args.addr:
        start_addr = int(args.addr, 16) if args.addr.startswith('0x') else int(args.addr, 0)

    # Initialize generator
    if not args.machine:
        print("🚀 Initializing AI Primitive Generator (JSON Schema Mode)...")

    generator = PrimitiveGenerator(
        network_path=Path(args.network)
    )

    if args.interactive:
        # Interactive loop
        print("\n" + "="*70)
        print("🎨 INTERACTIVE PRIMITIVE GENERATION (JSON Schema-Validated)")
        print("="*70)
        print("Describe features and I'll generate primitives!")
        print("All responses are JSON-validated for correctness.")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                feature = input("\n💡 Feature to implement: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting...")
                break

            if not feature or feature.lower() == 'exit':
                break

            success, data, raw = generator.generate_primitives(
                feature,
                start_addr
            )

            if success:
                print("\n📝 Generated Primitives:")
                print("-" * 70)
                lines = generator.json_to_primitive_lines(data)
                for line in lines:
                    print(line)
                print("-" * 70)

                print("\n📊 JSON Structure:")
                print(json.dumps(data, indent=2))

                save = input("\n💾 Append to commands file? (y/n): ").strip().lower()
                if save == 'y':
                    generator.append_to_commands_file(
                        data,
                        Path(args.output)
                    )
            else:
                print("\n❌ Generation failed after max retries")
                if not args.machine:
                    print("Raw response:", raw[:500])

    elif args.feature:
        # Single feature generation
        success, data, raw = generator.generate_primitives(
            args.feature,
            start_addr
        )

        if success:
            if args.machine:
                # Machine mode: output only JSON
                print(json.dumps(data))
            else:
                print("\n📝 Generated Primitives:")
                lines = generator.json_to_primitive_lines(data)
                for line in lines:
                    print(line)

                generator.append_to_commands_file(
                    data,
                    Path(args.output)
                )
        else:
            if args.machine:
                print(json.dumps({"error": "Generation failed", "raw": raw}))
            else:
                print("\n❌ Generation failed")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
