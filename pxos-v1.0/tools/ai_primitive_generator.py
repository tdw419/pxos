#!/usr/bin/env python3
"""
pxOS AI Primitive Generator v2.0
Generates WRITE/DEFINE/UTILITY_CALL primitives with utility library awareness

This generator integrates the L1 utility library, preferring high-level
utility calls over raw primitive sequences.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

class UtilityLibrary:
    """Manages the L1 utility library"""

    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.manifest = self._load_manifest()
        self.utilities = self._load_utilities()

    def _load_manifest(self) -> Dict:
        """Load the library manifest"""
        manifest_file = self.lib_path / "LIB_MANIFEST.json"
        if not manifest_file.exists():
            print(f"Warning: Library manifest not found at {manifest_file}")
            return {"utilities": []}

        with open(manifest_file, 'r') as f:
            return json.load(f)

    def _load_utilities(self) -> Dict[str, Dict]:
        """Load all utility definitions"""
        utilities = {}
        for util_info in self.manifest.get("utilities", []):
            util_name = util_info["name"]
            util_file = self.lib_path / util_info["file"]

            if util_file.exists():
                with open(util_file, 'r') as f:
                    utilities[util_name] = json.load(f)
            else:
                print(f"Warning: Utility file not found: {util_file}")

        return utilities

    def get_utility(self, name: str) -> Optional[Dict]:
        """Get a specific utility definition"""
        return self.utilities.get(name)

    def list_utilities(self, category: Optional[str] = None) -> List[Dict]:
        """List available utilities, optionally filtered by category"""
        if category:
            return [
                u for u in self.manifest.get("utilities", [])
                if u.get("category") == category
            ]
        return self.manifest.get("utilities", [])

    def format_for_prompt(self) -> str:
        """Format utility library for LLM prompt"""
        lines = ["Available L1 Utilities:", ""]

        for util_info in self.manifest.get("utilities", []):
            util_name = util_info["name"]
            util_desc = util_info["description"]
            util_size = util_info["size_bytes"]

            util_def = self.utilities.get(util_name, {})
            contract = util_def.get("contract", {})

            lines.append(f"• {util_name} ({util_size} bytes)")
            lines.append(f"  {util_desc}")

            if contract.get("inputs"):
                inputs = ", ".join([
                    f"{k}={v.get('register', v)}"
                    for k, v in contract["inputs"].items()
                ])
                lines.append(f"  Inputs: {inputs}")

            if contract.get("outputs"):
                outputs = ", ".join([
                    f"{k}={v.get('register', v)}"
                    for k, v in contract["outputs"].items()
                ])
                lines.append(f"  Outputs: {outputs}")

            lines.append(f"  Usage: UTILITY_CALL {util_name}")
            lines.append("")

        return "\n".join(lines)


class PrimitiveGenerator:
    """Generates primitives with utility awareness"""

    def __init__(self, lib_path: Path):
        self.library = UtilityLibrary(lib_path)
        self.current_address = 0x7C00

    def generate_prompt(self, goal: str) -> str:
        """Generate LLM prompt for a given goal"""
        prompt = f"""You are generating pxOS primitives to implement: {goal}

{self.library.format_for_prompt()}

IMPORTANT RULES:
1. PREFER UTILITY_CALL over raw WRITE primitives whenever possible
2. Only use WRITE primitives when:
   - Implementing a new utility
   - No existing utility fits the need
   - Doing very low-level work (boot sector, interrupts)
3. Always check the available utilities list before generating code
4. Use clear comments to explain your implementation

OUTPUT FORMAT (JSON):
{{
  "feature": "description of what this implements",
  "primitives": [
    {{
      "type": "UTILITY_CALL",
      "name": "util_name",
      "comment": "what this call does"
    }},
    {{
      "type": "WRITE",
      "address": "0x7C00",
      "value": "0xFA",
      "comment": "what this byte does"
    }},
    {{
      "type": "COMMENT",
      "text": "explanation of next section"
    }}
  ]
}}

Generate primitives for: {goal}
"""
        return prompt

    def expand_utility_call(self, util_name: str, base_address: int) -> List[str]:
        """Expand a UTILITY_CALL into WRITE primitives"""
        utility = self.library.get_utility(util_name)
        if not utility:
            raise ValueError(f"Unknown utility: {util_name}")

        implementation = utility.get("implementation", {})
        primitives = implementation.get("primitives", [])

        lines = []
        lines.append(f"COMMENT === {util_name} ===")

        for prim in primitives:
            offset = prim.get("offset", 0)
            value = prim.get("value")
            comment = prim.get("comment", "")

            addr = base_address + offset
            lines.append(f"WRITE 0x{addr:04X} {value}    COMMENT {comment}")

        lines.append(f"COMMENT === end {util_name} ===")
        return lines

    def convert_json_to_primitives(self, json_data: Dict, start_address: int = 0x7C00) -> List[str]:
        """Convert JSON primitive specification to pxOS commands"""
        lines = []
        current_addr = start_address

        feature = json_data.get("feature", "Unknown feature")
        lines.append(f"COMMENT ============================================")
        lines.append(f"COMMENT {feature}")
        lines.append(f"COMMENT ============================================")
        lines.append("")

        for prim in json_data.get("primitives", []):
            prim_type = prim.get("type")

            if prim_type == "UTILITY_CALL":
                util_name = prim.get("name")
                comment = prim.get("comment", "")
                inline = prim.get("inline", True)

                lines.append(f"COMMENT {comment}")

                if inline:
                    util_lines = self.expand_utility_call(util_name, current_addr)
                    lines.extend(util_lines)

                    # Update address for next primitive
                    utility = self.library.get_utility(util_name)
                    if utility:
                        size = utility["implementation"]["size_bytes"]
                        current_addr += size
                else:
                    # Generate CALL instruction (future feature)
                    lines.append(f"CALL {util_name}    COMMENT {comment}")

            elif prim_type == "WRITE":
                addr = prim.get("address")
                value = prim.get("value")
                comment = prim.get("comment", "")
                lines.append(f"WRITE {addr} {value}    COMMENT {comment}")
                current_addr += 1

            elif prim_type == "DEFINE":
                label = prim.get("label")
                addr = prim.get("address")
                comment = prim.get("comment", "")
                lines.append(f"DEFINE {label} {addr}    COMMENT {comment}")

            elif prim_type == "COMMENT":
                text = prim.get("text", "")
                lines.append(f"COMMENT {text}")

            lines.append("")

        return lines


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 ai_primitive_generator.py <goal>")
        print("   or: python3 ai_primitive_generator.py --convert <json_file>")
        print("")
        print("Examples:")
        print("  python3 ai_primitive_generator.py 'Add backspace support'")
        print("  python3 ai_primitive_generator.py --convert feature.json")
        sys.exit(1)

    # Find lib directory
    script_dir = Path(__file__).parent
    lib_path = script_dir.parent / "lib"

    generator = PrimitiveGenerator(lib_path)

    if sys.argv[1] == "--convert":
        # Convert JSON to primitives
        json_file = Path(sys.argv[2])
        if not json_file.exists():
            print(f"Error: {json_file} not found")
            sys.exit(1)

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        primitives = generator.convert_json_to_primitives(json_data)
        for line in primitives:
            print(line)

    elif sys.argv[1] == "--list-utilities":
        # List available utilities
        print(generator.library.format_for_prompt())

    else:
        # Generate prompt for LLM
        goal = " ".join(sys.argv[1:])
        prompt = generator.generate_prompt(goal)
        print(prompt)
        print("\n" + "="*60)
        print("Send the above prompt to your LLM (e.g., LM Studio)")
        print("Save the JSON response to a file, then run:")
        print(f"  python3 {sys.argv[0]} --convert response.json")


if __name__ == "__main__":
    main()
