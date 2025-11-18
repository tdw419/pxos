#!/usr/bin/env python3
"""
pxOS AI Utility Builder
Analyzes code patterns and helps create new L1 utilities

This tool assists in:
1. Detecting repeated primitive sequences
2. Extracting them into reusable utilities
3. Generating utility JSON definitions
4. Updating the library manifest
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


class PatternDetector:
    """Detects repeated patterns in primitive files"""

    def __init__(self, primitives_file: Path):
        self.primitives_file = primitives_file
        self.lines = []
        self.write_commands = []

    def load_primitives(self):
        """Load and parse primitive commands"""
        with open(self.primitives_file, 'r') as f:
            self.lines = f.readlines()

        # Extract WRITE commands
        for line in self.lines:
            line = line.strip()
            if line.startswith('WRITE'):
                parts = line.split()
                if len(parts) >= 3:
                    addr = parts[1]
                    value = parts[2]
                    comment = ""
                    if 'COMMENT' in line:
                        comment = line.split('COMMENT', 1)[1].strip()
                    self.write_commands.append({
                        'address': addr,
                        'value': value,
                        'comment': comment,
                        'line': line
                    })

    def find_sequences(self, min_length: int = 5) -> List[List[Dict]]:
        """Find repeated sequences of WRITE commands"""
        sequences = []

        # Look for sequences of consecutive writes
        i = 0
        while i < len(self.write_commands):
            # Try to build a sequence
            sequence = [self.write_commands[i]]

            for j in range(i + 1, len(self.write_commands)):
                # Check if addresses are consecutive
                try:
                    prev_addr = int(self.write_commands[j-1]['address'], 0)
                    curr_addr = int(self.write_commands[j]['address'], 0)

                    if curr_addr == prev_addr + 1:
                        sequence.append(self.write_commands[j])
                    else:
                        break
                except ValueError:
                    break

            if len(sequence) >= min_length:
                sequences.append(sequence)

            i += 1

        return sequences

    def analyze_patterns(self):
        """Analyze primitives and suggest utility candidates"""
        self.load_primitives()
        sequences = self.find_sequences(min_length=4)

        suggestions = []

        for seq in sequences:
            # Try to infer what this sequence does based on comments
            comments = [s['comment'] for s in seq if s['comment']]
            purpose = self._infer_purpose(comments, seq)

            suggestion = {
                'start_address': seq[0]['address'],
                'length': len(seq),
                'size_bytes': len(seq),
                'purpose': purpose,
                'primitives': seq
            }
            suggestions.append(suggestion)

        return suggestions

    def _infer_purpose(self, comments: List[str], sequence: List[Dict]) -> str:
        """Try to infer purpose from comments and opcodes"""
        # Look for key patterns in comments
        comment_text = " ".join(comments).lower()

        if "print" in comment_text or "teletype" in comment_text or "0x0e" in comment_text:
            if "string" in comment_text or "lodsb" in comment_text:
                return "print_string"
            return "print_char"

        if "clear" in comment_text or "0xb800" in comment_text:
            return "clear_screen"

        if "read" in comment_text and ("key" in comment_text or "0x16" in comment_text):
            return "read_key"

        if "newline" in comment_text or ("0x0d" in comment_text and "0x0a" in comment_text):
            return "print_newline"

        # Check for BIOS interrupt patterns
        opcodes = [s['value'] for s in sequence]
        if '0xCD' in opcodes:
            int_idx = opcodes.index('0xCD')
            if int_idx + 1 < len(opcodes):
                int_num = opcodes[int_idx + 1]
                if int_num == '0x10':
                    return "video_operation"
                elif int_num == '0x16':
                    return "keyboard_operation"

        return "unknown_operation"


class UtilityBuilder:
    """Builds new utility definitions"""

    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.manifest_file = lib_path / "LIB_MANIFEST.json"

    def create_utility(self, name: str, description: str, category: str,
                      primitives: List[Dict], contract: Dict) -> Dict:
        """Create a new utility definition"""

        # Build implementation
        implementation_prims = []
        for i, prim in enumerate(primitives):
            implementation_prims.append({
                "type": "WRITE",
                "offset": i,
                "value": prim['value'],
                "comment": prim['comment']
            })

        utility = {
            "name": name,
            "description": description,
            "layer": "L1",
            "contract": contract,
            "implementation": {
                "type": "inline_primitives",
                "primitives": implementation_prims,
                "size_bytes": len(primitives)
            },
            "tests": [
                f"Test {name} with valid inputs",
                f"Verify {name} side effects",
                f"Test {name} edge cases"
            ],
            "usage_example": f"UTILITY_CALL {name} COMMENT {description}"
        }

        return utility

    def save_utility(self, utility: Dict):
        """Save utility to JSON file"""
        util_name = utility["name"]
        util_file = self.lib_path / f"{util_name}.json"

        with open(util_file, 'w') as f:
            json.dump(utility, f, indent=2)

        print(f"Created utility: {util_file}")

    def update_manifest(self, utility: Dict):
        """Add utility to library manifest"""
        # Load existing manifest
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                "manifest_version": "1.0",
                "utilities": [],
                "total_utilities": 0
            }

        # Add new utility entry
        util_entry = {
            "name": utility["name"],
            "file": f"{utility['name']}.json",
            "category": utility.get("category", "system"),
            "size_bytes": utility["implementation"]["size_bytes"],
            "stability": "experimental",
            "description": utility["description"]
        }

        # Check if already exists
        existing = [u for u in manifest["utilities"] if u["name"] == utility["name"]]
        if existing:
            print(f"Warning: Utility {utility['name']} already exists in manifest")
            return

        manifest["utilities"].append(util_entry)
        manifest["total_utilities"] = len(manifest["utilities"])

        # Recalculate total size
        total_size = sum(u["size_bytes"] for u in manifest["utilities"])
        manifest["total_size_bytes"] = total_size

        # Save manifest
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Updated manifest: {self.manifest_file}")

    def interactive_create(self):
        """Interactive utility creation wizard"""
        print("=== pxOS Utility Builder ===\n")

        name = input("Utility name (e.g., util_my_function): ").strip()
        if not name.startswith("util_"):
            name = "util_" + name

        description = input("Description: ").strip()
        category = input("Category (input/output/string/memory/system): ").strip()

        print("\nDefine contract:")
        print("Inputs (comma-separated, e.g., char=AL,count=CX): ", end="")
        inputs_str = input().strip()

        print("Outputs (comma-separated, e.g., result=AX): ", end="")
        outputs_str = input().strip()

        print("Clobbers (comma-separated registers, e.g., AX,BX): ", end="")
        clobbers_str = input().strip()

        # Parse contract
        contract = {
            "inputs": {},
            "outputs": {},
            "clobbers": clobbers_str.split(",") if clobbers_str else []
        }

        for inp in inputs_str.split(","):
            if "=" in inp:
                var, reg = inp.split("=")
                contract["inputs"][var.strip()] = {"register": reg.strip()}

        for out in outputs_str.split(","):
            if "=" in out:
                var, reg = out.split("=")
                contract["outputs"][var.strip()] = {"register": reg.strip()}

        print("\nEnter primitives (one per line, format: address value comment)")
        print("Enter empty line when done:")

        primitives = []
        while True:
            line = input().strip()
            if not line:
                break

            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                primitives.append({
                    "address": parts[0],
                    "value": parts[1],
                    "comment": parts[2] if len(parts) > 2 else ""
                })

        # Create utility
        utility = self.create_utility(name, description, category, primitives, contract)

        # Save
        self.save_utility(utility)
        self.update_manifest(utility)

        print(f"\n✓ Utility {name} created successfully!")


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    lib_path = script_dir.parent / "lib"
    primitives_file = script_dir.parent / "pxos_commands.txt"

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 ai_utility_builder.py --analyze           # Analyze primitives for patterns")
        print("  python3 ai_utility_builder.py --create            # Interactive utility creator")
        print("  python3 ai_utility_builder.py --from-pattern N    # Create utility from Nth pattern")
        sys.exit(1)

    builder = UtilityBuilder(lib_path)

    if sys.argv[1] == "--analyze":
        # Analyze existing primitives
        if not primitives_file.exists():
            print(f"Error: {primitives_file} not found")
            sys.exit(1)

        detector = PatternDetector(primitives_file)
        suggestions = detector.analyze_patterns()

        print(f"Found {len(suggestions)} potential utility patterns:\n")

        for i, sug in enumerate(suggestions, 1):
            print(f"{i}. {sug['purpose']} ({sug['size_bytes']} bytes)")
            print(f"   Start: {sug['start_address']}")
            print(f"   Length: {sug['length']} primitives")
            print()

        if suggestions:
            print("To create a utility from a pattern:")
            print("  python3 ai_utility_builder.py --from-pattern <N>")

    elif sys.argv[1] == "--create":
        # Interactive creation
        builder.interactive_create()

    elif sys.argv[1] == "--from-pattern":
        # Create from detected pattern
        if len(sys.argv) < 3:
            print("Error: Specify pattern number")
            sys.exit(1)

        pattern_num = int(sys.argv[2])

        detector = PatternDetector(primitives_file)
        suggestions = detector.analyze_patterns()

        if pattern_num < 1 or pattern_num > len(suggestions):
            print(f"Error: Pattern {pattern_num} not found")
            sys.exit(1)

        pattern = suggestions[pattern_num - 1]

        # Auto-generate utility name
        purpose = pattern['purpose']
        name = f"util_{purpose}"
        description = f"Auto-detected {purpose} utility"
        category = "system"

        # Create simple contract
        contract = {
            "inputs": {},
            "outputs": {},
            "clobbers": ["AX"],
            "side_effects": [f"Performs {purpose} operation"]
        }

        utility = builder.create_utility(name, description, category,
                                        pattern['primitives'], contract)

        builder.save_utility(utility)
        builder.update_manifest(utility)

        print(f"✓ Created utility: {name}")
        print(f"  Edit {lib_path / name}.json to refine the contract")


if __name__ == "__main__":
    main()
