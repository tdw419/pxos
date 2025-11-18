#!/usr/bin/env python3
"""
pxOS Orchestrator v2.0
Orchestrates the AI build loop with L1 utility awareness

This orchestrator implements the two-phase build process:
  Phase A: Check if existing utilities can solve the goal
  Phase B: Generate feature implementation using utilities
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


class PlanningPhase:
    """Phase A: Analyze goals and match to existing utilities"""

    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load library manifest"""
        manifest_file = self.lib_path / "LIB_MANIFEST.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                return json.load(f)
        return {"utilities": []}

    def search_utilities(self, query: str) -> List[Dict]:
        """Search for utilities matching a query"""
        query_lower = query.lower()
        matches = []

        for util in self.manifest.get("utilities", []):
            # Search in name and description
            if (query_lower in util["name"].lower() or
                query_lower in util["description"].lower()):
                matches.append(util)

        return matches

    def analyze_goal(self, goal: str) -> Dict:
        """Analyze a goal and suggest approach"""
        # Extract key terms from goal
        key_terms = self._extract_key_terms(goal)

        # Search for matching utilities
        matching_utilities = []
        for term in key_terms:
            matches = self.search_utilities(term)
            matching_utilities.extend(matches)

        # Remove duplicates
        seen = set()
        unique_matches = []
        for util in matching_utilities:
            if util["name"] not in seen:
                unique_matches.append(util)
                seen.add(util["name"])

        analysis = {
            "goal": goal,
            "key_terms": key_terms,
            "matching_utilities": unique_matches,
            "approach": self._recommend_approach(goal, unique_matches)
        }

        return analysis

    def _extract_key_terms(self, goal: str) -> List[str]:
        """Extract key terms from goal description"""
        # Simple keyword extraction
        keywords = [
            "print", "clear", "read", "write", "screen", "key", "keyboard",
            "string", "char", "character", "newline", "input", "output",
            "backspace", "delete", "edit", "cursor", "help", "command"
        ]

        goal_lower = goal.lower()
        found_terms = [kw for kw in keywords if kw in goal_lower]

        return found_terms

    def _recommend_approach(self, goal: str, matching_utils: List[Dict]) -> str:
        """Recommend implementation approach"""
        if not matching_utils:
            return "new_implementation"

        if len(matching_utils) == 1:
            return "use_single_utility"

        if len(matching_utils) > 1:
            return "compose_utilities"

        return "new_implementation"

    def format_plan(self, analysis: Dict) -> str:
        """Format analysis as a human-readable plan"""
        lines = []
        lines.append("=" * 60)
        lines.append("PLANNING PHASE")
        lines.append("=" * 60)
        lines.append(f"\nGoal: {analysis['goal']}\n")

        if analysis['matching_utilities']:
            lines.append("Matching Utilities Found:")
            for util in analysis['matching_utilities']:
                lines.append(f"  • {util['name']}: {util['description']}")
                lines.append(f"    Size: {util['size_bytes']} bytes")
            lines.append("")

            approach = analysis['approach']
            if approach == "use_single_utility":
                lines.append("Recommended Approach: Use existing utility directly")
            elif approach == "compose_utilities":
                lines.append("Recommended Approach: Compose multiple utilities")
            else:
                lines.append("Recommended Approach: Implement new feature")
        else:
            lines.append("No matching utilities found.")
            lines.append("Recommended Approach: Create new utility or implement from scratch")

        lines.append("")
        lines.append("Next: Generate implementation using utilities")
        lines.append("=" * 60)

        return "\n".join(lines)


class ImplementationPhase:
    """Phase B: Generate feature implementation"""

    def __init__(self, lib_path: Path):
        self.lib_path = lib_path

    def generate_llm_prompt(self, goal: str, analysis: Dict) -> str:
        """Generate prompt for LLM based on planning analysis"""

        # Load utility details
        utility_details = []
        for util in analysis['matching_utilities']:
            util_file = self.lib_path / util['file']
            if util_file.exists():
                with open(util_file, 'r') as f:
                    util_def = json.load(f)
                    utility_details.append(util_def)

        # Build prompt
        prompt_lines = [
            f"Implement the following feature for pxOS: {goal}",
            "",
            "AVAILABLE UTILITIES:",
            ""
        ]

        for util_def in utility_details:
            name = util_def['name']
            desc = util_def['description']
            contract = util_def.get('contract', {})

            prompt_lines.append(f"• {name}")
            prompt_lines.append(f"  Description: {desc}")

            if contract.get('inputs'):
                inputs = ", ".join([
                    f"{k} in {v.get('register', v)}"
                    for k, v in contract['inputs'].items()
                ])
                prompt_lines.append(f"  Inputs: {inputs}")

            if contract.get('outputs'):
                outputs = ", ".join([
                    f"{k} in {v.get('register', v)}"
                    for k, v in contract['outputs'].items()
                ])
                prompt_lines.append(f"  Outputs: {outputs}")

            prompt_lines.append(f"  Usage: UTILITY_CALL {name}")
            prompt_lines.append("")

        prompt_lines.extend([
            "IMPLEMENTATION RULES:",
            "1. PREFER UTILITY_CALL over raw WRITE primitives",
            "2. Use utilities from the list above whenever possible",
            "3. Only write raw primitives if no utility fits",
            "4. Compose multiple utilities to build complex features",
            "",
            "OUTPUT FORMAT (JSON):",
            "{",
            '  "feature": "description",',
            '  "primitives": [',
            '    {"type": "UTILITY_CALL", "name": "util_name", "comment": "what it does"},',
            '    {"type": "WRITE", "address": "0xADDR", "value": "0xVAL", "comment": "..."}',
            '  ]',
            "}",
            "",
            f"Generate implementation for: {goal}"
        ])

        return "\n".join(prompt_lines)


class Orchestrator:
    """Main orchestrator coordinating the build process"""

    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.planning = PlanningPhase(lib_path)
        self.implementation = ImplementationPhase(lib_path)

    def process_goal(self, goal: str, output_prompt: bool = True):
        """Process a build goal through both phases"""

        print("\n" + "=" * 60)
        print("pxOS Orchestrator v2.0 - Utility-Aware Build System")
        print("=" * 60 + "\n")

        # Phase A: Planning
        print("Phase A: Analyzing goal and matching utilities...")
        analysis = self.planning.analyze_goal(goal)
        plan = self.planning.format_plan(analysis)
        print(plan)

        # Phase B: Implementation
        print("\nPhase B: Generating implementation prompt...")
        prompt = self.implementation.generate_llm_prompt(goal, analysis)

        if output_prompt:
            print("\n" + "=" * 60)
            print("LLM PROMPT (send this to your AI model):")
            print("=" * 60 + "\n")
            print(prompt)
            print("\n" + "=" * 60)
            print("\nNext steps:")
            print("1. Copy the prompt above to your LLM (e.g., LM Studio)")
            print("2. Save the JSON response to a file (e.g., feature.json)")
            print("3. Convert to primitives:")
            print("   python3 tools/ai_primitive_generator.py --convert feature.json")
            print("4. Append output to pxos_commands.txt")
            print("5. Build: python3 build_pxos.py")

        return analysis, prompt

    def refactor_mode(self):
        """Scan for abstraction opportunities"""
        print("\n" + "=" * 60)
        print("REFACTORING MODE - Searching for abstraction opportunities")
        print("=" * 60 + "\n")

        # Use pattern detector to find repeated sequences
        from ai_utility_builder import PatternDetector

        primitives_file = self.lib_path.parent / "pxos_commands.txt"
        if not primitives_file.exists():
            print("Error: pxos_commands.txt not found")
            return

        detector = PatternDetector(primitives_file)
        suggestions = detector.analyze_patterns()

        if not suggestions:
            print("No repeated patterns found. System is well-abstracted!")
            return

        print(f"Found {len(suggestions)} potential abstraction opportunities:\n")

        for i, sug in enumerate(suggestions, 1):
            print(f"{i}. {sug['purpose']} ({sug['size_bytes']} bytes)")
            print(f"   Start: {sug['start_address']}")
            print(f"   Can be extracted to util_{sug['purpose']}")
            print()

        print("To create utilities from these patterns:")
        print("  python3 tools/ai_utility_builder.py --from-pattern <N>")


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    lib_path = script_dir.parent / "lib"

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 pxos_orchestrator.py --goal '<description>'")
        print("  python3 pxos_orchestrator.py --refactor")
        print("")
        print("Examples:")
        print("  python3 pxos_orchestrator.py --goal 'Add backspace support to shell'")
        print("  python3 pxos_orchestrator.py --goal 'Implement a help command'")
        print("  python3 pxos_orchestrator.py --refactor")
        sys.exit(1)

    orchestrator = Orchestrator(lib_path)

    if sys.argv[1] == "--goal":
        if len(sys.argv) < 3:
            print("Error: Please provide a goal description")
            sys.exit(1)

        goal = " ".join(sys.argv[2:])
        orchestrator.process_goal(goal)

    elif sys.argv[1] == "--refactor":
        orchestrator.refactor_mode()

    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
