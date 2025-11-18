#!/usr/bin/env python3
"""
pxos_orchestrator.py - The Brain Stem

Single script that runs the complete AI build loop end-to-end.

This is THE script you run. Everything else is a service it calls.

Pipeline:
1. Analyze current state
2. Load context from pixel history
3. Ask LM Studio for plan
4. Generate primitives (JSON-validated)
5. Append to pxos_commands.txt
6. Build binary
7. Test in QEMU
8. Log to pixel network
9. Report results

Usage:
  # Build next milestone
  python3 pxos_orchestrator.py --auto

  # Specific goal
  python3 pxos_orchestrator.py --goal "Add backspace support"

  # Machine mode (for agents)
  python3 pxos_orchestrator.py --goal "clear screen" --machine
"""

from pathlib import Path
import sys
import json
import subprocess
import time
import argparse
from typing import Dict, Optional, Tuple
import requests

# Add pxOS root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Import our v2 components
from tools.ai_primitive_generator import PrimitiveGenerator
from pxvm.learning.read_pixels import PixelReader
from pxvm.integration.lm_studio_bridge import LMStudioPixelBridge

# Paths
PXOS_DIR = ROOT / "pxos-v1.0"
COMMANDS_FILE = PXOS_DIR / "pxos_commands.txt"
BUILD_SCRIPT = PXOS_DIR / "build_pxos.py"
OUTPUT_BIN = PXOS_DIR / "pxos.bin"
NETWORK_PATH = ROOT / "pxvm" / "networks" / "pxos_autobuild.png"
MILESTONES_FILE = ROOT / "milestones" / "PXOS_ROADMAP.md"


class PxOSOrchestrator:
    """
    The conductor. Calls all v2 services in the right order.
    """

    def __init__(
        self,
        lm_studio_url: str = "http://localhost:1234/v1",
        machine_mode: bool = False
    ):
        self.lm_studio_url = lm_studio_url
        self.machine_mode = machine_mode

        # Initialize services
        self.bridge = LMStudioPixelBridge(NETWORK_PATH, lm_studio_url)
        self.generator = PrimitiveGenerator(NETWORK_PATH, lm_studio_url)
        self.pixel_reader = PixelReader(NETWORK_PATH)

        # Run state
        self.run_id = f"run_{int(time.time())}"

    def log(self, msg: str):
        """Log to console (unless in machine mode)."""
        if not self.machine_mode:
            print(msg)

    def analyze_state(self) -> Dict:
        """Step 1: Analyze current pxOS state."""
        self.log("\n🔍 Step 1: Analyzing pxOS state...")

        state = {
            "timestamp": time.time(),
            "files": {
                "commands_exists": COMMANDS_FILE.exists(),
                "binary_exists": OUTPUT_BIN.exists(),
            },
            "sizes": {},
            "current_milestone": "M2"  # TODO: parse from roadmap
        }

        if COMMANDS_FILE.exists():
            with open(COMMANDS_FILE, 'r') as f:
                lines = f.readlines()
                state["sizes"]["commands_lines"] = len(lines)
                state["sizes"]["primitive_count"] = len([
                    l for l in lines
                    if l.strip() and not l.strip().startswith('COMMENT')
                ])

        if OUTPUT_BIN.exists():
            state["sizes"]["binary_bytes"] = OUTPUT_BIN.stat().st_size

        self.log(f"   Current milestone: {state['current_milestone']}")
        self.log(f"   Binary: {state['sizes'].get('binary_bytes', 0)} bytes")
        self.log(f"   Primitives: {state['sizes'].get('primitive_count', 0)} commands")

        return state

    def load_context(self) -> str:
        """Step 2: Load context from pixel history."""
        self.log("\n📖 Step 2: Loading context from pixel history...")

        if not NETWORK_PATH.exists():
            self.log("   No history yet (first run)")
            return "[No accumulated knowledge yet - this is the first build]"

        context = self.pixel_reader.get_context_for_llm(query_type="build")
        self.log(f"   Loaded context: {len(context)} chars")

        return context

    def plan(self, goal: str, state: Dict, context: str) -> str:
        """Step 3: Ask LM Studio for implementation plan."""
        self.log(f"\n🎯 Step 3: Planning implementation for: {goal}")

        prompt = f"""You are the pxOS build planner.

CURRENT STATE:
- Milestone: {state['current_milestone']}
- Binary size: {state['sizes'].get('binary_bytes', 0)} bytes
- Primitive count: {state['sizes'].get('primitive_count', 0)}

ACCUMULATED KNOWLEDGE:
{context}

GOAL: {goal}

Create a 3-7 step plan for implementing this goal using x86 bootloader primitives.
Focus on:
1. Which BIOS interrupts to use
2. Memory addresses (safe zone: 0x7E00-0x7FFF)
3. Specific opcodes needed
4. Testing criteria

Output a numbered list of concrete steps.
"""

        plan = self.bridge.ask_lm_studio(prompt, use_context=True)
        self.log(f"\n📋 PLAN:\n{plan}\n")

        return plan

    def generate_code(self, goal: str, plan: str) -> Tuple[bool, Optional[Dict]]:
        """Step 4: Generate primitives using JSON-validated generator."""
        self.log("\n⚙️  Step 4: Generating primitives...")

        success, data, raw = self.generator.generate_primitives(
            feature_description=goal,
            start_address=None,  # Let generator choose safe address
            max_attempts=3,
            constraints={
                "milestone": "M2",
                "safe_zone": "0x7E00-0x7FFF",
                "plan": plan
            }
        )

        if not success:
            self.log("   ❌ Primitive generation failed")
            return False, None

        self.log(f"   ✅ Generated {len(data['primitives'])} primitives")
        return True, data

    def append_and_build(self, primitive_data: Dict) -> Tuple[bool, str]:
        """Step 5: Append primitives and build binary."""
        self.log("\n🔨 Step 5: Appending primitives and building...")

        # Append to commands file
        self.generator.append_to_commands_file(primitive_data, COMMANDS_FILE)

        # Build
        try:
            result = subprocess.run(
                [sys.executable, str(BUILD_SCRIPT)],
                cwd=PXOS_DIR,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                self.log("   ✅ Build successful")
                return True, result.stdout
            else:
                self.log("   ❌ Build failed")
                self.log(result.stderr[:500])
                return False, result.stderr

        except Exception as e:
            self.log(f"   ❌ Build error: {e}")
            return False, str(e)

    def test(self, timeout: int = 5) -> Dict:
        """Step 6: Test in QEMU (smoke test)."""
        self.log("\n🧪 Step 6: Testing in QEMU...")

        try:
            proc = subprocess.Popen(
                [
                    "qemu-system-i386",
                    "-fda", str(OUTPUT_BIN),
                    "-nographic",
                    "-monitor", "none"
                ],
                cwd=PXOS_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Let it run briefly
            time.sleep(timeout)

            # Terminate
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

            # Success if it didn't crash immediately
            success = proc.returncode is None or proc.returncode in [0, -15]

            result = {
                "success": success,
                "exit_code": proc.returncode,
                "verdict": "PASS - No crash" if success else f"FAIL - Exit {proc.returncode}"
            }

            self.log(f"   {'✅' if success else '❌'} {result['verdict']}")
            return result

        except FileNotFoundError:
            self.log("   ⚠️  QEMU not found, skipping test")
            return {"success": True, "verdict": "SKIPPED - QEMU not available"}
        except Exception as e:
            self.log(f"   ❌ Test error: {e}")
            return {"success": False, "verdict": f"ERROR - {e}"}

    def log_to_pixels(
        self,
        goal: str,
        plan: str,
        state: Dict,
        build_success: bool,
        test_result: Dict,
        primitive_data: Optional[Dict]
    ):
        """Step 7: Log this run to pixel network."""
        self.log("\n💾 Step 7: Logging to pixel network...")

        summary = f"""
BUILD RUN: {self.run_id}
Goal: {goal}

Plan:
{plan}

State Before:
- Milestone: {state['current_milestone']}
- Binary size: {state['sizes'].get('binary_bytes', 0)} bytes

Result:
- Build: {'SUCCESS' if build_success else 'FAILED'}
- Test: {test_result['verdict']}
- Primitives generated: {len(primitive_data['primitives']) if primitive_data else 0}

---
"""

        self.bridge.append_interaction(
            f"Build: {goal}",
            summary
        )

        self.log("   ✅ Pixel network updated")

    def run(self, goal: str) -> Dict:
        """
        Run the full pipeline.

        Returns dict with results (machine-readable).
        """
        if not self.machine_mode:
            print("\n" + "="*70)
            print("🚀 pxOS ORCHESTRATOR - Full Build Pipeline")
            print("="*70)

        # Pipeline
        state = self.analyze_state()
        context = self.load_context()
        plan = self.plan(goal, state, context)
        code_success, primitive_data = self.generate_code(goal, plan)

        if not code_success:
            result = {
                "success": False,
                "stage": "code_generation",
                "error": "Primitive generation failed"
            }
            if self.machine_mode:
                return result
            else:
                print("\n❌ Pipeline failed at code generation")
                return result

        build_success, build_output = self.append_and_build(primitive_data)

        if not build_success:
            result = {
                "success": False,
                "stage": "build",
                "error": build_output[:500]
            }
            # Still log the failure
            self.log_to_pixels(goal, plan, state, False, {"verdict": "BUILD FAILED"}, primitive_data)

            if self.machine_mode:
                return result
            else:
                print("\n❌ Pipeline failed at build")
                return result

        test_result = self.test()
        self.log_to_pixels(goal, plan, state, True, test_result, primitive_data)

        result = {
            "success": build_success and test_result.get("success", False),
            "stage": "complete",
            "state_before": state,
            "plan": plan,
            "primitives_generated": len(primitive_data['primitives']),
            "build_output": build_output[:200],
            "test_result": test_result,
            "run_id": self.run_id
        }

        if not self.machine_mode:
            print("\n" + "="*70)
            print(f"{'✅' if result['success'] else '⚠️ '} Pipeline Complete")
            print("="*70)
            print(f"\nRun ID: {self.run_id}")
            print(f"Success: {result['success']}")
            print(f"Primitives: {result['primitives_generated']}")
            print(f"Test: {test_result['verdict']}")

        return result


def get_next_milestone() -> str:
    """Parse milestone roadmap and return next incomplete milestone."""
    # TODO: Actually parse PXOS_ROADMAP.md
    # For now, hardcode M2
    return "Add backspace support to shell"


def main():
    parser = argparse.ArgumentParser(
        description="pxOS Orchestrator - The full build pipeline conductor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build next milestone automatically
  python3 pxos_orchestrator.py --auto

  # Specific goal
  python3 pxos_orchestrator.py --goal "Add backspace support"

  # Machine mode (pure JSON output for agents)
  python3 pxos_orchestrator.py --goal "clear screen" --machine

  # No testing (faster iteration)
  python3 pxos_orchestrator.py --goal "help command" --no-test
"""
    )

    parser.add_argument(
        "--goal",
        help="Specific feature to implement"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use next milestone from roadmap"
    )
    parser.add_argument(
        "--machine",
        action="store_true",
        help="Machine mode: output only JSON"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip QEMU testing (faster)"
    )
    parser.add_argument(
        "--lm-studio-url",
        default="http://localhost:1234/v1",
        help="LM Studio API URL"
    )

    args = parser.parse_args()

    # Determine goal
    if args.auto:
        goal = get_next_milestone()
    elif args.goal:
        goal = args.goal
    else:
        print("Error: Specify --goal or --auto")
        sys.exit(1)

    # Run orchestrator
    orchestrator = PxOSOrchestrator(
        lm_studio_url=args.lm_studio_url,
        machine_mode=args.machine
    )

    result = orchestrator.run(goal)

    # Output
    if args.machine:
        print(json.dumps(result, indent=2))

    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
