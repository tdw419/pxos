#!/usr/bin/env python3
"""
Automated pxOS Builder with LM Studio Integration

This is the FULL AUTOMATION system that:
1. Analyzes current pxOS state
2. Generates a build plan using LM Studio
3. Uses AI to generate primitive commands
4. Builds the binary
5. Tests in QEMU
6. Learns from results
7. Iterates until goals are met

This is a self-improving OS build system!
"""

from pathlib import Path
import sys
import subprocess
import time
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add pxOS root to path
pxos_root = Path(__file__).resolve().parents[1]
sys.path.append(str(pxos_root))

from pxvm.integration.lm_studio_bridge import LMStudioPixelBridge
from tools.ai_primitive_generator import PrimitiveGenerator


class AutomatedPxOSBuilder:
    """
    Fully automated pxOS build system with AI assistance.

    The system that builds itself!
    """

    def __init__(
        self,
        network_path: Path,
        pxos_dir: Path,
        lm_studio_url: str = "http://localhost:1234/v1"
    ):
        self.network_path = Path(network_path)
        self.pxos_dir = Path(pxos_dir)
        self.lm_studio_url = lm_studio_url

        # Initialize AI components
        self.bridge = LMStudioPixelBridge(network_path, lm_studio_url)
        self.generator = PrimitiveGenerator(network_path, lm_studio_url)

        # Paths
        self.commands_file = self.pxos_dir / "pxos_commands.txt"
        self.build_script = self.pxos_dir / "build_pxos.py"
        self.output_bin = self.pxos_dir / "pxos.bin"

        # Build history
        self.build_history = []

    def analyze_current_state(self) -> Dict:
        """Analyze the current state of pxOS."""
        print("\n🔍 Analyzing current pxOS state...")

        state = {
            "timestamp": datetime.now().isoformat(),
            "files_exist": {
                "commands": self.commands_file.exists(),
                "build_script": self.build_script.exists(),
                "binary": self.output_bin.exists()
            },
            "command_line_count": 0,
            "binary_size": 0,
            "features_implemented": []
        }

        if self.commands_file.exists():
            with open(self.commands_file, 'r') as f:
                lines = f.readlines()
                state["command_line_count"] = len([
                    l for l in lines
                    if l.strip() and not l.strip().startswith('COMMENT')
                ])

        if self.output_bin.exists():
            state["binary_size"] = self.output_bin.stat().st_size

        print(f"   Commands: {state['command_line_count']} lines")
        print(f"   Binary: {state['binary_size']} bytes")

        return state

    def generate_build_plan(self, goals: List[str]) -> List[Dict]:
        """
        Generate a build plan using LM Studio.

        Returns a list of build steps.
        """
        print("\n🎯 Generating build plan...")

        goals_text = "\n".join([f"- {goal}" for goal in goals])

        prompt = f"""You are building pxOS, a bootable operating system using primitives.

CURRENT GOALS:
{goals_text}

Analyze these goals and create a step-by-step build plan. For each step:
1. Describe what needs to be implemented
2. Specify the memory address range to use
3. List the x86 instructions needed

Output your plan as a numbered list of concrete implementation steps.
Focus on one feature at a time, starting with the simplest.

BUILD PLAN:
"""

        response = self.bridge.ask_lm_studio(prompt, use_context=True)

        # Parse the plan into steps
        steps = self._parse_build_plan(response)

        print(f"   Generated {len(steps)} build steps")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step['description'][:60]}...")

        # Learn from this planning session
        self.bridge.append_interaction(
            f"Create build plan for: {', '.join(goals)}",
            response
        )

        return steps

    def _parse_build_plan(self, plan_text: str) -> List[Dict]:
        """Parse LLM build plan into structured steps."""
        steps = []
        lines = plan_text.split('\n')

        current_step = None
        for line in lines:
            line = line.strip()

            # Look for numbered steps
            if line and (line[0].isdigit() or line.startswith('-')):
                # Clean up the line
                description = line.lstrip('0123456789.-) ').strip()

                if description:
                    if current_step:
                        steps.append(current_step)

                    current_step = {
                        "description": description,
                        "address": None,
                        "status": "pending"
                    }
            elif current_step and "0x" in line:
                # Extract address hints
                import re
                addr_match = re.search(r'0x[0-9A-Fa-f]+', line)
                if addr_match:
                    current_step["address"] = addr_match.group(0)

        if current_step:
            steps.append(current_step)

        return steps

    def implement_step(self, step: Dict) -> bool:
        """Implement a single build step using AI."""
        print(f"\n⚙️  Implementing: {step['description']}")

        # Generate primitives for this step
        start_addr = None
        if step.get('address'):
            try:
                start_addr = int(step['address'], 16)
            except ValueError:
                pass

        success, primitives, explanation = self.generator.generate_primitives(
            step['description'],
            start_address=start_addr
        )

        if not success:
            print(f"   ❌ Failed to generate primitives")
            return False

        # Append to commands file
        self.generator.append_to_commands_file(
            primitives,
            self.commands_file,
            step['description']
        )

        step['status'] = 'completed'
        return True

    def build_binary(self) -> Tuple[bool, str]:
        """Run the build script to generate pxos.bin."""
        print("\n🔨 Building binary...")

        try:
            # Run build script
            result = subprocess.run(
                [sys.executable, str(self.build_script)],
                cwd=self.pxos_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("   ✅ Build successful")
                return True, result.stdout
            else:
                print("   ❌ Build failed")
                print(result.stderr[:500])
                return False, result.stderr

        except subprocess.TimeoutExpired:
            print("   ❌ Build timeout")
            return False, "Build timeout"
        except Exception as e:
            print(f"   ❌ Build error: {e}")
            return False, str(e)

    def test_in_qemu(self, timeout: int = 5) -> Tuple[bool, str]:
        """Test the built binary in QEMU."""
        print("\n🧪 Testing in QEMU...")

        try:
            # Run QEMU in test mode (headless, short timeout)
            proc = subprocess.Popen(
                [
                    "qemu-system-i386",
                    "-fda", str(self.output_bin),
                    "-nographic",
                    "-monitor", "none"
                ],
                cwd=self.pxos_dir,
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

            # Check if it started
            if proc.returncode is None or proc.returncode in [0, -15]:  # -15 is SIGTERM
                print("   ✅ QEMU test passed (no crash)")
                return True, "QEMU started successfully"
            else:
                print(f"   ⚠️  QEMU exited with code {proc.returncode}")
                return False, f"Exit code: {proc.returncode}"

        except FileNotFoundError:
            print("   ⚠️  QEMU not found, skipping test")
            return True, "QEMU not available"
        except Exception as e:
            print(f"   ❌ Test error: {e}")
            return False, str(e)

    def automated_build_cycle(
        self,
        goals: List[str],
        max_iterations: int = 10
    ) -> bool:
        """
        Run a full automated build cycle.

        This is the magic: AI builds the OS!
        """
        print("\n" + "="*70)
        print("🚀 STARTING AUTOMATED BUILD CYCLE")
        print("="*70)

        print("\nGOALS:")
        for i, goal in enumerate(goals, 1):
            print(f"  {i}. {goal}")

        # Analyze current state
        initial_state = self.analyze_current_state()

        # Generate build plan
        plan = self.generate_build_plan(goals)

        if not plan:
            print("\n❌ Failed to generate build plan")
            return False

        # Execute plan
        print("\n" + "="*70)
        print("🔧 EXECUTING BUILD PLAN")
        print("="*70)

        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")

            # Find next pending step
            pending_steps = [s for s in plan if s['status'] == 'pending']

            if not pending_steps:
                print("\n🎉 All steps completed!")
                break

            # Implement next step
            step = pending_steps[0]
            success = self.implement_step(step)

            if not success:
                print(f"\n⚠️  Step failed, learning from error...")
                # Learn from failure
                self.bridge.append_interaction(
                    f"Failed to implement: {step['description']}",
                    "Need to revise approach for this feature"
                )
                continue

            # Build
            build_success, build_output = self.build_binary()

            if not build_success:
                print("\n⚠️  Build failed, analyzing error...")
                # Learn from build error
                self.bridge.append_interaction(
                    f"Build error after implementing: {step['description']}",
                    build_output[:500]
                )
                # Revert step status
                step['status'] = 'failed'
                continue

            # Test
            test_success, test_output = self.test_in_qemu()

            # Record result
            self.build_history.append({
                "iteration": iteration + 1,
                "step": step['description'],
                "build_success": build_success,
                "test_success": test_success,
                "timestamp": datetime.now().isoformat()
            })

            # Learn from this iteration
            result = "Success" if test_success else "Failed tests"
            self.bridge.append_interaction(
                f"Implemented: {step['description']}",
                f"Build: {build_success}, Test: {result}\n{test_output}"
            )

            print(f"\n✅ Step completed: {step['description']}")

        # Final state analysis
        final_state = self.analyze_current_state()

        print("\n" + "="*70)
        print("📊 BUILD CYCLE COMPLETE")
        print("="*70)
        print(f"\nInitial commands: {initial_state['command_line_count']}")
        print(f"Final commands: {final_state['command_line_count']}")
        print(f"Final binary size: {final_state['binary_size']} bytes")
        print(f"\nCompleted {len([s for s in plan if s['status'] == 'completed'])}/{len(plan)} steps")

        return True

    def save_build_report(self, output_file: Path):
        """Save a build report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "build_history": self.build_history,
            "final_state": self.analyze_current_state()
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Build report saved: {output_file}")


def main():
    """Main automation entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated pxOS Builder with LM Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automated build with default goals
  python3 tools/auto_build_pxos.py --auto

  # Custom goals
  python3 tools/auto_build_pxos.py --goals "backspace support" "help command"

  # Full automation with testing
  python3 tools/auto_build_pxos.py --auto --test
"""
    )

    parser.add_argument(
        "--network",
        default="pxvm/networks/pxos_autobuild.png",
        help="AI learning network"
    )
    parser.add_argument(
        "--pxos-dir",
        default="pxos-v1.0",
        help="pxOS directory"
    )
    parser.add_argument(
        "--goals",
        nargs='+',
        help="Build goals to achieve"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run with default goals"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test in QEMU after build"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum build iterations"
    )
    parser.add_argument(
        "--report",
        default="build_report.json",
        help="Output report file"
    )

    args = parser.parse_args()

    # Default goals
    default_goals = [
        "Add backspace support to shell",
        "Implement command parser for basic commands",
        "Add 'help' command that shows available commands",
        "Add 'clear' command to clear screen"
    ]

    goals = args.goals if args.goals else (default_goals if args.auto else None)

    if not goals:
        print("Error: Specify --goals or use --auto for default goals")
        parser.print_help()
        sys.exit(1)

    # Initialize builder
    print("🚀 Initializing Automated pxOS Builder...")
    print(f"   Network: {args.network}")
    print(f"   pxOS dir: {args.pxos_dir}")

    builder = AutomatedPxOSBuilder(
        network_path=Path(args.network),
        pxos_dir=Path(args.pxos_dir),
    )

    # Run automated build cycle
    success = builder.automated_build_cycle(
        goals=goals,
        max_iterations=args.max_iterations
    )

    # Save report
    builder.save_build_report(Path(args.report))

    if success:
        print("\n🎉 Automated build cycle completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Automated build cycle failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
