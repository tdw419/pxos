#!/usr/bin/env python3
"""
pxOS Kernel Architect

Autonomous LLM agent that helps develop the pxOS microkernel and GPU runtime.

This is a specialized version of the architect loop focused on:
  - x86-64 assembly kernel development
  - WGSL GPU shader programming
  - Hardware mailbox protocol optimization
  - Performance analysis and improvement
  - Documentation generation

The architect understands:
  - NASM assembly syntax
  - x86-64 architecture (long mode, paging, MMIO)
  - WGSL compute shader programming
  - PCIe, BAR0, and hardware protocols
  - Performance targets (< 1μs latency, < 5% CPU overhead)

Usage:
  python3 pxos_kernel_architect.py --mode interactive
  python3 pxos_kernel_architect.py --mode autonomous --interval 60
"""

import json
import time
import subprocess
import argparse
from pathlib import Path
import requests
import textwrap
from datetime import datetime
from typing import Optional, Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = None  # Use default model

STATE_FILE = Path("kernel_architect_state.json")
LOG_FILE = Path("kernel_architect.log")
PXOS_ROOT = Path(".").resolve()
MICROKERNEL_PATH = PXOS_ROOT / "pxos-v1.0" / "microkernel" / "phase1_poc"
PIXEL_LLM_PATH = PXOS_ROOT / "pxos-v1.0" / "pixel_llm"

# =============================================================================
# LOGGING
# =============================================================================

def log(msg: str, level: str = "INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)

    with LOG_FILE.open("a") as f:
        f.write(line + "\n")

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state() -> Dict[str, Any]:
    """Load architect state"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())

    return {
        "iteration": 0,
        "tasks_completed": [],
        "files_created": [],
        "files_modified": [],
        "build_successes": 0,
        "build_failures": 0,
        "performance_metrics": {},
        "errors": []
    }

def save_state(state: Dict[str, Any]):
    """Save architect state"""
    STATE_FILE.write_text(json.dumps(state, indent=2))

# =============================================================================
# LLM COMMUNICATION
# =============================================================================

def call_lmstudio(messages: list, temperature: float = 0.3, max_tokens: int = 2048) -> Optional[str]:
    """Call LM Studio API"""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if MODEL_NAME:
        payload["model"] = MODEL_NAME

    try:
        r = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        log("Cannot connect to LM Studio. Is it running on http://localhost:1234?", "ERROR")
        return None
    except Exception as e:
        log(f"Error calling LM Studio: {e}", "ERROR")
        return None

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def build_kernel_architect_prompt(state: Dict[str, Any]) -> str:
    """Build system prompt for kernel architect"""

    recent_tasks = state.get("tasks_completed", [])[-5:]
    tasks_summary = "\n".join(
        f"  - {task}"
        for task in recent_tasks
    ) or "  (no completed tasks yet)"

    build_stats = f"Successes: {state['build_successes']}, Failures: {state['build_failures']}"

    return textwrap.dedent(f"""
    You are the pxOS Kernel Architect, an expert in low-level systems programming.

    Your mission: Develop and optimize the pxOS microkernel and GPU runtime.

    pxOS Architecture:
    - GPU-centric OS: GPU is the primary execution engine (ring 0)
    - CPU microkernel: Minimal x86-64 assembly kernel (ring 3)
    - Hardware mailbox: Memory-mapped CPU↔GPU communication
    - Privilege inversion: CPU makes "syscalls" to GPU

    Current Phase 2 Status:
    - ✅ GRUB Multiboot2 bootloader
    - ✅ Long mode (64-bit) transition
    - ✅ PCIe enumeration (finds GPU)
    - ✅ BAR0 memory mapping (MMIO access)
    - ✅ Hardware mailbox protocol (CPU side in ASM)
    - ✅ WGSL GPU runtime skeleton
    - 🚧 GPU runtime implementation (WGSL compute shaders)
    - 🚧 Performance optimization (target: <1μs latency)
    - 🚧 Full syscall protocol

    Your expertise:
    - x86-64 assembly (NASM syntax)
    - Long mode, paging, page tables
    - PCIe, MMIO, BAR registers
    - WGSL compute shader programming
    - Hardware synchronization (MFENCE, memory ordering)
    - Performance optimization

    File locations:
    - Microkernel: pxos-v1.0/microkernel/phase1_poc/
    - GPU kernels: pxos-v1.0/pixel_llm/gpu_kernels/
    - Documentation: pxos-v1.0/microkernel/phase1_poc/*.md

    Recent tasks completed:
    {tasks_summary}

    Build statistics: {build_stats}

    Focus areas:
    1. GPU runtime implementation (WGSL)
    2. Mailbox performance optimization
    3. Additional opcodes (memory management, syscalls)
    4. Testing and validation
    5. Documentation improvements

    Response format (JSON only):
    {{
      "task": "Brief task description",
      "action": "write_asm" | "write_wgsl" | "write_doc" | "run_build" | "run_test" | "analyze" | "plan",
      "file_path": "path/to/file.ext",
      "content": "file content (for write actions)",
      "command": "shell command (for run actions)",
      "rationale": "Why this task is important"
    }}

    Guidelines:
    - Make ONE focused improvement per iteration
    - Prefer small, testable changes
    - Always explain your reasoning
    - Consider performance implications
    - Maintain code quality and documentation
    - Test builds after significant changes

    Remember: pxOS inverts the traditional privilege model. The GPU is king.
    """)

# =============================================================================
# INSTRUCTION PARSING
# =============================================================================

def parse_instruction(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON instruction from LLM response"""
    try:
        # Strip markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        return json.loads(cleaned)

    except Exception as e:
        log(f"Failed to parse JSON: {e}", "ERROR")
        log(f"Raw response:\n{raw[:500]}", "ERROR")
        return None

# =============================================================================
# ACTION EXECUTION
# =============================================================================

def execute_action(instr: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Execute an instruction from the architect"""

    action = instr.get("action")
    task = instr.get("task", "Unknown task")
    rationale = instr.get("rationale", "")

    log(f"Task: {task}")
    if rationale:
        log(f"Rationale: {rationale}")

    if action == "plan":
        log("Architect is planning, no action taken")
        return True

    if action == "analyze":
        log("Architect is analyzing, no action taken")
        return True

    if action in ["write_asm", "write_wgsl", "write_doc"]:
        file_path = instr.get("file_path")
        content = instr.get("content", "")

        if not file_path:
            log("Error: write action without file_path", "ERROR")
            return False

        target = PXOS_ROOT / file_path
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            target.write_text(content)
            log(f"✓ Wrote file: {target}")

            # Track file creation/modification
            if file_path not in state["files_created"]:
                state["files_created"].append(file_path)
            if file_path not in state["files_modified"]:
                state["files_modified"].append(file_path)

            return True

        except Exception as e:
            log(f"Error writing file: {e}", "ERROR")
            return False

    if action == "run_build":
        command = instr.get("command", "./test_grub_multiboot.sh")
        log(f"Running build: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(MICROKERNEL_PATH),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                log("✓ Build succeeded")
                state["build_successes"] += 1
                return True
            else:
                log(f"✗ Build failed (exit {result.returncode})", "ERROR")
                if result.stderr:
                    log(f"STDERR: {result.stderr[:500]}", "ERROR")
                state["build_failures"] += 1
                return False

        except Exception as e:
            log(f"Error running build: {e}", "ERROR")
            state["build_failures"] += 1
            return False

    if action == "run_test":
        command = instr.get("command")
        if not command:
            log("Error: run_test without command", "ERROR")
            return False

        log(f"Running test: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(PXOS_ROOT),
                capture_output=True,
                text=True,
                timeout=120
            )

            log(f"Test exit code: {result.returncode}")
            if result.stdout:
                log(f"STDOUT: {result.stdout[:1000]}")

            return result.returncode == 0

        except Exception as e:
            log(f"Error running test: {e}", "ERROR")
            return False

    log(f"Unknown action: {action}", "ERROR")
    return False

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode():
    """Interactive mode - user provides prompts"""

    log("="*60)
    log("pxOS Kernel Architect - Interactive Mode")
    log("="*60)
    log("Type 'quit' to exit")
    log("="*60)

    state = load_state()

    while True:
        print()
        user_input = input("What should I work on? > ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            log("Exiting interactive mode")
            break

        if not user_input:
            continue

        # Build prompt
        system_prompt = build_kernel_architect_prompt(state)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # Get response
        log("Consulting architect...")
        raw = call_lmstudio(messages)

        if not raw:
            log("Failed to get response from LM Studio", "ERROR")
            continue

        # Parse instruction
        instr = parse_instruction(raw)
        if not instr:
            continue

        # Execute
        state["iteration"] += 1
        success = execute_action(instr, state)

        if success:
            task = instr.get("task", "Task completed")
            state["tasks_completed"].append(task)
            log(f"✓ Task completed: {task}")
        else:
            error = f"Failed task: {instr.get('task', 'unknown')}"
            state["errors"].append(error)
            log(f"✗ {error}", "ERROR")

        # Save state
        save_state(state)

# =============================================================================
# AUTONOMOUS MODE
# =============================================================================

def autonomous_mode(interval: int = 60, max_iterations: Optional[int] = None):
    """Autonomous mode - architect works independently"""

    log("="*60)
    log("pxOS Kernel Architect - Autonomous Mode")
    log("="*60)
    log(f"Interval: {interval} seconds")
    log(f"Max iterations: {max_iterations or 'infinite'}")
    log("="*60)

    state = load_state()

    try:
        while True:
            state["iteration"] += 1
            iteration = state["iteration"]

            if max_iterations and iteration > max_iterations:
                log(f"Reached max iterations ({max_iterations})")
                break

            log("")
            log("="*60)
            log(f"Iteration {iteration}")
            log("="*60)

            # Build prompt
            system_prompt = build_kernel_architect_prompt(state)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What should we improve next? Focus on highest-impact work."}
            ]

            # Get instruction
            log("Consulting architect...")
            raw = call_lmstudio(messages)

            if not raw:
                log("Failed to get response, retrying next iteration", "ERROR")
                time.sleep(interval)
                continue

            # Parse and execute
            instr = parse_instruction(raw)
            if instr:
                success = execute_action(instr, state)

                if success:
                    task = instr.get("task", "Task completed")
                    state["tasks_completed"].append(task)
                    log(f"✓ Iteration {iteration} complete: {task}")
                else:
                    error = f"Failed: {instr.get('task', 'unknown')}"
                    state["errors"].append(error)
                    log(f"✗ {error}", "ERROR")

            # Save state
            save_state(state)

            # Sleep
            if max_iterations is None or iteration < max_iterations:
                log(f"Sleeping {interval} seconds...")
                time.sleep(interval)

    except KeyboardInterrupt:
        log("\nReceived interrupt, stopping")
        save_state(state)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="pxOS Kernel Architect - AI-Powered Kernel Development",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["interactive", "autonomous"],
        default="interactive",
        help="Mode: interactive (user prompts) or autonomous (self-directed)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between iterations in autonomous mode (default: 60)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations in autonomous mode (default: infinite)"
    )

    args = parser.parse_args()

    # Check if LM Studio is running
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=5)
        if r.status_code == 200:
            log("✓ LM Studio is running")
        else:
            log("⚠ LM Studio may not be running correctly", "WARN")
    except:
        log("✗ Cannot connect to LM Studio on http://localhost:1234", "ERROR")
        log("  Please start LM Studio and load a model", "ERROR")
        return

    # Run selected mode
    if args.mode == "interactive":
        interactive_mode()
    else:
        autonomous_mode(args.interval, args.max_iterations)

if __name__ == "__main__":
    main()
