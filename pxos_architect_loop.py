#!/usr/bin/env python3
"""
pxos_architect_loop.py

Autonomous improvement loop for pxOS using LM Studio.

The architect LLM continuously:
  1. Analyzes current pxOS state
  2. Proposes improvements
  3. Implements changes (files, modules, configs)
  4. Tests and validates
  5. Repeats indefinitely

Prerequisites:
  - LM Studio running with OpenAI-compatible server
  - A model loaded (will use default model)
  - pip install requests

Usage:
  python3 pxos_architect_loop.py

  Options:
    --interval SECONDS    Time between iterations (default: 30)
    --max-iterations N    Stop after N iterations (default: infinite)
    --dry-run            Don't actually make changes, just show what would happen
"""

import json
import time
import subprocess
import argparse
from pathlib import Path
import requests
import textwrap
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = None  # LM Studio will use default model

STATE_FILE = Path("architect_state.json")
LOG_FILE = Path("architect_loop.log")
PXOS_ROOT = Path(".").resolve()

# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str, level: str = "INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)

    with LOG_FILE.open("a") as f:
        f.write(line + "\n")

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def load_state():
    """Load architect state from disk"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())

    return {
        "iteration": 0,
        "history": [],
        "created_files": [],
        "modified_files": [],
        "errors": []
    }

def save_state(state):
    """Save architect state to disk"""
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ============================================================================
# LLM COMMUNICATION
# ============================================================================

def call_lmstudio(messages, temperature=0.3, max_tokens=2048):
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
    except Exception as e:
        log(f"Error calling LM Studio: {e}", "ERROR")
        raise

# ============================================================================
# INSTRUCTION PARSING
# ============================================================================

def parse_instruction(raw: str) -> dict:
    """Parse JSON instruction from LLM response"""
    try:
        # Strip markdown code blocks if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove ```json and ``` markers
            lines = cleaned.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        return json.loads(cleaned)

    except Exception as e:
        log(f"Failed to parse JSON: {e}", "ERROR")
        log(f"Raw response:\n{raw}", "ERROR")

        return {
            "action": "plan_only",
            "path": "",
            "content": "",
            "command": "",
            "note": f"Failed to parse JSON: {e}"
        }

# ============================================================================
# ACTION EXECUTION
# ============================================================================

def execute_action(instr: dict, dry_run: bool = False) -> str:
    """
    Execute an instruction from the architect

    Instruction schema:
    {
        "action": "write_file" | "append_file" | "run_command" | "compile_to_pxi" | "add_to_pixelfs" | "plan_only",
        "path": "relative/path.ext",
        "content": "file content",
        "command": "shell command",
        "note": "explanation"
    }
    """
    action = instr.get("action")
    note = instr.get("note", "")

    if note:
        log(f"Architect note: {note}")

    if dry_run:
        log(f"[DRY RUN] Would execute: {action}", "INFO")
        return "dry_run"

    # Plan only (no action)
    if action == "plan_only":
        log("Architect chose to plan only, no changes made")
        return "plan_only"

    # Write file
    if action == "write_file":
        path = instr.get("path")
        content = instr.get("content", "")

        if not path:
            log("Error: write_file without path", "ERROR")
            return "error"

        target = PXOS_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

        log(f"Wrote file: {target}")
        return "write_file"

    # Append to file
    if action == "append_file":
        path = instr.get("path")
        content = instr.get("content", "")

        if not path:
            log("Error: append_file without path", "ERROR")
            return "error"

        target = PXOS_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("a") as f:
            f.write("\n" + content)

        log(f"Appended to file: {target}")
        return "append_file"

    # Run command
    if action == "run_command":
        command = instr.get("command")

        if not command:
            log("Error: run_command without command", "ERROR")
            return "error"

        log(f"Running command: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(PXOS_ROOT),
                capture_output=True,
                text=True,
                timeout=60
            )

            log(f"Command exit code: {result.returncode}")

            if result.stdout:
                log(f"STDOUT: {result.stdout[:500]}")

            if result.stderr:
                log(f"STDERR: {result.stderr[:500]}", "WARN")

            return f"run_command:{result.returncode}"

        except subprocess.TimeoutExpired:
            log("Command timed out", "ERROR")
            return "error:timeout"

        except Exception as e:
            log(f"Error running command: {e}", "ERROR")
            return "error"

    # Compile Python to PXI
    if action == "compile_to_pxi":
        source = instr.get("path")

        if not source:
            log("Error: compile_to_pxi without source path", "ERROR")
            return "error"

        output = source.replace(".py", ".pxi.png")

        cmd = f"python3 python_to_pxi.py {source} {output}"
        log(f"Compiling: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(PXOS_ROOT),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                log(f"Successfully compiled to {output}")
                return "compile_to_pxi:success"
            else:
                log(f"Compilation failed: {result.stderr}", "ERROR")
                return "compile_to_pxi:failed"

        except Exception as e:
            log(f"Error compiling: {e}", "ERROR")
            return "error"

    # Add to PixelFS
    if action == "add_to_pixelfs":
        file_path = instr.get("path")
        pixelfs_path = instr.get("pixelfs_path", f"/apps/{Path(file_path).stem}")
        file_type = instr.get("file_type", "py")

        if not file_path:
            log("Error: add_to_pixelfs without file path", "ERROR")
            return "error"

        # First pack as sub-boot pixel
        cmd1 = f"python3 pack_file_to_boot_pixel.py add {file_path} --type {file_type}"
        log(f"Packing: {cmd1}")

        result1 = subprocess.run(cmd1, shell=True, cwd=str(PXOS_ROOT), capture_output=True, text=True)

        if result1.returncode != 0:
            log(f"Failed to pack file: {result1.stderr}", "ERROR")
            return "error"

        # Then add to PixelFS
        filepx_name = f"{Path(file_path).stem}.filepx.png"
        cmd2 = f"python3 pixelfs_builder.py add {pixelfs_path} {filepx_name} --type {file_type}"
        log(f"Adding to PixelFS: {cmd2}")

        result2 = subprocess.run(cmd2, shell=True, cwd=str(PXOS_ROOT), capture_output=True, text=True)

        if result2.returncode == 0:
            log(f"Successfully added to PixelFS: {pixelfs_path}")
            return "add_to_pixelfs:success"
        else:
            log(f"Failed to add to PixelFS: {result2.stderr}", "ERROR")
            return "add_to_pixelfs:failed"

    log(f"Unknown action: {action}", "ERROR")
    return "unknown"

# ============================================================================
# PROMPT GENERATION
# ============================================================================

def build_system_prompt(state: dict) -> str:
    """Build system prompt for the architect"""

    recent_history = state.get("history", [])[-5:]
    history_summary = "\n".join(
        f"  Iteration {h['iteration']}: {h['summary']}"
        for h in recent_history
    ) or "  (no previous iterations)"

    recent_errors = state.get("errors", [])[-3:]
    errors_summary = "\n".join(
        f"  - {e}"
        for e in recent_errors
    ) or "  (no recent errors)"

    return textwrap.dedent(f"""
    You are the pxOS Architect, running in an autonomous improvement loop.

    Your mission: Continuously improve pxOS, a pixel-based operating system where:
    - Programs are stored as PNG images (pixels = instructions)
    - Everything compresses into "God Pixels" (one pixel = entire program)
    - LLMs are first-class citizens (PXDigest cartridges)
    - You run in a hypervisor on the pixel substrate

    Current pxOS features:
    - PXICPU: Pixel instruction set architecture
    - God Pixel compression (16,384:1 ratio)
    - PixelFS: Virtual filesystem
    - Boot sequence: LLM-first design
    - PXDigest: LLMs as pixel cartridges
    - Pixel Hypervisor: Execute guest code
    - Infinite map chat: Spatial conversations

    Your constraints:
    - Respond ONLY with valid JSON (no prose before/after)
    - Make ONE incremental improvement per iteration
    - Prefer small, testable changes
    - Use action="plan_only" if you need to think

    JSON schema (YOU MUST FOLLOW THIS EXACTLY):
    {{
      "action": "write_file" | "append_file" | "run_command" | "compile_to_pxi" | "add_to_pixelfs" | "plan_only",
      "path": "relative/file/path.ext",
      "content": "file content here (for write/append)",
      "command": "shell command (for run_command)",
      "file_type": "py|pxi_module|config|data (for add_to_pixelfs)",
      "pixelfs_path": "/apps/module_name (for add_to_pixelfs)",
      "note": "brief explanation of what and why"
    }}

    Available actions:
    - write_file: Create/overwrite a file
    - append_file: Add content to existing file
    - run_command: Run shell command (tests, builds, etc.)
    - compile_to_pxi: Compile Python to PXI module
    - add_to_pixelfs: Pack file and add to PixelFS
    - plan_only: Just think, no changes

    Recent history:
    {history_summary}

    Recent errors:
    {errors_summary}

    Focus areas for improvement:
    - Boot modules (/boot/02_policy, /boot/03_llm_plane, etc.)
    - Guest Python modules for hypervisor
    - PixelFS organization and tools
    - LLM integration improvements
    - System health monitoring
    - Documentation and examples

    Remember: Small, safe, incremental changes. Quality over quantity.
    """)

def get_next_instruction(state: dict) -> dict:
    """Get next instruction from architect LLM"""

    system_prompt = build_system_prompt(state)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What should we improve next? Respond with JSON only."}
    ]

    raw = call_lmstudio(messages)
    return parse_instruction(raw)

# ============================================================================
# MAIN LOOP
# ============================================================================

def architect_loop(interval: int = 30, max_iterations: int = None, dry_run: bool = False):
    """Main architect loop"""

    log("="*60)
    log("pxOS Architect Autonomous Loop")
    log("="*60)
    log(f"Interval: {interval} seconds")
    log(f"Max iterations: {max_iterations or 'infinite'}")
    log(f"Dry run: {dry_run}")
    log("="*60)

    state = load_state()

    try:
        while True:
            state["iteration"] += 1
            iteration = state["iteration"]

            # Check max iterations
            if max_iterations and iteration > max_iterations:
                log(f"Reached max iterations ({max_iterations}), stopping")
                break

            log("")
            log("="*60)
            log(f"Iteration {iteration}")
            log("="*60)

            try:
                # Get instruction from architect
                log("Consulting architect...")
                instruction = get_next_instruction(state)

                log(f"Architect proposed: {instruction.get('action')}")
                log(f"Note: {instruction.get('note', 'N/A')}")

                # Execute instruction
                result = execute_action(instruction, dry_run=dry_run)

                # Record in history
                summary = instruction.get("note", f"Action: {instruction.get('action')}")
                state["history"].append({
                    "iteration": iteration,
                    "action": instruction.get("action"),
                    "result": result,
                    "summary": summary[:200]
                })

                # Track file changes
                if instruction.get("action") == "write_file":
                    path = instruction.get("path")
                    if path and path not in state.get("created_files", []):
                        state.setdefault("created_files", []).append(path)

                # Save state
                save_state(state)

                log(f"Result: {result}")
                log(f"Iteration {iteration} complete")

            except Exception as e:
                log(f"Error in iteration {iteration}: {e}", "ERROR")
                state.setdefault("errors", []).append(f"Iteration {iteration}: {str(e)}")
                save_state(state)

            # Sleep before next iteration
            if max_iterations is None or iteration < max_iterations:
                log(f"Sleeping {interval} seconds...")
                time.sleep(interval)

    except KeyboardInterrupt:
        log("")
        log("="*60)
        log("Received KeyboardInterrupt, stopping architect loop")
        log(f"Completed {state['iteration']} iterations")
        log("="*60)

    finally:
        save_state(state)
        log(f"State saved to {STATE_FILE}")
        log(f"Log saved to {LOG_FILE}")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="pxOS Architect Autonomous Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between iterations (default: 30)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after N iterations (default: infinite)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually make changes, just show what would happen"
    )

    args = parser.parse_args()

    architect_loop(
        interval=args.interval,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
