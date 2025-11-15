#!/usr/bin/env python3
"""
run_pxos_lmstudio.py

One-command bootstrap for pxOS + LM Studio.

Prerequisites:
1. LM Studio running with OpenAI-compatible server
2. A model loaded (update MODEL_NAME below)
3. pip install pillow requests pygame

Usage:
    python3 run_pxos_lmstudio.py

What this does:
1. Tests LM Studio connection
2. Creates pxOS_Architect PXDigest model
3. Initializes PixelFS
4. (Optional) Runs boot sequence dry-run
5. Launches infinite map chat

You can then chat with the LLM architect to build pxOS!
"""

import subprocess
import sys
from pathlib import Path
import requests

# ============================================================================
# CONFIGURATION - Edit these to match your setup
# ============================================================================

# LM Studio endpoint
LMSTUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"

# Model name as LM Studio expects it
# Check LM Studio UI for the exact model name
MODEL_NAME = "qwen2.5-7b-instruct"

# Whether to run boot_kernel.py --dry-run before chat
RUN_BOOT_DRY_RUN = True

# ============================================================================


def info(msg: str):
    print(f"[pxOS] {msg}")


def error(msg: str):
    print(f"[pxOS:ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def test_lmstudio():
    """Test if LM Studio is responding"""
    info(f"Testing LM Studio at {LMSTUDIO_ENDPOINT}")
    info(f"Model: {MODEL_NAME}")

    try:
        resp = requests.post(
            LMSTUDIO_ENDPOINT,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "ping from pxOS"}],
                "max_tokens": 8,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data and data["choices"]:
            info("✅ LM Studio is online!")
            return True
        else:
            error("LM Studio response missing 'choices'")
            return False

    except requests.exceptions.ConnectionError:
        error(f"Cannot connect to {LMSTUDIO_ENDPOINT}\n"
              f"   Please start LM Studio and enable the local server")
    except Exception as e:
        error(f"LM Studio test failed: {e}")


def run_cmd(cmd, description, allow_fail=False):
    """Run a subprocess"""
    info(description)
    info(f"$ {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=not allow_fail, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if not allow_fail:
            error(f"Command failed: {e}")
        return False


def check_dependencies():
    """Check required Python packages"""
    required = ['PIL', 'requests', 'pygame']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        error(f"Missing packages: {', '.join(missing)}\n"
              f"   Install with: pip install {' '.join(missing)}")


def setup_architect():
    """Configure pxOS_Architect PXDigest model"""
    if not Path("setup_llm_architect.py").exists():
        error("setup_llm_architect.py not found in current directory")

    cmd = [
        sys.executable,
        "setup_llm_architect.py",
        "--setup",
        "lmstudio",
        "--model",
        MODEL_NAME,
    ]

    run_cmd(cmd, "Configuring pxOS_Architect...")


def setup_pixelfs():
    """Initialize PixelFS and auto-add files"""
    if not Path("pixelfs_builder.py").exists():
        error("pixelfs_builder.py not found")

    # Initialize if needed
    if not Path("pixelfs.json").exists():
        run_cmd(
            [sys.executable, "pixelfs_builder.py", "init"],
            "Initializing PixelFS..."
        )

    # Auto-add files
    run_cmd(
        [sys.executable, "pixelfs_builder.py", "auto-add"],
        "Auto-discovering and adding files to PixelFS...",
        allow_fail=True
    )

    # Show tree
    run_cmd(
        [sys.executable, "pixelfs_builder.py", "tree"],
        "PixelFS tree:",
        allow_fail=True
    )


def boot_dry_run():
    """Run boot sequence dry-run"""
    if not RUN_BOOT_DRY_RUN:
        return

    if not Path("boot_kernel.py").exists():
        info("boot_kernel.py not found, skipping dry-run")
        return

    run_cmd(
        [sys.executable, "boot_kernel.py", "--dry-run"],
        "Running boot sequence (dry-run)...",
        allow_fail=True
    )


def launch_chat():
    """Launch infinite map chat"""
    if not Path("infinite_map_chat.py").exists():
        error("infinite_map_chat.py not found")

    info("")
    info("="*60)
    info("Launching pxOS Infinite Map Chat")
    info("="*60)
    info("")
    info("Controls:")
    info("  Arrow keys / WASD - Navigate map")
    info("  Enter - Start chat on current tile")
    info("  1-9 - Select LLM model (pxOS_Architect)")
    info("  Tab - Show model info")
    info("  Esc - Quit")
    info("")
    info("Navigate to tile (0,0) and press Enter to talk to the Architect!")
    info("")

    try:
        subprocess.run([sys.executable, "infinite_map_chat.py"])
    except KeyboardInterrupt:
        info("\nChat closed")
    except Exception as e:
        error(f"Error launching chat: {e}")


def main():
    print("\n" + "="*60)
    print("pxOS Bootstrap - LM Studio Integration")
    print("="*60 + "\n")

    # Check dependencies
    check_dependencies()

    # Test LM Studio
    test_lmstudio()

    # Setup Architect
    setup_architect()

    # Setup PixelFS
    setup_pixelfs()

    # Optional boot dry-run
    boot_dry_run()

    # Launch chat
    launch_chat()


if __name__ == "__main__":
    main()
