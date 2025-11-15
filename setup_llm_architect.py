#!/usr/bin/env python3
"""
setup_llm_architect.py - Quick setup for LM Studio + pxOS integration

This script helps you connect pxOS to your local LLM (LM Studio/Ollama)
and configure it as the system architect.

Usage:
  python3 setup_llm_architect.py --test
  python3 setup_llm_architect.py --setup lmstudio
  python3 setup_llm_architect.py --setup ollama
  python3 setup_llm_architect.py --launch-chat
"""

import argparse
import requests
import json
from pathlib import Path
import subprocess
import sys

class LLMArchitectSetup:
    """Helper for setting up LM Studio / Ollama with pxOS"""

    def __init__(self):
        self.lmstudio_endpoint = "http://localhost:1234/v1/chat/completions"
        self.ollama_endpoint = "http://localhost:11434/v1/chat/completions"

    def test_endpoint(self, endpoint: str, model: str = "test-model") -> bool:
        """Test if an LLM endpoint is responding"""
        print(f"\n🔍 Testing endpoint: {endpoint}")

        try:
            response = requests.post(
                endpoint,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'pxOS online'"}],
                    "max_tokens": 10
                },
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ Endpoint is online!")
                print(f"   Response: {content}")
                return True
            else:
                print(f"❌ Endpoint returned {response.status_code}")
                print(f"   {response.text[:200]}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to {endpoint}")
            print(f"   Is the LLM server running?")
            return False

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def detect_backend(self):
        """Auto-detect which LLM backend is running"""
        print("\n🔍 Auto-detecting LLM backend...")

        # Try LM Studio
        print("\n  Trying LM Studio (port 1234)...")
        if self.test_endpoint(self.lmstudio_endpoint):
            print("  ✅ LM Studio detected")
            return "lmstudio", self.lmstudio_endpoint

        # Try Ollama
        print("\n  Trying Ollama (port 11434)...")
        if self.test_endpoint(self.ollama_endpoint):
            print("  ✅ Ollama detected")
            return "ollama", self.ollama_endpoint

        print("\n❌ No LLM backend detected")
        print("\nPlease start either:")
        print("  - LM Studio with local server on port 1234")
        print("  - Ollama with port 11434")
        return None, None

    def setup_architect(self, backend: str = None, model_name: str = None,
                       endpoint: str = None):
        """Setup LLM as pxOS architect"""
        print("\n" + "="*60)
        print("Setting up LLM Architect for pxOS")
        print("="*60)

        # Auto-detect if not specified
        if backend is None:
            backend, endpoint = self.detect_backend()
            if backend is None:
                return False

        # Use default endpoints if not specified
        if endpoint is None:
            if backend == "lmstudio":
                endpoint = self.lmstudio_endpoint
            elif backend == "ollama":
                endpoint = self.ollama_endpoint

        # Get model name
        if model_name is None:
            model_name = input(f"\nEnter model name for {backend} (e.g., 'qwen2.5-7b-instruct'): ").strip()
            if not model_name:
                print("❌ Model name required")
                return False

        # Test the configuration
        print(f"\n🧪 Testing configuration...")
        print(f"   Backend:  {backend}")
        print(f"   Endpoint: {endpoint}")
        print(f"   Model:    {model_name}")

        if not self.test_endpoint(endpoint, model_name):
            print("\n❌ Configuration test failed")
            return False

        # Create PXDigest cartridge
        print(f"\n📦 Creating PXDigest cartridge...")

        cmd = [
            "python3", "px_digest_model.py", "create", "pxOS_Architect",
            "--backend", backend,
            "--endpoint", endpoint,
            "--model", model_name,
            "--system-prompt",
            "You are the pxOS Architect, an LLM responsible for designing, extending, "
            "and maintaining a pixel-native operating system where everything is encoded as images. "
            "Your job is to propose concrete modules, improvements, and code for pxOS."
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)

            print("\n✅ pxOS Architect configured successfully!")
            print(f"\nYour LLM is now available as 'pxOS_Architect'")
            print(f"You can chat with it using:")
            print(f"  python3 infinite_map_chat.py")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create PXDigest cartridge")
            print(e.stderr)
            return False

    def launch_chat(self):
        """Launch infinite map chat with architect"""
        print("\n🚀 Launching pxOS Infinite Map Chat...")
        print("\nControls:")
        print("  Arrow keys / WASD - Navigate map")
        print("  Enter - Start chat on current tile")
        print("  1-9 - Select LLM model")
        print("  Tab - Show model info")
        print("  Esc - Quit")
        print("\nTip: Navigate to different tiles for different contexts!")
        print("     Each tile has its own conversation history.\n")

        try:
            subprocess.run(["python3", "infinite_map_chat.py"], check=True)
        except KeyboardInterrupt:
            print("\n\n👋 Chat closed")
        except FileNotFoundError:
            print("❌ infinite_map_chat.py not found")
        except Exception as e:
            print(f"❌ Error launching chat: {e}")

    def show_quickstart(self):
        """Show quickstart guide"""
        print("\n" + "="*60)
        print("pxOS + LM Studio Quick Start Guide")
        print("="*60)
        print("""
1️⃣  Start your LLM server:

   LM Studio:
     - Open LM Studio
     - Load a model (e.g., Qwen 2.5 7B Instruct)
     - Go to "Local Server" tab
     - Click "Start Server"
     - Server should be at: http://localhost:1234

   Ollama:
     - Install Ollama
     - Run: ollama serve
     - Server should be at: http://localhost:11434

2️⃣  Setup pxOS Architect:

   python3 setup_llm_architect.py --setup lmstudio
   # or
   python3 setup_llm_architect.py --setup ollama

3️⃣  Launch chat interface:

   python3 setup_llm_architect.py --launch-chat

4️⃣  Talk to your architect:

   Navigate to tile (0,0) and press Enter.

   Example prompts:
   - "What should we build next for pxOS?"
   - "Design a boot sequence for an LLM-first OS"
   - "Write a PXI module that displays PixelFS as a tree"
   - "How should we organize the virtual filesystem?"

5️⃣  Let the LLM build pxOS:

   The LLM can propose code, designs, and improvements.
   Copy its code into files and run:

   python3 python_to_pxi.py module.py module.pxi.png
   python3 pack_file_to_boot_pixel.py add module.pxi.png --type pxi_module

Now you're building an OS with an AI. 🎯
        """)


def main():
    parser = argparse.ArgumentParser(description="Setup LLM Architect for pxOS")
    parser.add_argument('--test', action='store_true',
                       help='Test LLM endpoints')
    parser.add_argument('--setup', choices=['lmstudio', 'ollama'],
                       help='Setup specific backend')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--endpoint', help='Custom endpoint URL')
    parser.add_argument('--launch-chat', action='store_true',
                       help='Launch infinite map chat')
    parser.add_argument('--quickstart', action='store_true',
                       help='Show quickstart guide')

    args = parser.parse_args()

    setup = LLMArchitectSetup()

    if args.quickstart:
        setup.show_quickstart()

    elif args.test:
        print("\n🧪 Testing LLM endpoints...")
        setup.detect_backend()

    elif args.setup:
        setup.setup_architect(
            backend=args.setup,
            model_name=args.model,
            endpoint=args.endpoint
        )

    elif args.launch_chat:
        setup.launch_chat()

    else:
        # Default: show quickstart
        setup.show_quickstart()


if __name__ == "__main__":
    main()
