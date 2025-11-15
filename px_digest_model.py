#!/usr/bin/env python3
"""
PXDigest - LLM Pixel Cartridge System

Turn local LLMs into pixel cartridges that can be swapped, stored, and booted.

Each LLM becomes a 1×1 pixel that encodes:
- Model configuration
- Backend (LM Studio, Ollama, custom)
- Connection parameters

Usage:
    px_digest_model.py create <name> --backend lmstudio --endpoint http://localhost:1234
    px_digest_model.py list
    px_digest_model.py show <name>
"""

import json
import random
import sys
from pathlib import Path
from PIL import Image
from typing import Optional


class PXDigestRegistry:
    """Registry of LLM pixel cartridges"""

    def __init__(self, registry_path: str = "llm_pixel_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load()

    def _load(self) -> dict:
        """Load registry from disk"""
        if not self.registry_path.exists():
            return {}
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def _save(self):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def create_llm_pixel(
        self,
        name: str,
        backend: str,
        endpoint: str,
        model_name: str = None,
        description: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str = None
    ) -> tuple:
        """Create a new LLM pixel cartridge"""

        # Check if name already exists
        for entry in self.registry.values():
            if entry.get("name") == name:
                raise ValueError(f"LLM pixel '{name}' already exists")

        # Generate unique 32-bit ID
        while True:
            pid = random.getrandbits(32)
            if str(pid) not in self.registry:
                break

        # Encode as RGBA
        R = (pid >> 24) & 0xFF
        G = (pid >> 16) & 0xFF
        B = (pid >> 8) & 0xFF
        A = pid & 0xFF

        # Create 1×1 pixel
        img = Image.new("RGBA", (1, 1), (R, G, B, A))
        filename = f"llm_{name.replace(' ', '_').lower()}.pxdigest.png"
        img.save(filename)

        # Store in registry
        self.registry[str(pid)] = {
            "name": name,
            "description": description or f"LLM cartridge: {name}",
            "backend": backend,
            "endpoint": endpoint,
            "model_name": model_name or name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "pixel": [R, G, B, A],
            "id": f"0x{pid:08X}",
            "file": filename
        }

        self._save()

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║          LLM PIXEL CARTRIDGE CREATED                      ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()
        print(f"Name: {name}")
        print(f"Backend: {backend}")
        print(f"Endpoint: {endpoint}")
        print()
        print(f"Pixel Color: RGBA({R}, {G}, {B}, {A})")
        print(f"Hex: #{R:02X}{G:02X}{B:02X}{A:02X}")
        print(f"ID: 0x{pid:08X}")
        print()
        print(f"File: {filename}")
        print()
        print("This pixel is now a bootable LLM cartridge.")
        print("Use it with SYS_LLM by setting R3 to the model ID.")

        return (R, G, B, A)

    def list_models(self):
        """List all registered LLM pixels"""
        if not self.registry:
            print("No LLM pixels registered yet.")
            print("\nCreate one with:")
            print("  px_digest_model.py create <name> --backend lmstudio --endpoint http://localhost:1234")
            return

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║              LLM PIXEL CARTRIDGE REGISTRY                 ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        for i, (pid, entry) in enumerate(self.registry.items(), 1):
            name = entry.get("name", "Unnamed")
            backend = entry.get("backend", "unknown")
            endpoint = entry.get("endpoint", "")

            r, g, b, a = entry.get("pixel", [0, 0, 0, 0])

            print(f"{i}. {name}")
            print(f"   Color: RGBA({r}, {g}, {b}, {a}) → #{r:02X}{g:02X}{b:02X}{a:02X}")
            print(f"   Backend: {backend}")
            print(f"   Endpoint: {endpoint}")
            print(f"   ID: {entry.get('id', 'unknown')}")
            print()

    def show_model(self, name: str):
        """Show details for a specific model"""
        found = None
        for pid, entry in self.registry.items():
            if entry.get("name", "").lower() == name.lower():
                found = entry
                break

        if not found:
            print(f"❌ LLM pixel '{name}' not found")
            print("\nAvailable models:")
            self.list_models()
            return

        r, g, b, a = found.get("pixel", [0, 0, 0, 0])

        print("╔═══════════════════════════════════════════════════════════╗")
        print(f"║  {found.get('name', 'Unnamed'):^57s}  ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()
        print(f"Pixel Color: RGBA({r}, {g}, {b}, {a})")
        print(f"Hex: #{r:02X}{g:02X}{b:02X}{a:02X}")
        print(f"ID: {found.get('id', 'unknown')}")
        print()
        print(f"Backend: {found.get('backend', 'unknown')}")
        print(f"Endpoint: {found.get('endpoint', '')}")
        print(f"Model Name: {found.get('model_name', '')}")
        print()
        print(f"Max Tokens: {found.get('max_tokens', 512)}")
        print(f"Temperature: {found.get('temperature', 0.7)}")
        print()
        print(f"Description: {found.get('description', 'No description')}")
        print()
        if found.get("system_prompt"):
            print("System Prompt:")
            print(f"  {found.get('system_prompt')}")
            print()
        print(f"File: {found.get('file', 'unknown')}")

    def get_model_config(self, model_id: int) -> Optional[dict]:
        """Get model configuration by ID"""
        return self.registry.get(str(model_id))

    def delete_model(self, name: str):
        """Delete a model from registry"""
        found_key = None
        for pid, entry in self.registry.items():
            if entry.get("name", "").lower() == name.lower():
                found_key = pid
                break

        if not found_key:
            print(f"❌ LLM pixel '{name}' not found")
            return

        # Confirm
        print(f"⚠️  Delete LLM pixel '{name}'?")
        response = input("   Type 'yes' to confirm: ")

        if response.lower() != 'yes':
            print("Cancelled.")
            return

        # Delete file if it exists
        filename = self.registry[found_key].get("file")
        if filename:
            filepath = Path(filename)
            if filepath.exists():
                filepath.unlink()
                print(f"✓ Deleted {filename}")

        # Remove from registry
        del self.registry[found_key]
        self._save()

        print(f"✓ LLM pixel '{name}' removed from registry")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PXDigest - LLM Pixel Cartridge System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create LM Studio cartridge
  px_digest_model.py create "TinyLlama" --backend lmstudio \\
      --endpoint http://localhost:1234/v1/chat/completions

  # Create Ollama cartridge
  px_digest_model.py create "Llama3" --backend ollama \\
      --endpoint http://localhost:11434/v1/chat/completions \\
      --model llama3.2:latest

  # List all cartridges
  px_digest_model.py list

  # Show details
  px_digest_model.py show "TinyLlama"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # create
    create_parser = subparsers.add_parser("create", help="Create new LLM pixel")
    create_parser.add_argument("name", help="Name for this LLM")
    create_parser.add_argument("--backend", "-b", required=True,
                               choices=["lmstudio", "ollama", "custom"],
                               help="Backend type")
    create_parser.add_argument("--endpoint", "-e", required=True,
                               help="HTTP endpoint URL")
    create_parser.add_argument("--model", "-m", help="Model name (for backend)")
    create_parser.add_argument("--desc", "-d", help="Description")
    create_parser.add_argument("--max-tokens", type=int, default=512)
    create_parser.add_argument("--temperature", type=float, default=0.7)
    create_parser.add_argument("--system-prompt", "-s", help="System prompt")

    # list
    subparsers.add_parser("list", help="List all LLM pixels")

    # show
    show_parser = subparsers.add_parser("show", help="Show LLM pixel details")
    show_parser.add_argument("name", help="Model name")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete LLM pixel")
    delete_parser.add_argument("name", help="Model name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    registry = PXDigestRegistry()

    if args.command == "create":
        registry.create_llm_pixel(
            name=args.name,
            backend=args.backend,
            endpoint=args.endpoint,
            model_name=args.model,
            description=args.desc or "",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=args.system_prompt
        )

    elif args.command == "list":
        registry.list_models()

    elif args.command == "show":
        registry.show_model(args.name)

    elif args.command == "delete":
        registry.delete_model(args.name)


if __name__ == "__main__":
    main()
