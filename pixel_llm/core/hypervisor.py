#!/usr/bin/env python3
"""
pxOS Hypervisor - Execution Controller Living in Pixels

This module lives INSIDE the pixel archive and controls all code execution.

THE KEY INSIGHT:
  The hypervisor itself is loaded from pixels.
  It then orchestrates execution of other code from pixels.
  The host Python is just a dumb loader.

WHAT THIS DOES:
  • Validates runtime environment (Python version, dependencies)
  • Resolves entrypoints from manifest
  • Sets up execution context (PixelVM, GPU, etc.)
  • Runs code with appropriate isolation/sandboxing
  • Handles errors and logging

ARCHITECTURE:
┌──────────────────────────────────────┐
│  Host Python (tiny shim)             │
└──────────────┬───────────────────────┘
               │ loads
┌──────────────▼───────────────────────┐
│  Hypervisor (from pixels) ← YOU ARE  │
│                             HERE     │
└──────────────┬───────────────────────┘
               │ runs
┌──────────────▼───────────────────────┐
│  Application Code (from pixels)      │
└──────────────────────────────────────┘

Philosophy:
"The hypervisor is part of the cartridge.
 The host is just the CPU.
 Everything else lives in pixels."
"""

import sys
import importlib
import importlib.metadata
from typing import Dict, Any, Optional, Callable
from pathlib import Path


class PixelHypervisor:
    """
    Execution controller for pixel-native code.

    This class is instantiated from the pixel archive and
    controls all subsequent execution.
    """

    def __init__(self, manifest: Dict[str, Any], archive_reader=None):
        self.manifest = manifest
        self.archive_reader = archive_reader
        self.entrypoints = manifest.get("entrypoints", {})
        self.runtime_requirements = manifest.get("python_runtime", {})

    def validate_runtime(self):
        """
        Validate that the host Python environment meets requirements.

        Checks:
          - Python version
          - Required packages
          - Optional packages (warning only)
        """
        print("=" * 70)
        print("pxOS HYPERVISOR: Validating Runtime")
        print("=" * 70)
        print()

        # Check Python version
        min_version = self.runtime_requirements.get("min_version", "3.11")
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        print(f"Python version: {current_version} (minimum: {min_version})")

        if sys.version_info < tuple(map(int, min_version.split("."))):
            raise RuntimeError(
                f"Python {min_version}+ required, but got {current_version}"
            )

        print("✅ Python version OK")
        print()

        # Check required packages
        required = self.runtime_requirements.get("required_packages", {})
        if required:
            print("Checking required packages:")
            for package, spec in required.items():
                try:
                    version = importlib.metadata.version(package)
                    print(f"  ✅ {package} {version} (required: {spec})")
                except importlib.metadata.PackageNotFoundError:
                    raise RuntimeError(
                        f"Required package '{package}' not installed (need {spec})"
                    )
            print()

        # Check optional packages (warnings only)
        optional = self.runtime_requirements.get("optional_packages", {})
        if optional:
            print("Checking optional packages:")
            for package, spec in optional.items():
                try:
                    version = importlib.metadata.version(package)
                    print(f"  ✅ {package} {version}")
                except importlib.metadata.PackageNotFoundError:
                    print(f"  ⚠️  {package} not installed (optional)")
            print()

        print("✅ Runtime validation complete")
        print()

    def list_entrypoints(self):
        """List all available entrypoints"""
        print("=" * 70)
        print("Available Entrypoints")
        print("=" * 70)
        print()

        for name, target in self.entrypoints.items():
            print(f"  {name:12s} → {target}")

        print()

    def resolve_entrypoint(self, name: Optional[str] = None) -> tuple:
        """
        Resolve an entrypoint name to (module_name, function_name).

        Args:
            name: Entrypoint name (e.g., "default", "vm") or full spec (e.g., "module:func")

        Returns:
            (module_name, function_name)
        """
        if name is None:
            name = "default"

        # Check if it's an entrypoint name
        if name in self.entrypoints:
            target = self.entrypoints[name]
        # Or a direct module:func spec
        elif ":" in name:
            target = name
        else:
            raise ValueError(
                f"Unknown entrypoint '{name}'. Available: {list(self.entrypoints.keys())}"
            )

        # Parse module:func
        if ":" not in target:
            raise ValueError(f"Invalid entrypoint target: {target} (expected module:func)")

        module_name, func_name = target.split(":", 1)
        return module_name, func_name

    def run_entrypoint(
        self,
        name: Optional[str] = None,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None
    ):
        """
        Run an entrypoint from the manifest.

        Args:
            name: Entrypoint name or module:func
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
        """
        args = args or []
        kwargs = kwargs or {}

        # Resolve entrypoint
        module_name, func_name = self.resolve_entrypoint(name)

        print("=" * 70)
        print("pxOS HYPERVISOR: Running Entrypoint")
        print("=" * 70)
        print()
        print(f"  Entrypoint: {name or 'default'}")
        print(f"  Target:     {module_name}:{func_name}")
        print()
        print("─" * 70)
        print()

        # Import module (from archive via custom importer)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"❌ Failed to import module '{module_name}'")
            print(f"   Error: {e}")
            print()
            print("   Module should be in the pixel archive.")
            raise

        # Get function
        try:
            func = getattr(module, func_name)
        except AttributeError:
            print(f"❌ Module '{module_name}' has no function '{func_name}'")
            raise

        # Verify it's callable
        if not callable(func):
            raise TypeError(f"{module_name}:{func_name} is not callable")

        print(f"✅ Loaded: {module.__file__}")
        print()

        # Execute
        try:
            result = func(*args, **kwargs)
            print()
            print("─" * 70)
            print("✅ Entrypoint completed successfully")
            print("=" * 70)
            return result

        except Exception as e:
            print()
            print("─" * 70)
            print(f"❌ Entrypoint failed: {e}")
            print("=" * 70)
            raise

    def run_pixelvm_program(self, program_path: str):
        """
        Run a PixelVM bytecode program from the archive.

        Args:
            program_path: Path to .pxi file in archive
        """
        print("=" * 70)
        print("pxOS HYPERVISOR: Running PixelVM Program")
        print("=" * 70)
        print()
        print(f"  Program: {program_path}")
        print()

        # Read program from archive
        if self.archive_reader is None:
            raise RuntimeError("Archive reader not available")

        program_bytes = self.archive_reader.read_file(program_path)

        print(f"  Size: {len(program_bytes)} bytes")
        print()

        # Load PixelVM
        from pixel_llm.core.pixel_vm import PixelVM

        # Execute
        vm = PixelVM(debug=False)
        vm.load_program(program_bytes)

        print("─" * 70)
        print("Executing...")
        print()

        vm.run()

        print()
        print("─" * 70)
        print("✅ PixelVM execution complete")
        print("=" * 70)


# Convenience functions for use by shim

def validate_runtime(manifest: Dict[str, Any]):
    """Validate runtime environment"""
    hypervisor = PixelHypervisor(manifest)
    hypervisor.validate_runtime()


def run_entrypoint(
    module_name: str,
    func_name: str,
    manifest: Dict[str, Any] = None,
    archive_reader=None,
    args: list = None,
    kwargs: dict = None
):
    """
    Run an entrypoint (legacy interface for compatibility).

    Args:
        module_name: Module to import
        func_name: Function to call
        manifest: Manifest dict (optional)
        archive_reader: Archive reader (optional)
        args: Positional arguments
        kwargs: Keyword arguments
    """
    # Build target spec
    target = f"{module_name}:{func_name}"

    # Create hypervisor
    manifest = manifest or {}
    hypervisor = PixelHypervisor(manifest, archive_reader)

    # Run
    return hypervisor.run_entrypoint(target, args=args, kwargs=kwargs)


def create_hypervisor(manifest: Dict[str, Any], archive_reader=None) -> PixelHypervisor:
    """Create a hypervisor instance"""
    return PixelHypervisor(manifest, archive_reader)


if __name__ == "__main__":
    print(__doc__)
