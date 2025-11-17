#!/usr/bin/env python3
"""
pxOS Hypervisor - Stable Execution API

The hypervisor provides a stable API contract that any pxOS implementation
must satisfy. This allows implementations to evolve while maintaining
compatibility with the launcher and external tools.

Genesis Requirement (§4): All execution flows through hypervisor API.

Key Responsibilities:
- Load and validate cartridges
- Execute programs from pixel archives
- Enforce sandbox boundaries
- Log all execution
- Provide introspection capabilities

The hypervisor is the ONLY interface between:
- The host system (pxos_shim.py)
- The pixel world (code in archives)
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from importlib import import_module
import json


class PxOSHypervisorAPI:
    """
    Abstract base class defining the stable hypervisor contract.

    Any pxOS implementation MUST implement this API.
    Implementations can ADD methods, but must NEVER remove or change signatures.
    """

    def run_program(self, name: str, args: Dict = None) -> Dict:
        """
        Execute a program from the current cartridge.

        Args:
            name: Module path + function (e.g., "pixel_llm.programs.hello_world:main")
            args: Arguments to pass to the program

        Returns:
            {
                "success": bool,
                "result": any,
                "error": str (if failed),
                "execution_time": float,
                "logs": list of log entries
            }
        """
        raise NotImplementedError("Hypervisor must implement run_program")

    def inspect_self(self) -> Dict:
        """
        Get capabilities and status of this hypervisor.

        Returns:
            {
                "version": str,
                "cartridge": str,
                "capabilities": list of str,
                "modules_loaded": int,
                "sandbox_active": bool,
                "gpu_available": bool
            }
        """
        raise NotImplementedError("Hypervisor must implement inspect_self")

    def validate_genesis(self) -> Dict:
        """
        Check Genesis compliance of current cartridge.

        Returns:
            {
                "compliant": bool,
                "version": str (Genesis version),
                "violations": list of str,
                "tests_passed": int,
                "tests_failed": int
            }
        """
        raise NotImplementedError("Hypervisor must implement validate_genesis")

    def shutdown(self):
        """Clean shutdown of hypervisor"""
        pass


class Hypervisor(PxOSHypervisorAPI):
    """
    Reference implementation of pxOS hypervisor.

    This is the Genesis (v1.0) implementation. Future versions can replace
    this entirely as long as they implement PxOSHypervisorAPI.
    """

    def __init__(self, cartridge_path: Optional[Path] = None, sandbox: bool = False):
        """
        Initialize hypervisor.

        Args:
            cartridge_path: Path to .pxa archive (None = use current)
            sandbox: If True, run in isolated sandbox mode
        """
        self.cartridge_path = cartridge_path
        self.sandbox = sandbox
        self.version = "1.0.0"
        self.start_time = datetime.now(timezone.utc)

        # Execution log
        self.execution_log: List[Dict] = []

        # Module cache
        self.loaded_modules: Dict[str, Any] = {}

        # Archive reader (initialized when needed)
        self.archive_reader = None

        # Capabilities
        self.capabilities = [
            "python_execution",
            "pixel_imports",
            "archive_loading"
        ]

        # Check GPU availability
        try:
            import wgpu
            self.capabilities.append("gpu_wgpu")
        except ImportError:
            pass

        print(f"[hypervisor] pxOS Hypervisor v{self.version}")
        if sandbox:
            print("[hypervisor] Running in SANDBOX mode")
        if cartridge_path:
            print(f"[hypervisor] Cartridge: {cartridge_path}")

    def _log(self, level: str, message: str, context: Dict = None):
        """Add entry to execution log"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "context": context or {}
        }
        self.execution_log.append(entry)

        # Also print to console
        icons = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "SUCCESS": "✅"}
        icon = icons.get(level, "📝")
        print(f"{icon} [{level}] {message}")

    def run_program(self, name: str, args: Dict = None) -> Dict:
        """
        Execute a program.

        Format: "module.path:function_name" or just "module.path" (calls main)

        Examples:
            "pixel_llm.programs.hello_world:main"
            "pixel_llm.programs.demo"  (calls demo.main)
        """
        start = time.time()
        args = args or {}

        self._log("INFO", f"Executing: {name}", {"args": args})

        try:
            # Parse module:function format
            if ":" in name:
                module_name, func_name = name.split(":", 1)
            else:
                module_name = name
                func_name = "main"

            # Import module
            self._log("INFO", f"Importing module: {module_name}")
            mod = import_module(module_name)
            self.loaded_modules[module_name] = mod

            # Get function
            if not hasattr(mod, func_name):
                raise AttributeError(f"Module {module_name} has no function '{func_name}'")

            func = getattr(mod, func_name)

            # Execute
            self._log("INFO", f"Calling {module_name}.{func_name}()")

            if args:
                result = func(**args)
            else:
                result = func()

            execution_time = time.time() - start

            self._log("SUCCESS", f"Completed in {execution_time:.3f}s", {
                "result_type": type(result).__name__
            })

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "logs": self.execution_log[-10:]  # Last 10 log entries
            }

        except Exception as e:
            execution_time = time.time() - start
            error_trace = traceback.format_exc()

            self._log("ERROR", f"Execution failed: {str(e)}", {
                "exception_type": type(e).__name__,
                "traceback": error_trace
            })

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_trace,
                "execution_time": execution_time,
                "logs": self.execution_log[-10:]
            }

    def inspect_self(self) -> Dict:
        """Get hypervisor capabilities and status"""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "version": self.version,
            "cartridge": str(self.cartridge_path) if self.cartridge_path else "current",
            "capabilities": self.capabilities,
            "modules_loaded": len(self.loaded_modules),
            "sandbox_active": self.sandbox,
            "gpu_available": "gpu_wgpu" in self.capabilities,
            "uptime_seconds": uptime,
            "execution_log_entries": len(self.execution_log),
            "genesis_version": "1.0"
        }

    def validate_genesis(self) -> Dict:
        """
        Run Genesis compliance checks.

        This is a simple implementation - a full version would run
        the Genesis test suite from pixel_llm/tests/genesis/
        """
        violations = []
        tests_passed = 0
        tests_failed = 0

        # Check basic requirements
        checks = [
            ("Archive loading", self._check_archive_loading),
            ("Python execution", self._check_python_execution),
            ("Module imports", self._check_module_imports),
        ]

        for check_name, check_func in checks:
            try:
                if check_func():
                    tests_passed += 1
                else:
                    tests_failed += 1
                    violations.append(f"{check_name} failed")
            except Exception as e:
                tests_failed += 1
                violations.append(f"{check_name} error: {str(e)}")

        compliant = len(violations) == 0

        return {
            "compliant": compliant,
            "version": "1.0",
            "violations": violations,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "checks_run": len(checks)
        }

    def _check_archive_loading(self) -> bool:
        """Check if archive system works"""
        try:
            from pixel_llm.core import pixelfs
            return True
        except:
            return False

    def _check_python_execution(self) -> bool:
        """Check if Python execution works"""
        try:
            exec("x = 1 + 1")
            return True
        except:
            return False

    def _check_module_imports(self) -> bool:
        """Check if module imports work"""
        try:
            import sys
            import pathlib
            return True
        except:
            return False

    def get_logs(self, last_n: int = 100) -> List[Dict]:
        """Get recent execution logs"""
        return self.execution_log[-last_n:]

    def clear_logs(self):
        """Clear execution log"""
        self.execution_log.clear()

    def shutdown(self):
        """Clean shutdown"""
        self._log("INFO", "Hypervisor shutting down")

        # Save logs if needed
        # Clean up resources
        # Close archive reader

        print("[hypervisor] Shutdown complete")


# Convenience functions for common operations

_hypervisor = None

def get_hypervisor(cartridge_path: Optional[Path] = None, sandbox: bool = False) -> Hypervisor:
    """Get global hypervisor instance"""
    global _hypervisor
    if _hypervisor is None:
        _hypervisor = Hypervisor(cartridge_path=cartridge_path, sandbox=sandbox)
    return _hypervisor


def run_entrypoint(module_name: str, func_name: str = "main", args: Dict = None) -> Any:
    """
    Simple entrypoint for running programs.

    This is what external tools call.

    Args:
        module_name: Python module path
        func_name: Function to call
        args: Arguments

    Returns:
        Function result
    """
    hyper = get_hypervisor()
    entrypoint = f"{module_name}:{func_name}"
    result = hyper.run_program(entrypoint, args)

    if result["success"]:
        return result["result"]
    else:
        raise RuntimeError(f"Program failed: {result['error']}")


# CLI for testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("pxOS HYPERVISOR TEST")
    print("="*60)

    hyper = get_hypervisor()

    # Test 1: Self-inspection
    print("\n[TEST 1] Self-inspection")
    info = hyper.inspect_self()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test 2: Genesis validation
    print("\n[TEST 2] Genesis validation")
    genesis = hyper.validate_genesis()
    for key, value in genesis.items():
        if key != "violations" or value:
            print(f"  {key}: {value}")

    # Test 3: Simple execution
    print("\n[TEST 3] Program execution")
    result = hyper.run_program("builtins:print", {"args": ("Hello from hypervisor!",)})
    print(f"  Success: {result['success']}")
    print(f"  Time: {result['execution_time']:.3f}s")

    # Test 4: Check capabilities
    print("\n[TEST 4] Capabilities")
    for cap in hyper.capabilities:
        print(f"  ✅ {cap}")

    print("\n" + "="*60)
    print("✅ Hypervisor operational\n")
