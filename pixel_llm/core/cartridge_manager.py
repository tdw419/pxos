#!/usr/bin/env python3
"""
Cartridge Version Management for pxOS

Manages versioned pixel archives (cartridges) to enable safe evolution:
- Track current and historical cartridges
- Register new implementations
- Promote tested cartridges to "current"
- Maintain evolution history
- Support rollback to any previous version

Key Concepts:
- **Cartridge**: A complete pxOS implementation in a .pxa archive
- **Current**: The cartridge that boots by default
- **Experiment**: A proposed new cartridge being tested
- **Generation**: Lineage number (Genesis = 1, children = 2, etc.)
- **Promotion**: Moving an experiment to "current" status
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import shutil


class CartridgeManager:
    """
    Manages the lifecycle of pxOS cartridge versions.

    Ensures:
    - Only one "current" cartridge at a time
    - All history is preserved (never delete old versions)
    - Changes are auditable (who, when, why)
    - Rollback is always possible
    """

    def __init__(self, manifest_path: str = "pixel_llm/meta/cartridges.json"):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load cartridge manifest from disk"""
        if not self.manifest_path.exists():
            # Create default manifest
            return {
                "current": None,
                "cartridges": {},
                "archive_history": [],
                "experiments": {},
                "metadata": {
                    "spec_version": "1.0",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "guardians": []
                }
            }

        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def _save_manifest(self):
        """Save manifest to disk"""
        self.manifest["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()

        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def get_current_cartridge(self) -> Optional[str]:
        """
        Get the name of the currently active cartridge.

        Returns:
            Cartridge filename (e.g., "pxos_v1_2_0.pxa") or None
        """
        return self.manifest.get("current")

    def get_cartridge_info(self, name: str) -> Optional[Dict]:
        """
        Get detailed information about a cartridge.

        Args:
            name: Cartridge filename

        Returns:
            Cartridge metadata dict or None if not found
        """
        return self.manifest["cartridges"].get(name)

    def list_cartridges(self, status: Optional[str] = None) -> List[Dict]:
        """
        List all cartridges, optionally filtered by status.

        Args:
            status: Filter by status ("current", "historical", "experimental")

        Returns:
            List of cartridge info dicts
        """
        cartridges = []

        for name, info in self.manifest["cartridges"].items():
            if status is None or info.get("status") == status:
                cartridges.append({"name": name, **info})

        # Sort by generation (newest first)
        cartridges.sort(key=lambda c: c.get("generation", 0), reverse=True)

        return cartridges

    def register_cartridge(
        self,
        name: str,
        version: str,
        parent: Optional[str],
        built_by: str,
        builder_name: str,
        notes: str,
        capabilities: List[str] = None,
        metrics: Dict = None,
        status: str = "experimental"
    ) -> bool:
        """
        Register a new cartridge (doesn't make it current).

        Args:
            name: Cartridge filename (e.g., "pxos_v1_2_0.pxa")
            version: Semantic version (e.g., "1.2.0")
            parent: Parent cartridge name (None for Genesis)
            built_by: "human", "llm", "hybrid"
            builder_name: Name of builder (e.g., "tdw419", "pixel_llm_coach")
            notes: Description of changes
            capabilities: List of features
            metrics: Performance/quality metrics
            status: "experimental", "approved", "current", "deprecated"

        Returns:
            True if registered successfully
        """
        if name in self.manifest["cartridges"]:
            print(f"⚠️  Cartridge {name} already exists")
            return False

        # Calculate generation
        generation = 1  # Genesis
        if parent:
            parent_info = self.manifest["cartridges"].get(parent)
            if parent_info:
                generation = parent_info.get("generation", 0) + 1

        # Create cartridge entry
        self.manifest["cartridges"][name] = {
            "version": version,
            "generation": generation,
            "parent": parent,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "built_by": built_by,
            "builder_name": builder_name,
            "genesis_version": "1.0",
            "status": status,
            "notes": notes,
            "metrics": metrics or {},
            "capabilities": capabilities or [],
            "compliance": {
                "genesis_v1": False,  # Must be tested
                "tested": False,
                "approved_by": None,
                "approved_at": None
            }
        }

        # Log to experiments if experimental
        if status == "experimental":
            self.manifest["experiments"][name] = {
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "tests_run": [],
                "test_results": {}
            }

        self._save_manifest()
        print(f"✅ Registered cartridge: {name} (generation {generation})")
        return True

    def promote_cartridge(
        self,
        name: str,
        approved_by: str,
        reason: str,
        force: bool = False
    ) -> bool:
        """
        Promote a cartridge to "current" status.

        This is the critical promotion workflow - requires:
        - Cartridge exists and is tested
        - Genesis compliance verified
        - Human approval (unless force=True)

        Args:
            name: Cartridge to promote
            approved_by: Name of approver
            reason: Reason for promotion
            force: Skip safety checks (dangerous!)

        Returns:
            True if promoted successfully
        """
        cartridge = self.manifest["cartridges"].get(name)

        if not cartridge:
            print(f"❌ Cartridge {name} not found")
            return False

        # Safety checks
        if not force:
            if not cartridge["compliance"]["tested"]:
                print(f"❌ Cartridge {name} has not been tested")
                print("   Run: python pxos_shim.py test --cartridge {name}")
                return False

            if not cartridge["compliance"]["genesis_v1"]:
                print(f"❌ Cartridge {name} does not comply with Genesis v1")
                print("   Fix Genesis compliance issues first")
                return False

        # Record history
        old_current = self.manifest.get("current")

        self.manifest["archive_history"].append({
            "from": old_current,
            "to": name,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "approved_by": approved_by,
            "forced": force
        })

        # Update statuses
        if old_current and old_current in self.manifest["cartridges"]:
            self.manifest["cartridges"][old_current]["status"] = "historical"

        self.manifest["cartridges"][name]["status"] = "current"
        self.manifest["cartridges"][name]["compliance"]["approved_by"] = approved_by
        self.manifest["cartridges"][name]["compliance"]["approved_at"] = \
            datetime.now(timezone.utc).isoformat()

        # Set as current
        self.manifest["current"] = name

        # Remove from experiments if present
        if name in self.manifest["experiments"]:
            del self.manifest["experiments"][name]

        self._save_manifest()

        print(f"✅ Promoted {name} to current")
        if old_current:
            print(f"   Previous: {old_current} (now historical)")

        return True

    def rollback_to(self, name: str, approved_by: str, reason: str) -> bool:
        """
        Rollback to a previous cartridge.

        This is just promote_cartridge() with clear intent.

        Args:
            name: Historical cartridge to restore
            approved_by: Name of approver
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        cartridge = self.manifest["cartridges"].get(name)

        if not cartridge:
            print(f"❌ Cartridge {name} not found")
            return False

        if cartridge["status"] == "current":
            print(f"ℹ️  {name} is already current")
            return True

        print(f"🔄 Rolling back to {name}...")
        return self.promote_cartridge(name, approved_by, f"ROLLBACK: {reason}", force=True)

    def mark_tested(self, name: str, genesis_compliant: bool, test_results: Dict = None):
        """
        Mark a cartridge as tested.

        Args:
            name: Cartridge name
            genesis_compliant: Did it pass Genesis tests?
            test_results: Detailed test results
        """
        cartridge = self.manifest["cartridges"].get(name)

        if not cartridge:
            print(f"❌ Cartridge {name} not found")
            return

        cartridge["compliance"]["tested"] = True
        cartridge["compliance"]["genesis_v1"] = genesis_compliant

        if name in self.manifest["experiments"]:
            self.manifest["experiments"][name]["tests_run"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "genesis_compliant": genesis_compliant,
                "results": test_results or {}
            })

        self._save_manifest()

        icon = "✅" if genesis_compliant else "❌"
        print(f"{icon} Marked {name} as tested (Genesis: {genesis_compliant})")

    def get_lineage(self, name: str) -> List[str]:
        """
        Get the lineage of a cartridge (back to Genesis).

        Args:
            name: Cartridge name

        Returns:
            List of cartridge names from Genesis to this one
        """
        lineage = [name]
        current = name

        while current:
            info = self.manifest["cartridges"].get(current)
            if not info or not info.get("parent"):
                break
            parent = info["parent"]
            lineage.insert(0, parent)
            current = parent

        return lineage

    def print_status(self):
        """Print current status of all cartridges"""
        print("\n" + "="*60)
        print("pxOS CARTRIDGE STATUS")
        print("="*60)

        current = self.get_current_cartridge()
        if current:
            info = self.get_cartridge_info(current)
            print(f"\n🎯 Current: {current}")
            print(f"   Version: {info['version']}")
            print(f"   Generation: {info['generation']}")
            print(f"   Built by: {info['built_by']} ({info['builder_name']})")
            print(f"   Notes: {info['notes']}")
        else:
            print("\n⚠️  No current cartridge set")

        # Show experiments
        experiments = list(self.manifest["experiments"].keys())
        if experiments:
            print(f"\n🧪 Experiments ({len(experiments)}):")
            for exp_name in experiments:
                info = self.get_cartridge_info(exp_name)
                tested = info["compliance"]["tested"]
                compliant = info["compliance"]["genesis_v1"]
                status = "✅ ready" if tested and compliant else "🔬 testing"
                print(f"   {status} {exp_name} (gen {info['generation']})")

        # Show history count
        historical = [c for c in self.manifest["cartridges"].values()
                     if c["status"] == "historical"]
        if historical:
            print(f"\n📚 Historical: {len(historical)} previous versions")

        print("="*60 + "\n")


# Convenience functions for common operations

_manager = None

def get_manager() -> CartridgeManager:
    """Get global cartridge manager instance"""
    global _manager
    if _manager is None:
        _manager = CartridgeManager()
    return _manager


def get_current_cartridge() -> Optional[str]:
    """Get current cartridge name"""
    return get_manager().get_current_cartridge()


def register_cartridge(name: str, **kwargs) -> bool:
    """Register a new cartridge"""
    return get_manager().register_cartridge(name, **kwargs)


def promote_cartridge(name: str, approved_by: str, reason: str) -> bool:
    """Promote cartridge to current"""
    return get_manager().promote_cartridge(name, approved_by, reason)


def rollback_to(name: str, approved_by: str, reason: str) -> bool:
    """Rollback to previous cartridge"""
    return get_manager().rollback_to(name, approved_by, reason)


# CLI for testing
if __name__ == "__main__":
    import sys

    manager = get_manager()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "status":
            manager.print_status()

        elif cmd == "list":
            cartridges = manager.list_cartridges()
            for c in cartridges:
                status_icon = {
                    "current": "🎯",
                    "experimental": "🧪",
                    "historical": "📚",
                    "deprecated": "⚠️"
                }.get(c.get("status", ""), "❓")

                print(f"{status_icon} {c['name']}")
                print(f"   v{c['version']} | gen {c['generation']} | {c['built_by']}")
                print(f"   {c['notes'][:60]}...")
                print()

        elif cmd == "lineage" and len(sys.argv) > 2:
            name = sys.argv[2]
            lineage = manager.get_lineage(name)
            print(f"\nLineage of {name}:")
            for i, ancestor in enumerate(lineage):
                indent = "  " * i
                info = manager.get_cartridge_info(ancestor)
                print(f"{indent}└─ {ancestor} (gen {info['generation']})")

        elif cmd == "register" and len(sys.argv) >= 5:
            # python cartridge_manager.py register NAME VERSION NOTES
            name = sys.argv[2]
            version = sys.argv[3]
            notes = " ".join(sys.argv[4:])

            current = manager.get_current_cartridge()
            manager.register_cartridge(
                name=name,
                version=version,
                parent=current,
                built_by="human",
                builder_name="cli",
                notes=notes
            )

        else:
            print("Unknown command")
            print("Usage:")
            print("  python cartridge_manager.py status")
            print("  python cartridge_manager.py list")
            print("  python cartridge_manager.py lineage <name>")
            print("  python cartridge_manager.py register <name> <version> <notes>")

    else:
        manager.print_status()
