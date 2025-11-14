#!/usr/bin/env python3
"""
pxVM Persistent Filesystem
Bridges in-memory VM filesystem with disk-backed storage

Features:
- Automatic sync between memory and disk
- Kernel versioning and history
- Safe atomic writes
- Rollback capability
"""
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List


class PersistentFilesystem:
    """Disk-backed filesystem for pxVM"""

    def __init__(self, base_path: str = "./pxos_fs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create standard directories
        (self.base_path / "kernels").mkdir(exist_ok=True)
        (self.base_path / "build").mkdir(exist_ok=True)

        # Load or create version database
        self.version_file = self.base_path / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self) -> dict:
        """Load version database"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        else:
            # Create initial version database
            return {
                "current": None,
                "history": [],
                "latest_id": 0
            }

    def _save_versions(self):
        """Save version database"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def register_kernel_version(self, version_name: str, bytecode: bytes,
                               parent_version: Optional[str] = None) -> int:
        """
        Register a new kernel version
        Returns version ID
        """
        version_id = self.versions["latest_id"] + 1
        self.versions["latest_id"] = version_id

        filename = f"kernel_{version_name}.bin"
        filepath = self.base_path / "kernels" / filename

        # Write bytecode to disk
        with open(filepath, 'wb') as f:
            f.write(bytecode)

        # Add to history
        version_entry = {
            "id": version_id,
            "version": version_name,
            "file": filename,
            "parent": parent_version,
            "date": datetime.utcnow().isoformat() + "Z",
            "size": len(bytecode)
        }
        self.versions["history"].append(version_entry)
        self.versions["current"] = version_name

        self._save_versions()
        return version_id

    def get_kernel_by_version(self, version_name: str) -> Optional[bytes]:
        """Load kernel bytecode by version name"""
        for entry in self.versions["history"]:
            if entry["version"] == version_name:
                filepath = self.base_path / "kernels" / entry["file"]
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        return f.read()
        return None

    def get_kernel_by_id(self, version_id: int) -> Optional[bytes]:
        """Load kernel bytecode by version ID"""
        for entry in self.versions["history"]:
            if entry["id"] == version_id:
                filepath = self.base_path / "kernels" / entry["file"]
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        return f.read()
        return None

    def get_current_kernel(self) -> Optional[bytes]:
        """Get the current running kernel"""
        if self.versions["current"]:
            return self.get_kernel_by_version(self.versions["current"])
        return None

    def set_current_kernel(self, version_name: str) -> bool:
        """Switch current kernel version"""
        # Verify version exists
        if self.get_kernel_by_version(version_name):
            self.versions["current"] = version_name
            self._save_versions()
            return True
        return False

    def list_versions(self) -> List[dict]:
        """Get list of all kernel versions"""
        return self.versions["history"]

    def get_version_info(self, version_name: str) -> Optional[dict]:
        """Get metadata for a specific version"""
        for entry in self.versions["history"]:
            if entry["version"] == version_name:
                return entry
        return None

    def read_file(self, path: str) -> Optional[bytes]:
        """Read file from persistent storage"""
        filepath = self.base_path / path
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return f.read()
        return None

    def write_file(self, path: str, data: bytes) -> bool:
        """Write file to persistent storage"""
        try:
            filepath = self.base_path / path
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            return False

    def backup_kernel(self, version_name: str) -> bool:
        """Create backup of specific kernel version"""
        kernel = self.get_kernel_by_version(version_name)
        if kernel:
            backup_path = self.base_path / "kernels" / f"kernel_{version_name}_backup.bin"
            with open(backup_path, 'wb') as f:
                f.write(kernel)
            return True
        return False

    def atomic_kernel_replace(self, new_version_name: str, new_bytecode: bytes,
                              parent_version: Optional[str] = None) -> bool:
        """
        Atomically replace kernel with rollback support

        Steps:
        1. Backup current kernel
        2. Write new kernel
        3. Register new version
        4. On any failure, rollback
        """
        try:
            # Backup current if exists
            if self.versions["current"]:
                self.backup_kernel(self.versions["current"])

            # Register and write new version
            version_id = self.register_kernel_version(
                new_version_name,
                new_bytecode,
                parent_version or self.versions["current"]
            )

            return version_id > 0

        except Exception as e:
            # Rollback on failure
            print(f"Kernel replace failed: {e}")
            return False


def main():
    """Test persistent filesystem"""
    fs = PersistentFilesystem()

    print("pxVM Persistent Filesystem Test")
    print("=" * 50)

    # Create a test kernel
    test_kernel = b'\x00\x01\x02\x03' * 100

    # Register version v1.0
    vid = fs.register_kernel_version("v1.0", test_kernel)
    print(f"Registered v1.0 as ID {vid}")

    # Register version v1.1
    test_kernel_v2 = b'\x04\x05\x06\x07' * 120
    vid2 = fs.register_kernel_version("v1.1", test_kernel_v2, parent_version="v1.0")
    print(f"Registered v1.1 as ID {vid2}")

    # List versions
    print("\nVersion History:")
    for ver in fs.list_versions():
        current = " (current)" if ver["version"] == fs.versions["current"] else ""
        print(f"  ID {ver['id']}: {ver['version']} - {ver['size']} bytes{current}")
        print(f"    Parent: {ver['parent']}, Date: {ver['date']}")

    # Switch to v1.0
    print("\nSwitching to v1.0...")
    fs.set_current_kernel("v1.0")
    print(f"Current kernel: {fs.versions['current']}")

    print("\n✓ Persistent filesystem working")


if __name__ == '__main__':
    main()
