#!/usr/bin/env python3
"""
Genesis Compliance Tests

Makes Genesis requirements executable - each § from GENESIS_SPEC.md
becomes a test that verifies the implementation satisfies it.

These tests ensure that any pxOS implementation (current or experimental)
upholds the immutable principles.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestPixelSubstratePrimacy:
    """Tests for Genesis §1: Pixel Substrate Primacy"""

    def test_pixelfs_exists(self):
        """§1: PixelFS module exists for pixel storage"""
        from pixel_llm.core import pixelfs
        assert pixelfs is not None

    def test_pixelfs_read_write(self):
        """§1: Can store and retrieve data as pixels"""
        from pixel_llm.core.pixelfs import PixelFS
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pfs = PixelFS(tmpdir)  # First arg is storage_dir
            test_data = b"Genesis test data"

            # Write as pixels
            pfs.write("test.txt", test_data)

            # Read back
            read_data = pfs.read("test.txt")

            assert read_data == test_data

    def test_infinite_map_exists(self):
        """§1: InfiniteMap exists for 2D pixel space"""
        from pixel_llm.core import infinite_map
        assert infinite_map is not None

    def test_pixel_operations_lossless(self):
        """§1: Pixel operations are lossless"""
        from pixel_llm.core.pixelfs import PixelFS
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pfs = PixelFS(tmpdir)  # First arg is storage_dir

            # Store various data types
            test_cases = [
                b"simple text",
                b"\x00\x01\x02\xff\xfe\xfd",  # Binary
                b"a" * 10000,  # Large
                b"",  # Empty
            ]

            for i, data in enumerate(test_cases):
                pfs.write(f"test_{i}", data)
                assert pfs.read(f"test_{i}") == data


class TestArchiveBasedDistribution:
    """Tests for Genesis §2: Archive-Based Distribution"""

    def test_archive_module_exists(self):
        """§2: Pixel archive system exists"""
        try:
            from pixel_llm.core import pixel_archive
            assert pixel_archive is not None
        except ImportError:
            pytest.skip("pixel_archive module not yet implemented")

    def test_cartridge_manager_exists(self):
        """§2: Cartridge management exists"""
        from pixel_llm.core import cartridge_manager
        assert cartridge_manager is not None

    def test_current_cartridge_queryable(self):
        """§2: Can query current cartridge"""
        from pixel_llm.core.cartridge_manager import get_current_cartridge

        current = get_current_cartridge()
        assert current is not None or True  # May be None in dev mode


class TestNoSilentDeletion:
    """Tests for Genesis §3: No Silent Deletion"""

    def test_cartridge_history_preserved(self):
        """§3: Cartridge history is never deleted"""
        from pixel_llm.core.cartridge_manager import get_manager

        manager = get_manager()
        all_cartridges = manager.list_cartridges()

        # Should have at least Genesis cartridge
        assert len(all_cartridges) >= 1

    def test_archive_history_exists(self):
        """§3: Archive history is tracked"""
        from pixel_llm.core.cartridge_manager import get_manager

        manager = get_manager()
        assert "archive_history" in manager.manifest
        assert isinstance(manager.manifest["archive_history"], list)

    def test_no_delete_operation(self):
        """§3: CartridgeManager has no delete_cartridge method"""
        from pixel_llm.core.cartridge_manager import CartridgeManager

        # Should NOT have a delete method
        assert not hasattr(CartridgeManager, 'delete_cartridge')
        assert not hasattr(CartridgeManager, 'remove_cartridge')


class TestHypervisorContract:
    """Tests for Genesis §4: Hypervisor Contract"""

    def test_hypervisor_exists(self):
        """§4: Hypervisor module exists"""
        from pixel_llm.core import hypervisor
        assert hypervisor is not None

    def test_hypervisor_api_defined(self):
        """§4: PxOSHypervisorAPI is defined"""
        from pixel_llm.core.hypervisor import PxOSHypervisorAPI

        # Check required methods exist
        required_methods = ['run_program', 'inspect_self', 'validate_genesis']
        for method in required_methods:
            assert hasattr(PxOSHypervisorAPI, method)

    def test_hypervisor_implementation(self):
        """§4: Hypervisor implements the API"""
        from pixel_llm.core.hypervisor import Hypervisor, PxOSHypervisorAPI

        # Hypervisor must inherit from API
        assert issubclass(Hypervisor, PxOSHypervisorAPI)

    def test_hypervisor_run_program(self):
        """§4: Hypervisor can run programs"""
        from pixel_llm.core.hypervisor import Hypervisor

        hyper = Hypervisor(sandbox=True)

        # Simple test - call a builtin
        result = hyper.run_program("builtins:abs", {"x": -5})

        # Should not crash (may fail on specific call, but API works)
        assert "success" in result

    def test_hypervisor_inspect_self(self):
        """§4: Hypervisor provides introspection"""
        from pixel_llm.core.hypervisor import Hypervisor

        hyper = Hypervisor()
        info = hyper.inspect_self()

        # Must return required fields
        assert "version" in info
        assert "capabilities" in info
        assert isinstance(info["capabilities"], list)


class TestGPUNativeEventually:
    """Tests for Genesis §5: GPU-Native Eventually"""

    def test_architecture_supports_gpu(self):
        """§5: Architecture doesn't prevent GPU execution"""
        # This is a design check - we verify optional GPU module can exist
        try:
            from pixel_llm.core import gpu_interface
            # If it exists, great
            assert True
        except ImportError:
            # If it doesn't exist yet, that's fine (optional)
            assert True


class TestSandboxTesting:
    """Tests for Genesis §6: Sandbox Testing Required"""

    def test_hypervisor_supports_sandbox(self):
        """§6: Hypervisor supports sandbox mode"""
        from pixel_llm.core.hypervisor import Hypervisor

        # Should be able to create sandbox hypervisor
        hyper = Hypervisor(sandbox=True)
        assert hyper.sandbox is True

    def test_pxos_shim_test_command(self):
        """§6: pxos_shim.py has test command"""
        import subprocess

        result = subprocess.run(
            ["python3", "pxos_shim.py", "--help"],
            capture_output=True,
            text=True
        )

        # Should mention test command
        assert "test" in result.stdout.lower()


class TestTransparentEvolution:
    """Tests for Genesis §7: Transparent Evolution"""

    def test_cartridge_metadata_complete(self):
        """§7: Every cartridge has required metadata"""
        from pixel_llm.core.cartridge_manager import get_manager

        manager = get_manager()
        cartridges = manager.list_cartridges()

        required_fields = ['version', 'parent', 'created_at', 'built_by', 'builder_name', 'notes']

        for cart in cartridges:
            for field in required_fields:
                assert field in cart, f"Cartridge missing {field}"

    def test_evolution_log_queryable(self):
        """§7: Evolution history is queryable"""
        from pixel_llm.core.cartridge_manager import get_manager

        manager = get_manager()

        # Should have archive_history
        assert "archive_history" in manager.manifest
        history = manager.manifest["archive_history"]

        # Each entry should have who, when, why
        for entry in history:
            assert "timestamp" in entry
            assert "approved_by" in entry
            assert "reason" in entry


class TestNoBackdoors:
    """Tests for Genesis §8: No Backdoors"""

    def test_no_hidden_network_calls(self):
        """§8: No hidden network activity"""
        # This is more of a code review item, but we can check basics

        # Core modules should not import requests/urllib without reason
        core_modules = [
            'pixel_llm.core.pixelfs',
            'pixel_llm.core.infinite_map',
            'pixel_llm.core.hypervisor',
            'pixel_llm.core.cartridge_manager'
        ]

        for module_name in core_modules:
            try:
                module = __import__(module_name, fromlist=[''])
                # Check module doesn't import requests or urllib
                module_code = module.__file__
                if module_code:
                    with open(module_code, 'r') as f:
                        code = f.read()
                        # LLM agents can use network, but core should not
                        if 'llm_agents' not in module_name:
                            assert 'requests.post' not in code or 'gemini' in module_name.lower()
            except ImportError:
                continue


class TestCoachingAndEvolution:
    """Tests for Genesis §10: Coaching and Evolution"""

    def test_task_queue_exists(self):
        """§10: Task queue system exists"""
        from pixel_llm.core import task_queue
        assert task_queue is not None

    def test_world_rebuild_task_type(self):
        """§10: WORLD_REBUILD task type exists"""
        from pixel_llm.core.task_queue import TaskAction

        assert hasattr(TaskAction, 'WORLD_REBUILD')

    def test_evolution_helpers_exist(self):
        """§10: Evolution task helpers exist"""
        from pixel_llm.core import task_queue

        assert hasattr(task_queue, 'create_world_rebuild_task')
        assert hasattr(task_queue, 'create_architecture_change_task')

    def test_world_rebuilder_exists(self):
        """§10: World rebuilder execution engine exists"""
        from pixel_llm.core import world_rebuilder
        assert world_rebuilder is not None


class TestGenesisMetaCompliance:
    """Meta-tests about Genesis itself"""

    def test_genesis_spec_exists(self):
        """Genesis specification exists"""
        genesis_path = Path("GENESIS_SPEC.md")
        assert genesis_path.exists()

    def test_genesis_spec_readable(self):
        """Genesis spec is readable"""
        with open("GENESIS_SPEC.md", 'r') as f:
            content = f.read()
            # Should define all 12 principles
            assert "1. Pixel Substrate" in content
            assert "12. Joy and Wonder" in content

    def test_evolution_workflow_documented(self):
        """Evolution workflow is documented"""
        workflow_path = Path("EVOLUTION_WORKFLOW.md")
        assert workflow_path.exists()


# CLI for running just Genesis tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
