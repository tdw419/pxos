#!/usr/bin/env python3
"""
Unit Tests for Semantic Abstraction Layer
==========================================

Tests the semantic pipeline: Intent → Concepts → Pixels → Code
"""

import unittest
import sys
sys.path.insert(0, '..')

from semantic_layer import (
    SemanticPipeline,
    IntentAnalyzer,
    PixelEncoder,
    X86CodeGenerator,
    ARMCodeGenerator,
    RISCVCodeGenerator,
    SemanticConcept,
    OperationType,
    Scope,
    Duration,
    Atomicity,
    SafetyLevel
)


class TestPixelEncoder(unittest.TestCase):
    """Test pixel encoding and decoding"""

    def setUp(self):
        self.encoder = PixelEncoder()

    def test_encode_critical_section(self):
        """Test encoding of critical section concept"""
        concept = SemanticConcept(
            operation=OperationType.ISOLATION,
            scope=Scope.CPU_LOCAL,
            duration=Duration.TEMPORARY,
            atomicity=Atomicity.ATOMIC,
            safety=SafetyLevel.CRITICAL
        )

        pixel = self.encoder.encode(concept)

        # Should be RGB(255, 32, 255)
        self.assertEqual(pixel[0], 255)  # Red: Isolation + Critical
        self.assertEqual(pixel[1], 32)   # Green: CPU_LOCAL + TEMPORARY
        self.assertEqual(pixel[2], 255)  # Blue: ATOMIC

    def test_encode_memory_allocation(self):
        """Test encoding of memory allocation concept"""
        concept = SemanticConcept(
            operation=OperationType.MEMORY,
            scope=Scope.SYSTEM_WIDE,
            duration=Duration.PERSISTENT,
            atomicity=Atomicity.ATOMIC,
            safety=SafetyLevel.CRITICAL
        )

        pixel = self.encoder.encode(concept)

        self.assertEqual(pixel[0], 128)  # Red: Memory operation
        # Green: SYSTEM_WIDE (224) + PERSISTENT (64) = 288, clamped to 255
        self.assertEqual(pixel[1], 255)  # Clamped to max RGB value
        self.assertEqual(pixel[2], 255)  # Blue: ATOMIC

    def test_encode_sequence(self):
        """Test encoding multiple concepts"""
        concepts = [
            SemanticConcept(
                operation=OperationType.ISOLATION,
                scope=Scope.CPU_LOCAL,
                duration=Duration.TEMPORARY,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL
            ),
            SemanticConcept(
                operation=OperationType.MEMORY,
                scope=Scope.SYSTEM_WIDE,
                duration=Duration.PERSISTENT,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL
            )
        ]

        pixels = self.encoder.encode_sequence(concepts)

        self.assertEqual(len(pixels), 2)
        self.assertIsInstance(pixels[0], tuple)
        self.assertIsInstance(pixels[1], tuple)
        self.assertEqual(len(pixels[0]), 3)  # RGB

    def test_decode_pixel(self):
        """Test decoding pixel back to semantics"""
        pixel = (255, 32, 255)
        decoded = self.encoder.decode(pixel)

        self.assertEqual(decoded['operation'], 'isolation')
        self.assertEqual(decoded['scope'], 'cpu_local')
        self.assertEqual(decoded['atomicity'], 'atomic')
        self.assertIn('rgb', decoded)

    def test_pixel_channels_in_range(self):
        """Test that all pixel channels are valid RGB values (0-255)"""
        concept = SemanticConcept(
            operation=OperationType.ISOLATION,
            scope=Scope.SYSTEM_WIDE,
            duration=Duration.PERSISTENT,
            atomicity=Atomicity.ATOMIC,
            safety=SafetyLevel.CRITICAL
        )

        r, g, b = self.encoder.encode(concept)

        self.assertGreaterEqual(r, 0)
        self.assertLessEqual(r, 255)
        self.assertGreaterEqual(g, 0)
        self.assertLessEqual(g, 255)
        self.assertGreaterEqual(b, 0)
        self.assertLessEqual(b, 255)


class TestIntentAnalyzer(unittest.TestCase):
    """Test OS intent analysis"""

    def setUp(self):
        self.analyzer = IntentAnalyzer()

    def test_critical_section_intent(self):
        """Test critical section intent analysis"""
        intent = {
            'goal': 'critical_section',
            'context': {'reason': 'test'},
            'constraints': {}
        }

        concepts = self.analyzer.analyze(intent)

        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].operation, OperationType.ISOLATION)
        self.assertEqual(concepts[0].scope, Scope.CPU_LOCAL)
        self.assertEqual(concepts[0].atomicity, Atomicity.ATOMIC)
        self.assertEqual(concepts[0].safety, SafetyLevel.CRITICAL)

    def test_memory_allocation_intent(self):
        """Test memory allocation intent analysis"""
        intent = {
            'goal': 'memory_allocation',
            'context': {
                'size': 4096,
                'alignment': 'page_boundary',
                'purpose': 'kernel'
            }
        }

        concepts = self.analyzer.analyze(intent)

        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].operation, OperationType.MEMORY)
        self.assertEqual(concepts[0].scope, Scope.SYSTEM_WIDE)
        self.assertEqual(concepts[0].duration, Duration.PERSISTENT)
        self.assertEqual(concepts[0].metadata['size'], 4096)

    def test_interrupt_handling_intent(self):
        """Test interrupt handling intent analysis"""
        intent = {
            'goal': 'handle_interrupt',
            'context': {
                'irq': 0
            }
        }

        concepts = self.analyzer.analyze(intent)

        # Should produce 3 concepts: save, handle, restore
        self.assertEqual(len(concepts), 3)
        self.assertEqual(concepts[0].operation, OperationType.STATE_MANAGEMENT)
        self.assertEqual(concepts[1].operation, OperationType.INTERRUPT)
        self.assertEqual(concepts[2].operation, OperationType.STATE_MANAGEMENT)

    def test_context_switch_intent(self):
        """Test context switch intent analysis"""
        intent = {
            'goal': 'context_switch',
            'context': {
                'from_pid': 1,
                'to_pid': 2
            }
        }

        concepts = self.analyzer.analyze(intent)

        # Should produce 3 concepts: save, isolate, load
        self.assertEqual(len(concepts), 3)
        self.assertEqual(concepts[0].operation, OperationType.STATE_MANAGEMENT)
        self.assertEqual(concepts[1].operation, OperationType.ISOLATION)
        self.assertEqual(concepts[2].operation, OperationType.STATE_MANAGEMENT)

    def test_io_operation_intent(self):
        """Test I/O operation intent analysis"""
        intent = {
            'goal': 'io_operation',
            'context': {
                'device': 'ATA0',
                'operation': 'read'
            }
        }

        concepts = self.analyzer.analyze(intent)

        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].operation, OperationType.IO_OPERATION)
        self.assertEqual(concepts[0].scope, Scope.SYSTEM_WIDE)
        self.assertEqual(concepts[0].atomicity, Atomicity.BEST_EFFORT)

    def test_unknown_intent(self):
        """Test that unknown intent raises error"""
        intent = {
            'goal': 'unknown_goal_xyz'
        }

        with self.assertRaises(ValueError):
            self.analyzer.analyze(intent)


class TestX86CodeGenerator(unittest.TestCase):
    """Test x86_64 code generation"""

    def setUp(self):
        self.codegen = X86CodeGenerator()

    def test_generate_disable_interrupts(self):
        """Test generating CLI instruction"""
        concepts = [
            SemanticConcept(
                operation=OperationType.ISOLATION,
                scope=Scope.CPU_LOCAL,
                duration=Duration.TEMPORARY,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL,
                metadata={'action': 'disable_interrupts'}
            )
        ]

        code = self.codegen.generate(concepts)

        self.assertIn('cli', code)

    def test_generate_save_state(self):
        """Test generating state save instructions"""
        concepts = [
            SemanticConcept(
                operation=OperationType.STATE_MANAGEMENT,
                scope=Scope.CPU_LOCAL,
                duration=Duration.TEMPORARY,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL,
                metadata={'action': 'save_state'}
            )
        ]

        code = self.codegen.generate(concepts)

        self.assertIn('pusha', code)
        self.assertIn('pushf', code)

    def test_generate_memory_allocation(self):
        """Test generating memory allocation code"""
        concepts = [
            SemanticConcept(
                operation=OperationType.MEMORY,
                scope=Scope.SYSTEM_WIDE,
                duration=Duration.PERSISTENT,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL,
                metadata={'action': 'allocate', 'size': 4096}
            )
        ]

        code = self.codegen.generate(concepts)

        self.assertTrue(any('kmalloc' in instruction for instruction in code))


class TestARMCodeGenerator(unittest.TestCase):
    """Test ARM64 code generation"""

    def setUp(self):
        self.codegen = ARMCodeGenerator()

    def test_generate_disable_interrupts(self):
        """Test generating CPSID instruction"""
        concepts = [
            SemanticConcept(
                operation=OperationType.ISOLATION,
                scope=Scope.CPU_LOCAL,
                duration=Duration.TEMPORARY,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL,
                metadata={'action': 'disable_interrupts'}
            )
        ]

        code = self.codegen.generate(concepts)

        self.assertIn('cpsid i', code)

    def test_generate_save_state(self):
        """Test generating state save instructions"""
        concepts = [
            SemanticConcept(
                operation=OperationType.STATE_MANAGEMENT,
                scope=Scope.CPU_LOCAL,
                duration=Duration.TEMPORARY,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL,
                metadata={'action': 'save_state'}
            )
        ]

        code = self.codegen.generate(concepts)

        self.assertTrue(any('push' in instruction for instruction in code))


class TestRISCVCodeGenerator(unittest.TestCase):
    """Test RISC-V code generation"""

    def setUp(self):
        self.codegen = RISCVCodeGenerator()

    def test_generate_disable_interrupts(self):
        """Test generating interrupt disable instruction"""
        concepts = [
            SemanticConcept(
                operation=OperationType.ISOLATION,
                scope=Scope.CPU_LOCAL,
                duration=Duration.TEMPORARY,
                atomicity=Atomicity.ATOMIC,
                safety=SafetyLevel.CRITICAL,
                metadata={'action': 'disable_interrupts'}
            )
        ]

        code = self.codegen.generate(concepts)

        self.assertIn('csrc mstatus, 8', code)


class TestSemanticPipeline(unittest.TestCase):
    """Test complete semantic pipeline"""

    def test_critical_section_x86(self):
        """Test complete pipeline for critical section on x86"""
        pipeline = SemanticPipeline(target_platform='x86_64')

        intent = {
            'goal': 'critical_section',
            'context': {'reason': 'test'}
        }

        result = pipeline.process(intent)

        self.assertEqual(result['platform'], 'x86_64')
        self.assertIn('concepts', result)
        self.assertIn('pixels', result)
        self.assertIn('code', result)
        self.assertIn('cli', result['code'])

    def test_critical_section_arm(self):
        """Test complete pipeline for critical section on ARM"""
        pipeline = SemanticPipeline(target_platform='arm64')

        intent = {
            'goal': 'critical_section',
            'context': {'reason': 'test'}
        }

        result = pipeline.process(intent)

        self.assertEqual(result['platform'], 'arm64')
        self.assertIn('cpsid i', result['code'])

    def test_critical_section_riscv(self):
        """Test complete pipeline for critical section on RISC-V"""
        pipeline = SemanticPipeline(target_platform='riscv')

        intent = {
            'goal': 'critical_section',
            'context': {'reason': 'test'}
        }

        result = pipeline.process(intent)

        self.assertEqual(result['platform'], 'riscv')
        self.assertIn('csrc mstatus, 8', result['code'])

    def test_multi_platform_same_pixels(self):
        """Test that same intent produces same pixels across platforms"""
        intent = {
            'goal': 'critical_section',
            'context': {'reason': 'test'}
        }

        pipeline_x86 = SemanticPipeline('x86_64')
        pipeline_arm = SemanticPipeline('arm64')
        pipeline_riscv = SemanticPipeline('riscv')

        result_x86 = pipeline_x86.process(intent)
        result_arm = pipeline_arm.process(intent)
        result_riscv = pipeline_riscv.process(intent)

        # Pixels should be identical (platform-independent)
        self.assertEqual(result_x86['pixels'], result_arm['pixels'])
        self.assertEqual(result_arm['pixels'], result_riscv['pixels'])

        # Code should be different (platform-specific)
        self.assertNotEqual(result_x86['code'], result_arm['code'])
        self.assertNotEqual(result_arm['code'], result_riscv['code'])

    def test_invalid_platform(self):
        """Test that invalid platform raises error"""
        with self.assertRaises(ValueError):
            SemanticPipeline(target_platform='invalid_platform_xyz')

    def test_memory_allocation_pipeline(self):
        """Test complete pipeline for memory allocation"""
        pipeline = SemanticPipeline('x86_64')

        intent = {
            'goal': 'memory_allocation',
            'context': {
                'size': 4096,
                'purpose': 'kernel'
            }
        }

        result = pipeline.process(intent)

        self.assertIn('concepts', result)
        self.assertEqual(len(result['concepts']), 1)
        self.assertEqual(result['concepts'][0]['operation'], 'memory')


class TestSemanticConsistency(unittest.TestCase):
    """Test semantic consistency across the pipeline"""

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding preserves approximate semantics"""
        encoder = PixelEncoder()

        concept = SemanticConcept(
            operation=OperationType.ISOLATION,
            scope=Scope.CPU_LOCAL,
            duration=Duration.TEMPORARY,
            atomicity=Atomicity.ATOMIC,
            safety=SafetyLevel.CRITICAL
        )

        # Encode
        pixel = encoder.encode(concept)

        # Decode
        decoded = encoder.decode(pixel)

        # Check that main properties are preserved
        self.assertEqual(decoded['operation'], 'isolation')
        self.assertEqual(decoded['scope'], 'cpu_local')
        self.assertEqual(decoded['atomicity'], 'atomic')


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("=" * 80)
    print("SEMANTIC ABSTRACTION LAYER - UNIT TESTS")
    print("=" * 80)
    print()

    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)

    sys.exit(0 if result.wasSuccessful() else 1)
