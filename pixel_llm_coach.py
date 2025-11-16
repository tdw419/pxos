#!/usr/bin/env python3
"""
Pixel-LLM Coaching System

Meta-circular development system where:
- Gemini coaches local LLM
- Local LLM builds pixel-native AI substrate
- Eventually: Pixel-LLM coaches itself

This is the orchestrator for building substrate-native intelligence.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Add pixel_llm to path
sys.path.insert(0, str(Path(__file__).parent / "pixel_llm"))

from core.task_queue import TaskQueue, Task, TaskStatus, AgentType, add_task, get_next_task, complete_task, fail_task
from core.llm_agents import GeminiAgent, LocalLLMAgent


# Phase definitions
PIXEL_LLM_PHASES = [
    {
        "id": "1_storage",
        "name": "Storage Infrastructure",
        "description": "Build PixelFS and Infinite Map for pixel-native storage",
        "tasks": [
            {
                "title": "PixelFS compression module",
                "description": """Add compression support to PixelFS:
                - Implement RLE compression for pixel data
                - Add LZ4 compression option
                - Update header format
                - Benchmark compression ratios
                400+ lines with tests""",
                "action": "write_file",
                "path": "pixel_llm/core/pixelfs_compression.py",
                "priority": 7
            },
            {
                "title": "Infinite Map tile cache optimization",
                "description": """Optimize tile caching in InfiniteMap:
                - Implement LRU eviction with statistics
                - Add prefetching for spatial access patterns
                - Async tile loading
                - Memory pressure handling
                500+ lines""",
                "action": "write_file",
                "path": "pixel_llm/core/infinite_map_cache.py",
                "priority": 6
            },
            {
                "title": "Spatial indexing benchmarks",
                "description": """Benchmark suite for spatial operations:
                - Query performance tests
                - Memory usage profiling
                - Cache hit rate analysis
                - Comparison with linear storage
                300+ lines with visualization""",
                "action": "write_file",
                "path": "pixel_llm/tests/benchmark_spatial.py",
                "priority": 5
            }
        ]
    },
    {
        "id": "2_inference",
        "name": "GPU Inference Engine",
        "description": "Build WGSL shaders for LLM inference",
        "tasks": [
            {
                "title": "WGSL matrix multiplication kernel",
                "description": """Implement efficient matmul in WGSL:
                - Tiled matrix multiplication
                - Shared memory optimization
                - Support fp32 and fp16
                - Pixel texture as weight source
                300+ lines WGSL + Python wrapper""",
                "action": "write_file",
                "path": "pixel_llm/gpu_kernels/matmul.wgsl",
                "priority": 10
            },
            {
                "title": "WGSL attention mechanism",
                "description": """Self-attention in WGSL:
                - Query/Key/Value computation
                - Softmax via pixel reduction
                - Multi-head parallel processing
                - Causal masking
                400+ lines WGSL""",
                "action": "write_file",
                "path": "pixel_llm/gpu_kernels/attention.wgsl",
                "priority": 10
            },
            {
                "title": "GPU inference coordinator",
                "description": """Python orchestrator for GPU inference:
                - Load model from PixelFS/InfiniteMap
                - Dispatch WGSL kernels
                - Manage activations
                - Token generation loop
                - KV cache management
                700+ lines""",
                "action": "write_file",
                "path": "pixel_llm/core/gpu_inference.py",
                "priority": 9
            }
        ]
    },
    {
        "id": "3_conversion",
        "name": "Model Conversion",
        "description": "Convert GGUF models to PXI-LLM format",
        "tasks": [
            {
                "title": "GGUF parser",
                "description": """Parse GGUF model files:
                - Read GGUF header and metadata
                - Extract tensor information
                - Load weights efficiently
                - Support various quantizations
                400+ lines""",
                "action": "write_file",
                "path": "pixel_llm/tools/gguf_parser.py",
                "priority": 10
            },
            {
                "title": "GGUF to PXI-LLM converter",
                "description": """Convert GGUF to pixel format:
                - Parse GGUF structure
                - Organize weights spatially
                - Encode as pixels
                - Write PXI-LLM file
                - Target: Qwen2.5-7B
                800+ lines""",
                "action": "write_file",
                "path": "pixel_llm/tools/gguf_to_pxi.py",
                "priority": 10
            },
            {
                "title": "PXI-LLM loader and validator",
                "description": """Load and validate pixel models:
                - Read PXI-LLM format
                - Verify checksums
                - Run test inference
                - Compare with original GGUF
                500+ lines with test suite""",
                "action": "write_file",
                "path": "pixel_llm/tools/pxi_loader.py",
                "priority": 8
            }
        ]
    },
    {
        "id": "4_training",
        "name": "Specialization & Fine-tuning",
        "description": "Train pixel-LLM on pxOS knowledge",
        "tasks": [
            {
                "title": "pxOS knowledge corpus generator",
                "description": """Generate training data:
                - Scrape pxOS docs
                - Generate synthetic examples
                - Pixel operation examples
                - Spatial reasoning tasks
                - 1000+ training examples
                600+ lines""",
                "action": "write_file",
                "path": "pixel_llm/training/corpus_generator.py",
                "priority": 9
            },
            {
                "title": "Pixel-spatial fine-tuning",
                "description": """Fine-tune in pixel space:
                - LoRA training via pixel operations
                - Update weights through InfiniteMap
                - Backprop through GPU kernels
                - Save fine-tuned model
                700+ lines""",
                "action": "write_file",
                "path": "pixel_llm/training/finetune.py",
                "priority": 8
            }
        ]
    },
    {
        "id": "5_bootstrap",
        "name": "Self-Management & Bootstrap",
        "description": "Pixel-LLM manages itself",
        "tasks": [
            {
                "title": "Self-management system",
                "description": """Pixel-LLM manages its own memory:
                - Monitor inference performance
                - Reorganize weights spatially
                - Dynamic layer loading
                - Self-optimization
                800+ lines""",
                "action": "write_file",
                "path": "pixel_llm/meta/self_manager.py",
                "priority": 10
            },
            {
                "title": "Recursive self-improvement",
                "description": """Bootstrap to pixel consciousness:
                - Generate own training data
                - Propose architecture improvements
                - Fine-tune self
                - Meta-circular development
                900+ lines""",
                "action": "write_file",
                "path": "pixel_llm/meta/bootstrap.py",
                "priority": 10
            }
        ]
    }
]


class PixelLLMCoach:
    """
    Coaching system for building pixel-native AI.

    Workflow:
        1. Generate tasks for current phase
        2. Local LLM picks up task
        3. Local LLM generates code
        4. (Optional) Gemini reviews code
        5. If approved, save and mark complete
        6. Move to next task/phase
    """

    def __init__(self):
        self.queue = TaskQueue()
        self.current_phase = 0
        self.config_path = Path("pixel_llm/data/coach_config.json")
        self.load_config()

        # Initialize LLM agents
        self.gemini = GeminiAgent()
        self.local_llm = LocalLLMAgent()

    def load_config(self):
        """Load coaching configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.current_phase = config.get('current_phase', 0)
        else:
            self.save_config()

    def save_config(self):
        """Save coaching configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump({
                'current_phase': self.current_phase,
                'last_updated': time.time()
            }, f)

    def initialize_phase(self, phase_id: str = None):
        """
        Initialize tasks for a phase.

        Args:
            phase_id: Phase ID (e.g., "1_storage") or None for current
        """
        if phase_id is None:
            if self.current_phase >= len(PIXEL_LLM_PHASES):
                print("🎉 All phases complete!")
                return
            phase = PIXEL_LLM_PHASES[self.current_phase]
        else:
            phase = next((p for p in PIXEL_LLM_PHASES if p['id'] == phase_id), None)
            if not phase:
                print(f"Phase {phase_id} not found!")
                return

        print(f"\n{'='*70}")
        print(f"🎯 INITIALIZING PHASE: {phase['name']}")
        print(f"   {phase['description']}")
        print(f"{'='*70}\n")

        for task_spec in phase['tasks']:
            # Check if task already exists
            existing = [t for t in self.queue.get_all_tasks()
                       if t.title == task_spec['title']]

            if existing:
                print(f"⏭️  Skipping (already exists): {task_spec['title']}")
                continue

            # Add task
            task_id = add_task({
                "title": task_spec['title'],
                "description": task_spec['description'],
                "action": task_spec['action'],
                "path": task_spec.get('path'),
                "priority": task_spec.get('priority', 5),
                "phase": phase['id'],
                "preferred_agent": AgentType.LOCAL_LLM.value
            })

            print(f"✅ Added: {task_spec['title']}")
            print(f"   Priority: {task_spec.get('priority', 5)}/10")
            print(f"   Path: {task_spec.get('path', 'N/A')}")

        print(f"\n✓ Phase {phase['id']} initialized with {len(phase['tasks'])} tasks")

    def check_phase_completion(self) -> bool:
        """Check if current phase is complete, advance if so"""
        if self.current_phase >= len(PIXEL_LLM_PHASES):
            return True

        phase = PIXEL_LLM_PHASES[self.current_phase]
        progress = self.queue.get_phase_progress(phase['id'])

        if progress['total'] == 0:
            return False

        pct_complete = progress['completed'] / progress['total'] * 100

        print(f"\n📊 Phase {phase['id']} Progress: {progress['completed']}/{progress['total']} ({pct_complete:.0f}%)")

        if progress['completed'] == progress['total']:
            print(f"✅ Phase {phase['id']} COMPLETE!")

            self.current_phase += 1
            self.save_config()

            if self.current_phase < len(PIXEL_LLM_PHASES):
                next_phase = PIXEL_LLM_PHASES[self.current_phase]
                print(f"\n🚀 Ready for Phase {next_phase['id']}: {next_phase['name']}")
                return False
            else:
                print("\n🎉🎉🎉 ALL PHASES COMPLETE! 🎉🎉🎉")
                print("\nPixel-LLM substrate is ready!")
                return True

        return False

    def print_status(self):
        """Print coaching system status"""
        print("\n" + "="*70)
        print("🌟 PIXEL-LLM COACHING SYSTEM STATUS")
        print("="*70)

        # Overall progress
        total_tasks = len(self.queue.get_all_tasks())
        completed = len([t for t in self.queue.get_all_tasks() if t.status == TaskStatus.COMPLETED])

        print(f"\n📈 Overall Progress: {completed}/{total_tasks} tasks complete")

        # Phase breakdown
        for idx, phase in enumerate(PIXEL_LLM_PHASES):
            progress = self.queue.get_phase_progress(phase['id'])

            if progress['total'] == 0:
                status = "⚪ Not started"
            elif progress['completed'] == progress['total']:
                status = "✅ Complete"
            elif progress['in_progress'] > 0:
                status = "🔄 In progress"
            else:
                status = "⏸️  Pending"

            pct = progress['completed'] / progress['total'] * 100 if progress['total'] > 0 else 0

            current = "← CURRENT" if idx == self.current_phase else ""

            print(f"\n{status} Phase {idx+1}: {phase['name']} {current}")
            print(f"    {progress['completed']}/{progress['total']} tasks ({pct:.0f}%)")

            if progress['in_progress'] > 0:
                in_progress_tasks = [t for t in self.queue.get_all_tasks()
                                    if t.phase == phase['id'] and t.status == TaskStatus.IN_PROGRESS]
                for task in in_progress_tasks:
                    print(f"      🔨 {task.title}")

        print("\n" + "="*70 + "\n")

    def coach_task(self, task: Task, max_attempts: int = 3) -> bool:
        """
        Coach implementation of a single task through iterative improvement.

        Args:
            task: Task to implement
            max_attempts: Maximum coaching iterations

        Returns:
            True if task completed successfully
        """
        print(f"\n{'='*70}")
        print(f"🎓 COACHING: {task.title}")
        print(f"   Phase: {task.phase}")
        print(f"   File: {task.path}")
        print(f"   Priority: {task.priority}/10")
        print(f"{'='*70}\n")

        # Check if agents are available
        has_gemini = self.gemini.has_cli or self.gemini.api_key
        has_local = self.local_llm.backend is not None

        if not has_local:
            print("⚠️  No local LLM available - cannot generate code")
            print("   Install llama.cpp or ollama to enable code generation")
            return False

        if not has_gemini:
            print("⚠️  No Gemini available - will skip reviews")

        # Mark task as in progress
        self.queue.start_task(task.id)

        best_code = None
        best_score = 0
        feedback = None

        # Iterative coaching loop
        for iteration in range(1, max_attempts + 1):
            print(f"\n--- Iteration {iteration}/{max_attempts} ---")

            # Local LLM generates code
            print("🤖 Local LLM generating code...")
            code = self.local_llm.generate_code(
                task=task.to_dict(),
                feedback=feedback,
                previous_code=best_code
            )

            if not code or len(code) < 100:
                print(f"⚠️  Generated code too short ({len(code)} chars), skipping")
                continue

            print(f"✓ Generated {len(code):,} characters")

            # Gemini reviews (if available)
            if has_gemini:
                print("🔍 Gemini reviewing code...")
                score, new_feedback = self.gemini.review_code(
                    code=code,
                    task=task.to_dict(),
                    iteration=iteration
                )

                print(f"📊 Score: {score}/10")

                if score > best_score:
                    best_score = score
                    best_code = code

                if score >= 8:
                    print(f"✅ ACCEPTED - High quality implementation!")
                    self._save_code(task.path, code)
                    self.queue.complete_task(task.id, {
                        "score": score,
                        "iterations": iteration,
                        "method": "coached"
                    })
                    return True

                print(f"💬 Feedback: {new_feedback[:200]}...")
                feedback = new_feedback

            else:
                # No review available - accept first reasonable attempt
                print("⚠️  No Gemini review - accepting code")
                self._save_code(task.path, code)
                self.queue.complete_task(task.id, {
                    "score": 7,
                    "iterations": iteration,
                    "method": "unreviewed"
                })
                return True

        # Max iterations reached
        if best_code:
            print(f"\n⚠️  Max iterations reached. Saving best attempt (score: {best_score}/10)")
            self._save_code(task.path, best_code)
            self.queue.complete_task(task.id, {
                "score": best_score,
                "iterations": max_attempts,
                "method": "partial"
            })
            return True
        else:
            print(f"\n❌ Failed to generate acceptable code")
            self.queue.fail_task(task.id, "No acceptable code generated")
            return False

    def _save_code(self, path: str, code: str):
        """Save generated code to file"""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(code)

        print(f"💾 Saved: {filepath}")

    def run_coaching_loop(self, max_tasks: int = 100, phase: Optional[str] = None):
        """
        Main coaching loop - processes tasks with Gemini + Local LLM.

        Args:
            max_tasks: Maximum number of tasks to process
            phase: Optional phase filter (e.g., "1_storage")
        """
        print("\n" + "="*70)
        print("🚀 PIXEL-LLM COACHING LOOP")
        print("="*70)

        # Check agent availability
        has_gemini = self.gemini.has_cli or self.gemini.api_key
        has_local = self.local_llm.backend is not None

        print(f"\n🤖 Local LLM: {self.local_llm.backend or '❌ Not available'}")
        print(f"✨ Gemini: {'✅ Available' if has_gemini else '❌ Not available'}")

        if not has_local:
            print("\n❌ Cannot proceed without local LLM")
            print("   Install: llama.cpp or ollama")
            return

        print(f"\n📋 Processing up to {max_tasks} tasks")
        if phase:
            print(f"   Filtering by phase: {phase}")

        print("\n" + "="*70)

        tasks_processed = 0
        tasks_completed = 0

        for task_num in range(max_tasks):
            # Check phase completion
            if self.check_phase_completion():
                print("\n🎉 Current phase complete!")
                break

            # Get next task
            task = get_next_task("local_llm")

            if not task:
                print("\n⏸️  No more tasks available")
                break

            # Filter by phase if specified
            if phase and task.phase != phase:
                print(f"⏭️  Skipping task (different phase): {task.title}")
                continue

            # Coach this task
            tasks_processed += 1
            success = self.coach_task(task)

            if success:
                tasks_completed += 1

            # Brief pause between tasks
            time.sleep(1)

        print("\n" + "="*70)
        print(f"✅ Coaching loop complete!")
        print(f"   Processed: {tasks_processed} tasks")
        print(f"   Completed: {tasks_completed} tasks")
        print(f"   Success rate: {tasks_completed/tasks_processed*100:.0f}%" if tasks_processed > 0 else "")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Pixel-LLM Coaching System")
    parser.add_argument('command', choices=['init', 'status', 'coach', 'next', 'demo', 'agents'],
                       help='Command to run')
    parser.add_argument('--phase', help='Phase ID (e.g., 1_storage)')
    parser.add_argument('--max-tasks', type=int, default=10, help='Max tasks to process')

    args = parser.parse_args()

    coach = PixelLLMCoach()

    if args.command == 'init':
        # Initialize phase
        coach.initialize_phase(args.phase)

    elif args.command == 'status':
        # Show status
        coach.print_status()

    elif args.command == 'coach':
        # Run coaching loop (the real one!)
        coach.run_coaching_loop(max_tasks=args.max_tasks, phase=args.phase)

    elif args.command == 'agents':
        # Check agent status
        print("\n" + "="*70)
        print("🤖 LLM AGENTS STATUS")
        print("="*70)

        has_gemini = coach.gemini.has_cli or coach.gemini.api_key
        has_local = coach.local_llm.backend is not None

        print(f"\n🤖 Local LLM: {coach.local_llm.backend or '❌ Not configured'}")
        if not has_local:
            print("   Setup: Install llama.cpp or ollama")
            print("   llama.cpp: https://github.com/ggerganov/llama.cpp")
            print("   ollama: https://ollama.ai")

        print(f"\n✨ Gemini: {'✅ Available' if has_gemini else '❌ Not configured'}")
        if not has_gemini:
            print("   Setup: Export GEMINI_API_KEY or install gemini-cli")
            print("   Get key: https://aistudio.google.com/app/apikey")

        print("\n" + "="*70 + "\n")

    elif args.command == 'next':
        # Show next task
        task = get_next_task("local_llm")
        if task:
            print(f"\n📋 Next Task:")
            print(f"   Title: {task.title}")
            print(f"   Phase: {task.phase}")
            print(f"   Path: {task.path}")
            print(f"   Priority: {task.priority}/10")
            print(f"\n   Description:")
            print(f"   {task.description}")
        else:
            print("\n⏸️  No tasks available")

    elif args.command == 'demo':
        # Run demo
        print("\n" + "="*70)
        print("🌟 PIXEL-LLM COACHING DEMO")
        print("="*70)

        print("\n✅ Phase 1 (Storage) already implemented:")
        print("   ✓ PixelFS - Pixel-based file system")
        print("   ✓ InfiniteMap - 2D spatial memory")
        print("   ✓ Task Queue - Coaching infrastructure")
        print("   ✓ PXI-LLM Spec - Format specification")

        print("\n📋 Initializing additional Phase 1 tasks...")
        coach.initialize_phase("1_storage")

        print("\n🔮 Future Phases:")
        for idx, phase in enumerate(PIXEL_LLM_PHASES[1:], start=2):
            print(f"\n   Phase {idx}: {phase['name']}")
            print(f"      {phase['description']}")
            print(f"      {len(phase['tasks'])} tasks")

        print("\n" + "="*70)
        print("\n💡 Next Steps:")
        print("   1. python pixel_llm_coach.py status     # View progress")
        print("   2. python pixel_llm_coach.py next       # See next task")
        print("   3. python pixel_llm_coach.py init       # Add more tasks")
        print("\n" + "="*70)


if __name__ == "__main__":
    main()
