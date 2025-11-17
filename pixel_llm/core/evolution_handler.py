#!/usr/bin/env python3
"""
Evolution Handler - Process Evolution Tasks in Coaching System

This module bridges the coaching system and world rebuilder.
When the coach encounters a WORLD_REBUILD or ARCHITECTURE_CHANGE task,
this handler executes it and updates the task queue.

Usage in coaching loop:
    from pixel_llm.core.evolution_handler import handle_evolution_task

    if task.action in ["world_rebuild", "architecture_change"]:
        result = handle_evolution_task(task)
"""

from typing import Dict, Optional
from pathlib import Path
import json

from pixel_llm.core.task_queue import Task, TaskStatus, TaskAction, get_queue
from pixel_llm.core.world_rebuilder import execute_world_rebuild
from pixel_llm.core.cartridge_manager import get_manager


class EvolutionGuardrails:
    """
    Guardrails for when evolution is allowed.

    Prevents the LLM from spamming rebuilds or making changes at bad times.
    """

    @staticmethod
    def should_allow_world_rebuild(context: Dict = None) -> tuple[bool, str]:
        """
        Determine if a WORLD_REBUILD should be allowed.

        Args:
            context: Current system state (tests passing, tech debt, etc.)

        Returns:
            (allowed: bool, reason: str)
        """
        context = context or {}

        # Check 1: Are tests currently passing?
        if not context.get("tests_passing", True):
            return False, "Cannot rebuild while tests are failing - fix tests first"

        # Check 2: Is there already an experiment in progress?
        manager = get_manager()
        experiments = manager.manifest.get("experiments", {})

        if len(experiments) >= 3:
            return False, f"Too many experiments ({len(experiments)}) - promote or deprecate some first"

        # Check 3: Has there been a recent rebuild?
        recent_cartridges = manager.list_cartridges()
        if recent_cartridges:
            latest = recent_cartridges[0]
            # If latest cartridge is less than 1 hour old, slow down
            # (This is simplified - would check created_at timestamp in production)
            if latest.get("status") == "experimental":
                return False, "Recent rebuild in progress - test/promote existing experiment first"

        # Check 4: Tech debt threshold (if provided)
        tech_debt = context.get("tech_debt_score", 0)
        if tech_debt < 0.7:
            return False, f"Tech debt ({tech_debt:.1f}) not high enough to justify rebuild"

        # All checks passed
        return True, "Guardrails satisfied - rebuild allowed"

    @staticmethod
    def should_allow_architecture_change(context: Dict = None) -> tuple[bool, str]:
        """
        Determine if an ARCHITECTURE_CHANGE should be allowed.

        Similar to world rebuild but less strict.
        """
        context = context or {}

        # Architecture changes are less risky than full rebuilds
        # Allow more frequently

        if not context.get("tests_passing", True):
            return False, "Cannot change architecture while tests failing"

        return True, "Architecture change allowed"


def handle_evolution_task(task: Task, context: Dict = None) -> Dict:
    """
    Handle evolution tasks (WORLD_REBUILD, ARCHITECTURE_CHANGE, MIGRATION).

    This is called by the coaching system when it dequeues an evolution task.

    Args:
        task: The evolution task to handle
        context: Current system context (tests, metrics, etc.)

    Returns:
        Result dict with success status, outputs, etc.
    """
    queue = get_queue()
    action = task.action

    print(f"\n{'='*60}")
    print(f"EVOLUTION HANDLER: {action}")
    print(f"Task: {task.title}")
    print(f"{'='*60}\n")

    try:
        # Check guardrails
        if action == "world_rebuild":
            allowed, reason = EvolutionGuardrails.should_allow_world_rebuild(context)
        elif action == "architecture_change":
            allowed, reason = EvolutionGuardrails.should_allow_architecture_change(context)
        else:
            allowed, reason = True, "No specific guardrails for this action"

        if not allowed:
            print(f"❌ Guardrail blocked evolution: {reason}")
            queue.fail_task(task.id, f"Guardrail: {reason}")
            return {
                "success": False,
                "blocked": True,
                "reason": reason
            }

        print(f"✅ Guardrails passed: {reason}")

        # Mark task as in progress
        queue.start_task(task.id)

        # Route to appropriate handler
        if action == "world_rebuild":
            result = _handle_world_rebuild(task, context)
        elif action == "architecture_change":
            result = _handle_architecture_change(task, context)
        elif action == "migration":
            result = _handle_migration(task, context)
        else:
            raise ValueError(f"Unknown evolution action: {action}")

        # Update task based on result
        if result["success"]:
            queue.complete_task(task.id, result={
                "cartridge": result.get("cartridge"),
                "test_results": result.get("test_results"),
                "workspace": result.get("workspace")
            })
            print(f"\n✅ Evolution task completed: {task.title}")
        else:
            queue.fail_task(task.id, result.get("error", "Unknown error"))
            print(f"\n❌ Evolution task failed: {result.get('error')}")

        return result

    except Exception as e:
        print(f"\n❌ Evolution handler error: {str(e)}")
        import traceback
        traceback.print_exc()

        queue.fail_task(task.id, str(e))

        return {
            "success": False,
            "error": str(e)
        }


def _handle_world_rebuild(task: Task, context: Dict = None) -> Dict:
    """Execute a WORLD_REBUILD task"""
    print("\n🌍 Executing WORLD_REBUILD...")

    # Use world rebuilder
    result = execute_world_rebuild(task)

    if result["success"]:
        cartridge = result["cartridge"]
        print(f"\n✨ New cartridge created: {cartridge}")
        print(f"   Workspace: {result['workspace']}")
        print(f"   Modules built: {result['modules_built']}")
        print(f"   Tests: {result['test_results'].get('tests_passed', 0)} passed")

        # Auto-run Genesis tests if configured
        auto_test = context.get("auto_test_evolution", False) if context else False
        if auto_test:
            print(f"\n🧪 Auto-testing {cartridge}...")
            # Would run: subprocess.run(["python", "pxos_shim.py", "test", "--cartridge", cartridge])
            print("   (Auto-test would run here)")

        print(f"\n📋 Next steps:")
        print(f"   1. Test: python pxos_shim.py test --cartridge {cartridge}")
        print(f"   2. Review: Check {result['workspace']}/GENESIS_COMPLIANCE.md")
        print(f"   3. Promote: python pxos_shim.py promote {cartridge}")

    return result


def _handle_architecture_change(task: Task, context: Dict = None) -> Dict:
    """Execute an ARCHITECTURE_CHANGE task"""
    print("\n🔧 Executing ARCHITECTURE_CHANGE...")

    # Architecture change is lighter than full rebuild
    # In this version, it's similar to world_rebuild but could be:
    # - Design doc generation
    # - Proof of concept
    # - Benchmark comparison

    # For now, treat it like a world rebuild
    # (In production, you might have a separate handler)

    print("⚠️  Architecture change handler is stub - treating as world rebuild")
    return _handle_world_rebuild(task, context)


def _handle_migration(task: Task, context: Dict = None) -> Dict:
    """Execute a MIGRATION task"""
    print("\n🚚 Executing MIGRATION...")

    # Migration would involve:
    # 1. Read migration plan
    # 2. Execute steps
    # 3. Validate backward compatibility
    # 4. Run full test suite

    print("⚠️  Migration handler is stub")

    return {
        "success": False,
        "error": "Migration handler not yet implemented"
    }


def can_propose_evolution(reason: str, current_context: Dict = None) -> bool:
    """
    Check if LLM can propose evolution given current context.

    This is a helper for the coaching system to decide whether to even
    consider evolution suggestions from the LLM.

    Args:
        reason: LLM's stated reason for evolution
        current_context: System metrics, test status, etc.

    Returns:
        True if evolution proposal should be accepted
    """
    # Require substantive reason
    if not reason or len(reason) < 20:
        print("❌ Evolution proposal rejected: insufficient justification")
        return False

    # Check guardrails
    allowed, guardrail_reason = EvolutionGuardrails.should_allow_world_rebuild(current_context)

    if not allowed:
        print(f"❌ Evolution proposal rejected: {guardrail_reason}")
        return False

    # Keyword checks - require certain phrases
    required_indicators = [
        any(word in reason.lower() for word in ["better", "simpler", "faster", "cleaner"]),
        any(word in reason.lower() for word in ["architecture", "design", "structure"]),
    ]

    if not all(required_indicators):
        print("❌ Evolution proposal rejected: doesn't explain architectural improvement")
        return False

    print(f"✅ Evolution proposal accepted: {reason[:100]}...")
    return True


# System prompts for LLM awareness

EVOLUTION_PROMPT_SNIPPET = """
## Evolution Capabilities

You can propose architectural improvements to pxOS when you discover better designs.

### When to Propose Evolution:

✅ **Do propose** when:
- Tests are passing but architecture feels blocked/tangled
- You find a fundamentally simpler/faster design
- Genesis principles could be satisfied better
- Technical debt is high across multiple modules

❌ **Don't propose** when:
- Tests are failing (fix tests first)
- Only minor refactoring needed (just do it)
- No clear architectural improvement
- Recent evolution already in progress

### How to Propose:

```python
from pixel_llm.core.task_queue import create_world_rebuild_task

task_id = create_world_rebuild_task(
    target_version="1.1.0",  # or "2.0.0" for major redesign
    parent_cartridge="pxos_v1_0_0.pxa",
    reason="Detailed explanation: current arch X has problems Y, new arch Z solves them by..."
)
```

### What Happens Next:

1. Coaching system validates your proposal
2. World rebuilder creates new cartridge in isolation
3. Genesis tests validate compliance
4. Human guardian reviews and approves/rejects
5. If approved: new version becomes current, old preserved
6. If rejected: stays as historical experiment

### Examples:

**Good proposal:**
- "PixelFS and InfiniteMap have overlapping responsibilities. Unified PixelStore would be simpler,
  eliminate 500 lines, and better satisfy Genesis §1."

**Bad proposal:**
- "Let's rebuild everything" (no clear improvement)
- "Found a bug in PixelFS" (just fix the bug, don't rebuild)
"""


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EVOLUTION HANDLER TEST")
    print("="*60)

    # Test guardrails
    print("\n1. Testing guardrails...")

    # Should fail - tests not passing
    allowed, reason = EvolutionGuardrails.should_allow_world_rebuild({
        "tests_passing": False
    })
    print(f"   Tests failing: {allowed} - {reason}")

    # Should fail - tech debt too low
    allowed, reason = EvolutionGuardrails.should_allow_world_rebuild({
        "tests_passing": True,
        "tech_debt_score": 0.3
    })
    print(f"   Low tech debt: {allowed} - {reason}")

    # Should succeed
    allowed, reason = EvolutionGuardrails.should_allow_world_rebuild({
        "tests_passing": True,
        "tech_debt_score": 0.8
    })
    print(f"   Good conditions: {allowed} - {reason}")

    # Test proposal validation
    print("\n2. Testing proposal validation...")

    # Bad reason
    ok = can_propose_evolution("rebuild", {"tests_passing": True})
    print(f"   Bad reason: {ok}")

    # Good reason
    ok = can_propose_evolution(
        "Current architecture has tangled dependencies between PixelFS and InfiniteMap. "
        "A simpler unified design would be better and cleaner.",
        {"tests_passing": True, "tech_debt_score": 0.8}
    )
    print(f"   Good reason: {ok}")

    print("\n" + "="*60)
    print("✅ Evolution handler ready")
    print("="*60 + "\n")
