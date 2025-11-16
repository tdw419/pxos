# Autonomous Evolution Integration Guide

**How to Wire Evolution into the Coaching System**

This guide shows how to integrate the evolution system into `pixel_llm_coach.py` so LLMs can autonomously propose and execute improvements to pxOS.

---

## Overview: The Complete Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LLM Discovers Better Architecture                        │
│    → Calls create_world_rebuild_task()                      │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Task Added to Queue                                       │
│    → Status: PENDING, Action: world_rebuild                 │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Coaching Loop Dequeues Task                              │
│    → Detects action == "world_rebuild"                      │
│    → Calls handle_evolution_task()                          │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Evolution Handler Checks Guardrails                      │
│    → Tests passing? ✓                                       │
│    → Tech debt high enough? ✓                               │
│    → No recent experiments? ✓                               │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. World Rebuilder Executes                                 │
│    → Creates /tmp/pxos_world_build_X/                       │
│    → Generates modules (via coaching)                       │
│    → Runs tests                                             │
│    → Packs into pxos_vX_Y_Z.pxa                             │
│    → Registers as experimental                              │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Human Guardian Reviews                                   │
│    → python pxos_shim.py test --cartridge <name>            │
│    → python pxos_shim.py lineage <name>                     │
│    → Review GENESIS_COMPLIANCE.md                           │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Promotion or Rejection                                   │
│    → python pxos_shim.py promote <name> (if good)           │
│    → Leave as historical experiment (if not)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Update Coaching System Imports

Add these imports to `pixel_llm_coach.py`:

```python
from pixel_llm.core.task_queue import TaskAction
from pixel_llm.core.evolution_handler import handle_evolution_task, EVOLUTION_PROMPT_SNIPPET
```

---

## Step 2: Add Evolution Handler to Main Loop

In your main coaching loop, add evolution task detection:

```python
def process_task(task: Task) -> Dict:
    """
    Process a single task from the queue.

    Now handles evolution tasks (WORLD_REBUILD, etc.)
    """

    # Check if this is an evolution task
    if task.action in [TaskAction.WORLD_REBUILD, TaskAction.ARCHITECTURE_CHANGE, TaskAction.MIGRATION]:
        # Get current system context
        context = get_system_context()  # You define this

        # Handle evolution
        result = handle_evolution_task(task, context=context)

        return result

    # Existing handlers for regular tasks
    elif task.action == TaskAction.WRITE_FILE:
        return handle_write_file(task)

    elif task.action == TaskAction.EDIT_FILE:
        return handle_edit_file(task)

    # ... other handlers ...

    else:
        raise ValueError(f"Unknown task action: {task.action}")
```

---

## Step 3: Define System Context Helper

The evolution handler needs context about current system state:

```python
def get_system_context() -> Dict:
    """
    Get current system state for evolution decisions.

    This is used by guardrails to decide if evolution should be allowed.
    """
    import subprocess

    # Run tests to check if they're passing
    result = subprocess.run(
        ["./pixel_llm/tests/run_tests.sh"],
        capture_output=True,
        text=True
    )
    tests_passing = (result.returncode == 0)

    # Calculate tech debt (simplified example)
    # In production, you might analyze:
    # - Code complexity metrics
    # - Number of TODOs/FIXMEs
    # - Test coverage trends
    # - Module coupling
    tech_debt_score = calculate_tech_debt()  # You implement

    # Check for blocked work
    queue = get_queue()
    blocked_tasks = len(queue.get_all_tasks(status=TaskStatus.BLOCKED))

    return {
        "tests_passing": tests_passing,
        "tech_debt_score": tech_debt_score,
        "blocked_tasks": blocked_tasks,
        "auto_test_evolution": True,  # Auto-test new cartridges
    }


def calculate_tech_debt() -> float:
    """
    Calculate tech debt score (0.0 = clean, 1.0 = very messy).

    Example implementation:
    """
    # Count TODOs
    todo_count = 0
    for py_file in Path("pixel_llm").rglob("*.py"):
        with open(py_file) as f:
            todo_count += f.read().lower().count("todo")

    # Get test coverage
    # (Would parse coverage report in production)
    coverage = 0.55  # 55%

    # Simple heuristic
    score = 0.0
    score += min(todo_count / 100, 0.3)  # TODOs contribute up to 0.3
    score += (1.0 - coverage) * 0.5      # Low coverage adds up to 0.5

    return min(score, 1.0)
```

---

## Step 4: Add Evolution Prompt to LLM System Prompt

When initializing your LLM agent, include evolution instructions:

```python
from pixel_llm.core.evolution_handler import EVOLUTION_PROMPT_SNIPPET

def create_coaching_prompt() -> str:
    """Build system prompt for coaching LLM"""

    base_prompt = """
You are the Pixel-LLM coaching system. Your job is to build and improve
pxOS - a pixel-native AI system where code and data live IN pixels.

...existing instructions...
    """

    # Add evolution capabilities
    full_prompt = base_prompt + "\n\n" + EVOLUTION_PROMPT_SNIPPET

    return full_prompt
```

---

## Step 5: Example Coaching Loop Integration

Here's a minimal complete integration:

```python
#!/usr/bin/env python3
"""
pixel_llm_coach.py - With Evolution Support
"""

from pixel_llm.core.task_queue import get_queue, TaskAction, TaskStatus
from pixel_llm.core.evolution_handler import handle_evolution_task, can_propose_evolution
from pixel_llm.core.llm_agents import LocalLLMAgent, GeminiAgent


def main():
    """Main coaching loop with evolution support"""

    # Initialize agents
    local_llm = LocalLLMAgent()
    gemini = GeminiAgent()

    queue = get_queue()

    print("🎓 Coaching system starting (with evolution support)...")

    while True:
        # Get next task
        task = queue.get_next_task(agent=AgentType.LOCAL_LLM)

        if not task:
            print("✅ No tasks in queue")
            break

        print(f"\n📋 Processing: {task.title}")
        print(f"   Action: {task.action}")

        # Route to appropriate handler
        try:
            if task.action in [TaskAction.WORLD_REBUILD, TaskAction.ARCHITECTURE_CHANGE]:
                # Evolution task
                print("🌍 Evolution task detected")

                context = get_system_context()
                result = handle_evolution_task(task, context=context)

                if result["success"]:
                    print(f"✅ Evolution completed: {result.get('cartridge')}")
                else:
                    print(f"❌ Evolution failed: {result.get('error')}")

            elif task.action == TaskAction.WRITE_FILE:
                # Regular code generation
                result = handle_regular_task(task, local_llm, gemini)

            else:
                print(f"⚠️  Unknown task action: {task.action}")
                queue.fail_task(task.id, f"Unknown action: {task.action}")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Task failed: {str(e)}")
            queue.fail_task(task.id, str(e))

    print("\n🎓 Coaching session complete")


def handle_regular_task(task, local_llm, gemini):
    """Handle regular code generation tasks"""
    # Existing coaching logic
    # (generate → review → iterate)
    pass


def get_system_context():
    """Get system state for evolution decisions"""
    return {
        "tests_passing": True,  # Would actually check
        "tech_debt_score": 0.6,
        "blocked_tasks": 0,
        "auto_test_evolution": True,
    }


if __name__ == "__main__":
    main()
```

---

## Step 6: Testing the Integration

### Test 1: Manual Evolution Task

```python
from pixel_llm.core.task_queue import create_world_rebuild_task

# LLM creates this task
task_id = create_world_rebuild_task(
    target_version="1.1.0",
    parent_cartridge="pxos_v1_0_0.pxa",
    reason="Simplified PixelFS architecture with unified storage API"
)

print(f"Created evolution task: {task_id}")
```

### Test 2: Run Coaching Loop

```bash
python pixel_llm_coach.py
# Should detect evolution task and call evolution handler
```

### Test 3: Check Results

```bash
# See what was created
python pxos_shim.py status

# Test the new cartridge
python pxos_shim.py test --cartridge pxos_v1_1_0.pxa

# View lineage
python pxos_shim.py lineage pxos_v1_1_0.pxa

# Promote if good
python pxos_shim.py promote pxos_v1_1_0.pxa \
  --reason "Simpler architecture, all tests pass"
```

---

## Guardrails: When Evolution is Blocked

The system will **block evolution** if:

1. **Tests are failing** → "Fix tests before proposing evolution"
2. **Too many experiments** (>3) → "Promote or deprecate existing experiments first"
3. **Recent rebuild in progress** → "Test/promote existing experiment first"
4. **Tech debt too low** (<0.7) → "Not enough justification for rebuild"
5. **Insufficient reasoning** → "Explain architectural improvement better"

### Example Rejection:

```
❌ Evolution proposal rejected: Cannot rebuild while tests are failing - fix tests first
```

---

## Advanced: LLM Self-Initiated Evolution

For fully autonomous evolution, add this to your coaching loop:

```python
def check_for_autonomous_evolution(context):
    """
    Let LLM autonomously propose evolution when conditions are right.

    This is called periodically (e.g., after completing a phase).
    """

    # Only check occasionally
    if not should_check_evolution(context):
        return

    # Ask LLM: "Should we evolve?"
    prompt = f"""
Current pxOS state:
- Tests passing: {context['tests_passing']}
- Tech debt: {context['tech_debt_score']:.2f}
- Coverage: {context.get('coverage', 0):.1%}

Given this state and Genesis principles, should we propose a WORLD_REBUILD?

If yes, explain:
1. What's wrong with current architecture
2. What the new architecture would be
3. How it better satisfies Genesis

If no, just say "No evolution needed".
    """

    response = local_llm.generate(prompt)

    if "no evolution needed" in response.lower():
        return

    # Parse LLM's proposal
    if can_propose_evolution(response, context):
        # Create evolution task
        task_id = create_world_rebuild_task(
            target_version=get_next_version(),
            parent_cartridge=get_current_cartridge(),
            reason=response
        )
        print(f"🌍 LLM autonomously proposed evolution: {task_id}")
```

---

## Safety Mechanisms

### 1. Human Approval Required

Evolution never auto-promotes. Always requires:

```bash
python pxos_shim.py promote <cartridge> --approved-by <guardian>
```

### 2. Instant Rollback

If promoted version has issues:

```bash
python pxos_shim.py rollback <old-cartridge>
# < 5 seconds to revert
```

### 3. History Preserved

All versions kept forever (Genesis §3):

```bash
python pxos_shim.py lineage
# Shows complete ancestry
```

### 4. Genesis Validation

Every cartridge must pass Genesis tests:

```bash
python pxos_shim.py test --cartridge <name>
# 27 Genesis compliance tests
```

---

## Complete Example Session

```bash
# 1. LLM proposes evolution
$ python -c "
from pixel_llm.core.task_queue import create_world_rebuild_task
create_world_rebuild_task('1.2.0', 'pxos_v1_0_0.pxa',
  'Unified pixel storage eliminates PixelFS/InfiniteMap overlap')
"

# 2. Coaching system processes it
$ python pixel_llm_coach.py
🎓 Coaching system starting (with evolution support)...
📋 Processing: Rebuild pxOS v1.2.0
   Action: world_rebuild
🌍 Evolution task detected
✅ Guardrails passed: Guardrails satisfied - rebuild allowed
🌍 Executing WORLD_REBUILD...
✅ New cartridge created: pxos_v1_2_0.pxa

# 3. Human reviews
$ python pxos_shim.py test --cartridge pxos_v1_2_0.pxa
✅ pxos_v1_2_0.pxa is Genesis compliant

$ python pxos_shim.py lineage pxos_v1_2_0.pxa
└─ pxos_v1_0_0.pxa (gen 1)
  🎯 pxos_v1_2_0.pxa (gen 2)

# 4. Human approves
$ python pxos_shim.py promote pxos_v1_2_0.pxa \
    --reason "Cleaner architecture, unified storage"
✅ Promoted pxos_v1_2_0.pxa to current

# 5. System now runs new version
$ python pxos_shim.py run pixel_llm.programs.hello_world:main
🚀 Loading pxOS from: pxos_v1_2_0.pxa
```

---

## Troubleshooting

### Evolution Task Stays Pending

**Cause**: Guardrails blocking or coaching loop not running

**Fix**:
```bash
# Check guardrails
python -c "
from pixel_llm.core.evolution_handler import EvolutionGuardrails
allowed, reason = EvolutionGuardrails.should_allow_world_rebuild({'tests_passing': True})
print(f'{allowed}: {reason}')
"

# Check queue
python pixel_llm/core/task_queue.py list
```

### World Rebuilder Fails

**Cause**: Missing modules or test failures

**Fix**: Check build log in task metadata:
```python
from pixel_llm.core.task_queue import get_queue
task = get_queue().get_task("<task-id>")
print(task.metadata.get("world_rebuild_result", {}).get("build_log"))
```

### Genesis Tests Fail

**Cause**: New cartridge doesn't satisfy Genesis requirements

**Fix**: Review `GENESIS_COMPLIANCE.md` in build workspace:
```bash
cat /tmp/pxos_world_build_X/GENESIS_COMPLIANCE.md
```

---

## Next Steps

1. **Wire into your coaching system**: Add evolution handler to main loop
2. **Test with small change**: Propose minor architectural improvement
3. **Review and promote**: Go through full cycle once
4. **Enable autonomous evolution**: Let LLM check periodically
5. **Monitor and refine**: Adjust guardrails based on experience

---

**Evolution is now fully autonomous (with human approval).** 🎨→🤖→✨

The system can discover better architectures, propose improvements, build them in isolation, test automatically, and present for human review - all while preserving complete history and enabling instant rollback.
