# pxOS Evolution Workflow

**How pxOS Improves Itself Without Breaking**

This document explains how LLMs can propose, build, test, and promote improvements to pxOS - all while maintaining stability, auditability, and the ability to roll back.

---

## Overview: The Evolution Cycle

```
┌──────────────┐
│ Current pxOS │  ← User boots from this
│  v1.0.0.pxa  │
└──────┬───────┘
       │
       │ 1. LLM discovers better architecture
       ↓
┌──────────────┐
│  Proposal    │  ← Design doc + reasoning
│  "v1.1.0"    │
└──────┬───────┘
       │
       │ 2. Create WORLD_REBUILD or ARCHITECTURE_CHANGE task
       ↓
┌──────────────┐
│ Build in     │  ← Fresh workspace, LLM generates code
│ Sandbox      │
└──────┬───────┘
       │
       │ 3. Pack into new cartridge
       ↓
┌──────────────┐
│ pxos_v1_1_0  │  ← Status: experimental
│   .pxa       │
└──────┬───────┘
       │
       │ 4. Run Genesis tests
       ↓
┌──────────────┐
│  Genesis     │  ← Compliant? Ready for promotion
│  Validation  │
└──────┬───────┘
       │
       │ 5. Human approval
       ↓
┌──────────────┐
│   Promote    │  ← v1.1.0 becomes current
│ to Current   │
└──────┬───────┘
       │
       │ 6. Boot from new version
       ↓
┌──────────────┐
│ Current pxOS │  ← v1.0.0 preserved as historical
│  v1.1.0.pxa  │
└──────────────┘
```

---

## Phase 1: Proposal

### When LLMs Propose Changes

An LLM (via coaching system or self-inspection) identifies an improvement:

**Examples:**
- "PixelFS could be 10x faster with block-level caching"
- "InfiniteMap should use a different spatial index"
- "We should move more ops to GPU"
- "The entire architecture could be simpler"

### Creating a Proposal

**Option A: Incremental Change** (architecture improvement)

```python
from pixel_llm.core.task_queue import create_architecture_change_task

task_id = create_architecture_change_task(
    change_description="Add block-level caching to PixelFS for 10x speedup",
    affected_modules=["pixel_llm/core/pixelfs.py"],
    priority=80
)
```

**Option B: Full Rebuild** (start from scratch)

```python
from pixel_llm.core.task_queue import create_world_rebuild_task

task_id = create_world_rebuild_task(
    target_version="1.1.0",
    parent_cartridge="pxos_v1_0_0.pxa",
    reason="Simplified architecture with unified pixel storage API",
    priority=100
)
```

### What Happens Next

- Task goes to queue
- Coaching system picks it up
- LLM begins implementation in isolated workspace

---

## Phase 2: Build

### Workspace Isolation

Each new version builds in a fresh directory:

```
/tmp/pxos_world_build_1_1_0/
├── pixel_llm/
│   ├── core/
│   │   ├── pixelfs_v2.py       ← New implementation
│   │   ├── infinite_map.py     ← Modified
│   │   └── hypervisor.py       ← Updated API
│   ├── tests/
│   │   └── genesis/            ← Must pass
│   └── meta/
└── GENESIS_COMPLIANCE.md       ← Maps Genesis → implementation
```

### Build Process

**For WORLD_REBUILD tasks:**

1. **Create workspace**
   ```bash
   mkdir -p /tmp/pxos_world_build_1_1_0
   cd /tmp/pxos_world_build_1_1_0
   ```

2. **Load Genesis + Template**
   ```bash
   cp /path/to/GENESIS_SPEC.md .
   cp /path/to/templates/pxos_world_template.yaml .
   ```

3. **LLM generates each module**
   - Coaching system iterates through modules
   - Gemini reviews → Local LLM generates → Iterate until score ≥ 8/10
   - Each module written to workspace

4. **Generate compliance doc**
   ```markdown
   # GENESIS_COMPLIANCE.md

   ## Pixel Substrate Primacy (§1)
   Implementation: pixel_llm/core/pixelfs_v2.py, lines 50-200
   Proof: All data stored as RGB via PixelFile class

   ## Archive-Based Distribution (§2)
   Implementation: pixel_llm/core/archive_v2.py
   Proof: pack_world.py creates single .pxa file

   ...
   ```

5. **Pack into cartridge**
   ```bash
   python pack_repository.py --output pxos_v1_1_0.pxa
   ```

**For ARCHITECTURE_CHANGE tasks:**

1. **Create design doc** (LLM + Gemini)
2. **Implement proof-of-concept**
3. **Run benchmarks** (old vs new)
4. **Write migration plan**

---

## Phase 3: Test

### Register as Experimental

```bash
python pxos_shim.py status
```

Or via Python:

```python
from pixel_llm.core.cartridge_manager import register_cartridge

register_cartridge(
    name="pxos_v1_1_0.pxa",
    version="1.1.0",
    parent="pxos_v1_0_0.pxa",
    built_by="llm",
    builder_name="pixel_llm_coach",
    notes="Unified pixel storage API with 10x faster PixelFS",
    capabilities=["pixel_storage_v2", "gpu_textures", "..."],
    metrics={"build_time_seconds": 1200, "test_coverage": 85},
    status="experimental"
)
```

### Run Genesis Tests

```bash
python pxos_shim.py test --cartridge pxos_v1_1_0.pxa
```

This runs:
1. **Basic checks**: Archive loads, Python runs, imports work
2. **Genesis compliance**: Each §requirement verified
3. **Full test suite**: All tests from pixel_llm/tests/
4. **Performance benchmarks**: vs parent cartridge

### Test Output

```
🧪 Testing cartridge: pxos_v1_1_0.pxa

🔍 Running Genesis compliance checks...

📊 Results:
   Compliant: ✅ YES
   Genesis version: 1.0
   Tests passed: 47
   Tests failed: 0

✅ pxos_v1_1_0.pxa is Genesis compliant
   Ready for promotion with: python pxos_shim.py promote pxos_v1_1_0.pxa
```

---

## Phase 4: Promotion

### Human Review

Before promotion, a human (guardian) reviews:

1. **Check lineage**
   ```bash
   python pxos_shim.py lineage pxos_v1_1_0.pxa
   ```

2. **Read proposal & reasoning**
   ```bash
   cat /tmp/pxos_world_build_1_1_0/PROPOSAL.md
   ```

3. **Review compliance doc**
   ```bash
   cat /tmp/pxos_world_build_1_1_0/GENESIS_COMPLIANCE.md
   ```

4. **Check test results**
   ```bash
   cat /tmp/pxos_world_build_1_1_0/test_results.json
   ```

### Approve Promotion

```bash
python pxos_shim.py promote pxos_v1_1_0.pxa \
  --approved-by tdw419 \
  --reason "10x performance improvement, all tests pass, Genesis compliant"
```

Or via Python:

```python
from pixel_llm.core.cartridge_manager import promote_cartridge

promote_cartridge(
    name="pxos_v1_1_0.pxa",
    approved_by="tdw419",
    reason="10x performance improvement, Genesis compliant"
)
```

### What Happens

- `pxos_v1_0_0.pxa` status → "historical"
- `pxos_v1_1_0.pxa` status → "current"
- Evolution log updated with reasoning
- Next boot uses v1.1.0

---

## Phase 5: Boot New Version

### Run from New Cartridge

```bash
python pxos_shim.py run pixel_llm.programs.hello_world:main
```

This automatically:
- Loads pxos_v1_1_0.pxa (current)
- Initializes hypervisor
- Runs the program

### Verify Everything Works

```bash
# Run full test suite
python pxos_shim.py run pixel_llm.tests:run_all

# Check GPU works
python pxos_shim.py run pixel_llm.programs.gpu_test:main

# Inspect capabilities
python pxos_shim.py run pixel_llm.core.hypervisor:inspect_self
```

---

## Rollback: When Things Go Wrong

### Immediate Rollback

If v1.1.0 has problems:

```bash
python pxos_shim.py rollback pxos_v1_0_0.pxa \
  --approved-by tdw419 \
  --reason "v1.1.0 has critical bug in PixelFS, reverting"
```

This immediately:
- Sets v1.0.0 back to "current"
- Marks v1.1.0 as "deprecated"
- Logs the rollback reasoning
- Next boot uses v1.0.0

### Rollback is Always Safe

- Old cartridges are NEVER deleted (Genesis §3)
- Can roll back to any historical version
- Evolution log preserved
- No data loss

---

## Advanced: Migration Between Architectures

### When to Use Migration Tasks

When the change is too big for incremental:

**Example**: "Move from Python VM to Rust VM"

### Create Migration Task

```python
from pixel_llm.core.task_queue import create_migration_task

task_id = create_migration_task(
    from_architecture="Python PixelVM (pixel_vm.py)",
    to_architecture="Rust PixelVM (compiled to WASM)",
    migration_plan_path="docs/migrations/python_to_rust_vm.md",
    priority=90
)
```

### Migration Plan Document

```markdown
# Migration: Python VM → Rust VM

## Rationale
- 100x faster execution
- Memory safe
- Can compile to WASM for browser use

## Compatibility Strategy
- Keep Python VM for 2 versions (backward compat)
- Add VM_TYPE field to cartridge manifest
- Hypervisor detects and loads correct VM

## Steps
1. Implement Rust VM with same opcode set
2. Add opcode tests (both VMs must pass)
3. Benchmark: Rust vs Python on standard suite
4. Pack dual-VM cartridge (both included)
5. Promote, monitor for issues
6. Deprecate Python VM in v1.3.0

## Rollback Plan
- If Rust VM fails, hypervisor falls back to Python
- Full rollback possible to v1.0.0 (no Rust dependency)
```

---

## Governance: Who Decides

### Automatic (No Human Needed)

- **Registering experiments**: LLM can create experimental cartridges
- **Running tests**: Automatic Genesis validation
- **Building in sandbox**: LLM has full control

### Requires Human Approval

- **Promoting to "current"**: Human must approve
- **Breaking Genesis**: Not allowed, period
- **Adding Genesis requirements**: Requires guardian consensus

### Guardian Veto

Guardians (currently @tdw419) can:
- Reject any promotion
- Force rollback
- Deprecate cartridges
- Add Genesis requirements

---

## Example Scenarios

### Scenario 1: Performance Optimization

**LLM discovers**: "PixelFS cache is inefficient"

1. Creates `ARCHITECTURE_CHANGE` task
2. Builds proof-of-concept with new caching
3. Runs benchmarks: 10x faster
4. Packs into v1.0.1 cartridge
5. Tests pass
6. Human reviews, approves
7. Promotion → v1.0.1 is now current

**Time**: ~2 hours (mostly LLM generating + testing)

### Scenario 2: Complete Redesign

**LLM proposes**: "Unified pixel storage API (combine PixelFS + InfiniteMap)"

1. Creates `WORLD_REBUILD` task for v2.0.0
2. Generates fresh codebase from Genesis + template
3. Coaching system builds all modules
4. 5,000 lines of new code generated
5. Tests: 90% coverage, Genesis compliant
6. Human reviews design doc + code samples
7. Approves → v2.0.0 promoted
8. v1.x.x preserved as historical

**Time**: ~1 day (LLM generates ~500 lines/hour)

### Scenario 3: Bug Found After Promotion

**Issue**: v1.1.0 has critical bug

1. Guardian runs: `python pxos_shim.py rollback pxos_v1_0_0.pxa`
2. Immediately back to stable v1.0.0
3. v1.1.0 marked "deprecated"
4. LLM creates fix task
5. Builds v1.1.1 with fix
6. Tests, promotes
7. Evolution log shows: v1.0.0 → v1.1.0 (bug) → v1.0.0 (rollback) → v1.1.1 (fixed)

**Time**: <5 minutes to rollback

---

## Key Principles

### 1. Experiments are Cheap
- LLM can create as many experimental cartridges as needed
- Build in isolated workspaces
- No risk to current system

### 2. Promotion is Careful
- Requires tests passing
- Requires Genesis compliance
- Requires human approval (for breaking changes)

### 3. History is Sacred
- Never delete old versions
- Always traceable (who, when, why)
- Always reversible

### 4. Genesis is Immutable
- LLMs can't change core principles
- Only guardians can add Genesis requirements
- Implementations evolve, Genesis stays stable

---

## Tools Summary

### For Humans

```bash
# View status
python pxos_shim.py status

# Run programs
python pxos_shim.py run <program>

# Test cartridge
python pxos_shim.py test --cartridge <name>

# Promote
python pxos_shim.py promote <name>

# Rollback
python pxos_shim.py rollback <name>

# Show lineage
python pxos_shim.py lineage [name]
```

### For LLMs (via Python)

```python
from pixel_llm.core.task_queue import (
    create_world_rebuild_task,
    create_architecture_change_task,
    create_migration_task
)
from pixel_llm.core.cartridge_manager import (
    register_cartridge,
    get_current_cartridge,
    get_cartridge_info
)
from pixel_llm.core.hypervisor import get_hypervisor
```

---

## Next Steps

1. **Create world template**: `templates/pxos_world_template.yaml`
2. **Implement rebuild script**: Coaching system to execute WORLD_REBUILD tasks
3. **Add Genesis test suite**: `pixel_llm/tests/genesis/`
4. **Enable LLM self-inspection**: Let PixelLLM read its own cartridge

---

**Evolution is a feature, not a bug.** 🎨→🤖→✨

pxOS is designed to improve itself, safely and transparently, forever.
