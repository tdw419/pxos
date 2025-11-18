# pxOS Abstraction Utilities Layer

## Overview

The pxOS Abstraction Utilities Layer is a three-tier architecture that transforms raw bootloader development into a composable, reusable system:

- **L0 – Primitives**: Raw `WRITE/DEFINE/COMMENT` commands (existing)
- **L1 – Utilities**: Reusable subroutines with contracts (new)
- **L2 – Features**: High-level functionality built by composing L1 utilities (new)

This system treats **abstractions as first-class products**, not afterthoughts.

---

## Philosophy

### The Problem

Without abstractions, every feature requires writing bootloader assembly from scratch:
- Repeated boilerplate code
- Inconsistent implementations
- Hard to maintain and evolve
- AI systems waste tokens re-implementing the same patterns

### The Solution

Build a **standard library for pxOS** where:
- Common operations become reusable utilities
- Utilities have well-defined contracts (inputs, outputs, side effects)
- AI generators prefer composition over raw code generation
- The system automatically identifies abstraction opportunities

---

## Architecture

### Layer 0: Primitives

Raw memory manipulation commands:
```
WRITE 0x7C00 0xFA    COMMENT CLI instruction
DEFINE label 0x7C00  COMMENT Symbol definition
```

### Layer 1: Utilities

Reusable subroutines with contracts:
```json
{
  "name": "util_print_string",
  "contract": {
    "inputs": {"string_ptr": {"register": "SI"}},
    "outputs": {"string_ptr": {"register": "SI"}},
    "clobbers": ["AL", "AH"]
  }
}
```

### Layer 2: Features

High-level functionality built from utilities:
```json
{
  "feature": "Display welcome banner",
  "primitives": [
    {"type": "UTILITY_CALL", "name": "util_clear_screen"},
    {"type": "UTILITY_CALL", "name": "util_print_string"},
    {"type": "UTILITY_CALL", "name": "util_print_newline"}
  ]
}
```

---

## Directory Structure

```
pxos-v1.0/
├── lib/                          # L1 Utility Library
│   ├── LIB_MANIFEST.json         # Master catalog of utilities
│   ├── SCHEMA_EXTENSION.json    # JSON schema for UTILITY_CALL
│   ├── util_print_char.json     # Individual utility definitions
│   ├── util_print_string.json
│   ├── util_clear_screen.json
│   ├── util_print_newline.json
│   └── util_read_key.json
├── tools/                        # Build tools
│   ├── ai_primitive_generator.py    # Utility-aware code generator
│   ├── ai_utility_builder.py        # Create new utilities
│   └── pxos_orchestrator.py         # Two-phase build orchestrator
└── docs/
    └── ABSTRACTION_UTILITIES.md  # This file
```

---

## Available Utilities (v1.0)

| Utility | Size | Category | Description |
|---------|------|----------|-------------|
| `util_print_char` | 5 bytes | output | Print single character (AL) |
| `util_print_string` | 12 bytes | output | Print null-terminated string (SI) |
| `util_clear_screen` | 17 bytes | output | Clear screen with spaces |
| `util_print_newline` | 11 bytes | output | Print CR+LF sequence |
| `util_read_key` | 5 bytes | input | Read keystroke (returns in AL) |

**Total library size**: 50 bytes

---

## Utility Contract Model

Every utility defines a contract:

### Inputs
What the utility expects (register state before call):
```json
"inputs": {
  "string_ptr": {"register": "SI", "description": "Pointer to string"}
}
```

### Outputs
What the utility produces (register state after call):
```json
"outputs": {
  "char": {"register": "AL", "description": "Character read"}
}
```

### Clobbers
Registers the utility may modify:
```json
"clobbers": ["AX", "BX", "CX"]
```

### Side Effects
Observable changes beyond register state:
```json
"side_effects": [
  "Prints characters to screen",
  "Advances cursor position"
]
```

---

## Workflow

### 1. Planning Phase (Phase A)

When you have a goal (e.g., "Add help command"), the orchestrator:

```bash
python3 tools/pxos_orchestrator.py --goal "Add help command"
```

**Output**:
- Searches `LIB_MANIFEST.json` for matching utilities
- Lists relevant utilities (e.g., `util_print_string`, `util_print_newline`)
- Recommends approach: compose utilities vs. write new code

### 2. Implementation Phase (Phase B)

The orchestrator generates an LLM prompt:
- Lists available utilities and their contracts
- Emphasizes preference for `UTILITY_CALL` over raw `WRITE`
- Requests JSON output in the extended schema

**You send this prompt to your LLM** (LM Studio, Claude, etc.)

### 3. Conversion Phase

LLM returns JSON feature definition:
```json
{
  "feature": "Help command",
  "primitives": [
    {"type": "UTILITY_CALL", "name": "util_print_string", "comment": "Print help text"}
  ]
}
```

Convert to primitives:
```bash
python3 tools/ai_primitive_generator.py --convert feature.json > new_primitives.txt
```

**Output**: Expands `UTILITY_CALL` into actual `WRITE` commands

### 4. Integration Phase

Append to your commands file:
```bash
cat new_primitives.txt >> pxos_commands.txt
python3 build_pxos.py
```

---

## Creating New Utilities

### Automatic Discovery

Detect repeated patterns:
```bash
python3 tools/ai_utility_builder.py --analyze
```

**Output**:
```
Found 3 potential utility patterns:
1. print_char (5 bytes) at 0x7C40
2. cursor_move (8 bytes) at 0x7D00
3. string_compare (15 bytes) at 0x7E00
```

Extract a pattern:
```bash
python3 tools/ai_utility_builder.py --from-pattern 2
```

Creates `lib/util_cursor_move.json` and updates manifest.

### Manual Creation

Interactive wizard:
```bash
python3 tools/ai_utility_builder.py --create
```

Prompts for:
- Utility name
- Description
- Contract (inputs, outputs, clobbers)
- Primitive implementation

---

## Refactoring Mode

Scan codebase for abstraction opportunities:
```bash
python3 tools/pxos_orchestrator.py --refactor
```

**Output**:
- Identifies repeated primitive sequences
- Suggests utilities to extract
- Estimates size savings from centralization

---

## Integration with AI Build Loop

### Before (Pure Primitives)

```
Goal: "Add backspace support"
  ↓
LLM generates raw WRITE primitives
  ↓
20+ lines of assembly-level code
  ↓
Hard to reuse
```

### After (Utility-Aware)

```
Goal: "Add backspace support"
  ↓
Orchestrator: "Found util_backspace_lineedit"
  ↓
LLM generates: UTILITY_CALL util_backspace_lineedit
  ↓
Converter expands to primitives
  ↓
Single line, reusable across features
```

---

## Example: Building a Feature

### Goal
"Display a help menu with three commands"

### Step 1: Plan
```bash
python3 tools/pxos_orchestrator.py --goal "Display help menu"
```

**Finds**:
- `util_clear_screen`
- `util_print_string`
- `util_print_newline`

### Step 2: Generate (LLM)

Prompt includes utility contracts. LLM returns:
```json
{
  "feature": "Help menu display",
  "primitives": [
    {"type": "UTILITY_CALL", "name": "util_clear_screen"},
    {"type": "UTILITY_CALL", "name": "util_print_string", "comment": "Print 'Commands:'"},
    {"type": "UTILITY_CALL", "name": "util_print_newline"},
    {"type": "UTILITY_CALL", "name": "util_print_string", "comment": "Print 'cls - clear'"},
    {"type": "UTILITY_CALL", "name": "util_print_newline"}
  ]
}
```

### Step 3: Convert
```bash
python3 tools/ai_primitive_generator.py --convert help_feature.json
```

**Output**: 55 bytes of primitives (vs. 100+ if written from scratch)

### Step 4: Build
```bash
python3 build_pxos.py
```

---

## Benefits

### 1. **Reusability**
Write once, use everywhere. `util_print_string` used in:
- Boot banner
- Shell prompt
- Help command
- Error messages

### 2. **Consistency**
All string printing uses the same tested implementation.

### 3. **Maintainability**
Fix a bug in `util_print_string` once, all features benefit.

### 4. **AI Efficiency**
- LLM prompts are shorter (reference utilities, not raw assembly)
- Generation is faster (compose vs. write from scratch)
- Fewer tokens used per feature

### 5. **Automatic Improvement**
Refactor mode discovers abstraction opportunities over time.

---

## Future Enhancements

### Phase 1 (Current)
- ✅ Utility library with 5 core utilities
- ✅ JSON schema extension for `UTILITY_CALL`
- ✅ Generator with utility awareness
- ✅ Orchestrator with two-phase planning
- ✅ Pattern detection for new utilities

### Phase 2 (Next)
- [ ] CALL instruction mode (non-inline utilities)
- [ ] Utility versioning (backward compatibility)
- [ ] Performance profiling (which utilities are hot?)
- [ ] Automatic refactoring (replace duplicates with utilities)

### Phase 3 (Future)
- [ ] Macro utilities (L1.5 layer)
- [ ] Conditional compilation (feature flags)
- [ ] Cross-module utilities (shared across projects)
- [ ] Knowledge base integration (semantic search for utilities)

---

## Command Reference

| Command | Purpose |
|---------|---------|
| `pxos_orchestrator.py --goal "<description>"` | Plan and generate prompts for goal |
| `pxos_orchestrator.py --refactor` | Find abstraction opportunities |
| `ai_primitive_generator.py --list-utilities` | Show available utilities |
| `ai_primitive_generator.py --convert <json>` | Convert JSON to primitives |
| `ai_utility_builder.py --analyze` | Detect repeated patterns |
| `ai_utility_builder.py --create` | Interactive utility creator |
| `ai_utility_builder.py --from-pattern N` | Extract pattern N as utility |

---

## Philosophy: Abstractions as Products

This system treats utilities **not as side effects**, but as **primary deliverables**:

- Every feature implementation should **add to or use** the utility library
- The library **grows organically** as you build
- Repetition is a **signal to abstract**, not a cost to accept
- The AI system is a **collaborator in refactoring**, not just a code generator

**Goal**: Build pxOS like a real OS—with a platform, not just a pile of boot sector bytes.

---

## Getting Started

1. **List available utilities**:
   ```bash
   python3 tools/ai_primitive_generator.py --list-utilities
   ```

2. **Plan a feature**:
   ```bash
   python3 tools/pxos_orchestrator.py --goal "Add cursor movement"
   ```

3. **Create first utility**:
   ```bash
   python3 tools/ai_utility_builder.py --create
   ```

4. **Look for patterns**:
   ```bash
   python3 tools/ai_utility_builder.py --analyze
   ```

---

**Next**: [extensions.md](extensions.md) — Extending the utility library
