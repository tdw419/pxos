# pxOS Build Tools

Utility-aware build tools for pxOS development.

## Tools

### pxos_orchestrator.py
**Two-phase build orchestrator with utility awareness**

```bash
# Plan a feature (finds relevant utilities)
python3 pxos_orchestrator.py --goal "Add backspace support"

# Find abstraction opportunities
python3 pxos_orchestrator.py --refactor
```

**What it does**:
- Phase A: Searches utility library for matches
- Phase B: Generates LLM prompt with utility context
- Prefers composition over raw code generation

---

### ai_primitive_generator.py
**Generate and convert primitives with utility support**

```bash
# List available utilities
python3 ai_primitive_generator.py --list-utilities

# Generate LLM prompt for a goal
python3 ai_primitive_generator.py "Add clear command"

# Convert JSON feature to primitives
python3 ai_primitive_generator.py --convert feature.json
```

**What it does**:
- Loads utility library at generation time
- Expands `UTILITY_CALL` primitives into `WRITE` commands
- Handles address allocation for inline utilities

---

### ai_utility_builder.py
**Create new utilities from patterns or manually**

```bash
# Analyze primitives for repeated patterns
python3 ai_utility_builder.py --analyze

# Extract pattern N as a utility
python3 ai_utility_builder.py --from-pattern 2

# Interactive utility creator wizard
python3 ai_utility_builder.py --create
```

**What it does**:
- Detects repeated primitive sequences
- Creates utility JSON definitions
- Updates `LIB_MANIFEST.json` automatically

---

## Workflow

### 1. Plan your feature
```bash
python3 pxos_orchestrator.py --goal "Add help command"
```

**Output**:
- Lists matching utilities
- Generates LLM prompt
- Recommends approach (compose vs. implement)

### 2. Send prompt to LLM
Copy the generated prompt to your LLM (LM Studio, Claude, GPT, etc.)

### 3. Save LLM response as JSON
```json
{
  "feature": "Help command",
  "primitives": [
    {"type": "UTILITY_CALL", "name": "util_print_string"}
  ]
}
```

### 4. Convert to primitives
```bash
python3 ai_primitive_generator.py --convert help.json > help_primitives.txt
```

### 5. Integrate and build
```bash
cat help_primitives.txt >> ../pxos_commands.txt
python3 ../build_pxos.py
```

---

## Examples

### Example 1: Use Existing Utility
```bash
# Goal: "Display welcome message"
python3 pxos_orchestrator.py --goal "Display welcome message"

# Finds: util_print_string, util_clear_screen
# LLM generates: UTILITY_CALL util_print_string
# Convert and build
```

### Example 2: Create New Utility
```bash
# Analyze existing code for patterns
python3 ai_utility_builder.py --analyze

# Found: cursor_move pattern (8 bytes)
python3 ai_utility_builder.py --from-pattern 1

# Created: lib/util_cursor_move.json
# Now available for future features
```

### Example 3: Refactor Pass
```bash
# Find repeated code
python3 pxos_orchestrator.py --refactor

# Suggests: 3 patterns can be abstracted
# Extract them as utilities
# Future builds will use utilities instead
```

---

## Integration with AI Loop

### Phase A: Planning (Do we have a utility?)
1. Parse goal
2. Search utility library
3. Recommend: use utility vs. write new code

### Phase B: Implementation (Generate with utilities)
1. Load matching utility definitions
2. Generate LLM prompt with contracts
3. LLM prefers `UTILITY_CALL` over raw primitives
4. Convert JSON to primitives

### Phase C: Refactoring (Grow the library)
1. Detect repeated patterns
2. Propose new utilities
3. Update library
4. Future builds benefit

---

## File Formats

### Input: Goal (Plain Text)
```
"Add backspace support to the shell"
```

### Output: LLM Prompt (Markdown)
```
Implement: Add backspace support

Available utilities:
• util_backspace_lineedit (18 bytes)
  ...

Prefer UTILITY_CALL over WRITE primitives.
```

### LLM Response: Feature (JSON)
```json
{
  "feature": "Backspace support",
  "primitives": [
    {"type": "UTILITY_CALL", "name": "util_backspace_lineedit"}
  ]
}
```

### Final Output: Primitives (pxOS Commands)
```
COMMENT === util_backspace_lineedit ===
WRITE 0x7E00 0xB4    COMMENT MOV AH, 0x03
...
```

---

## Dependencies

- Python 3.7+
- JSON support (standard library)
- pathlib (standard library)

No external dependencies required.

---

## See Also

- [../lib/README.md](../lib/README.md) - Utility library reference
- [../docs/ABSTRACTION_UTILITIES.md](../docs/ABSTRACTION_UTILITIES.md) - Complete documentation
- [../docs/primitives.md](../docs/primitives.md) - Primitive command reference
