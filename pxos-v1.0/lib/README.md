# pxOS L1 Utility Library

This directory contains the **L1 Utility Library** – reusable subroutines for building pxOS features.

## Quick Start

```bash
# List all available utilities
python3 ../tools/ai_primitive_generator.py --list-utilities

# Plan a feature using utilities
python3 ../tools/pxos_orchestrator.py --goal "Add help command"

# Create a new utility
python3 ../tools/ai_utility_builder.py --create
```

## Files

- **LIB_MANIFEST.json**: Master catalog of all utilities
- **SCHEMA_EXTENSION.json**: JSON schema for UTILITY_CALL primitive
- **util_*.json**: Individual utility definitions

## Utility Structure

Each utility file contains:
- **name**: Unique identifier (e.g., `util_print_string`)
- **description**: What it does
- **layer**: Always "L1" for utilities
- **contract**: Inputs, outputs, clobbers, side effects
- **implementation**: Primitive sequence that implements it
- **tests**: Test scenarios
- **usage_example**: How to call it

## Current Utilities (v1.0)

| Name | Bytes | Description |
|------|-------|-------------|
| util_print_char | 5 | Print single character |
| util_print_string | 12 | Print null-terminated string |
| util_clear_screen | 17 | Clear entire screen |
| util_print_newline | 11 | Print CR+LF |
| util_read_key | 5 | Read keystroke |

## Adding New Utilities

### Method 1: Automatic Detection
```bash
python3 ../tools/ai_utility_builder.py --analyze
python3 ../tools/ai_utility_builder.py --from-pattern 1
```

### Method 2: Manual Creation
```bash
python3 ../tools/ai_utility_builder.py --create
```

## Contract Model

Every utility must define:

**Inputs**: What register state is expected before call
```json
"inputs": {"char": {"register": "AL"}}
```

**Outputs**: What register state is produced after call
```json
"outputs": {"result": {"register": "AX"}}
```

**Clobbers**: Which registers may be modified
```json
"clobbers": ["AX", "CX"]
```

**Side Effects**: Observable changes beyond registers
```json
"side_effects": ["Prints to screen", "Advances cursor"]
```

## Usage in Features

Instead of writing raw primitives:
```json
{"type": "WRITE", "address": "0x7C00", "value": "0xB4"},
{"type": "WRITE", "address": "0x7C01", "value": "0x0E"},
...
```

Use utility calls:
```json
{"type": "UTILITY_CALL", "name": "util_print_char"}
```

## Documentation

See [ABSTRACTION_UTILITIES.md](../docs/ABSTRACTION_UTILITIES.md) for complete documentation.
