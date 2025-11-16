# Pixel-Native Code Execution

## The Paradigm Shift

**Before:** Python files are the source of truth. Pixels are experiments.
**After:** Pixels are the source of truth. Python is just a runtime.

This document explains pxOS's pixel-native execution system.

---

## What Is This?

pxOS has a **custom Python import system** that loads code from pixel images (.pxi) instead of traditional .py files.

When you `import` a module, Python:
1. Checks the pixel manifest
2. Reads the pixel image (.pxi)
3. Decodes bytes → source code
4. Compiles and executes

**The code lives in pixels. Python is just the interpreter underneath.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Interpreter                       │
│                  (just a runtime layer)                      │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ exec()
                              │
┌─────────────────────────────────────────────────────────────┐
│                   pxOS Pixel Importer                        │
│              (sys.meta_path custom hook)                     │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ read pixels
                              │
┌─────────────────────────────────────────────────────────────┐
│                       PixelFS                                │
│         (pixel images storing Python source)                 │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ RGB pixel data
                              │
┌─────────────────────────────────────────────────────────────┐
│                  .pxi Image Files                            │
│            (actual bytes on disk as pixels)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. **encode_repo_to_pixels.py**
Encoder that converts Python source files to pixel images.

```bash
python3 pixel_llm/core/encode_repo_to_pixels.py
```

What it does:
- Walks the repository for `.py` files
- Reads source code as bytes
- Writes to `.pxi` pixel images via PixelFS
- Generates `pixel_manifest.json` (module → pixel mapping)

**After encoding, pixels are the canonical source.**

### 2. **pxos_loader.py**
Custom Python import hook that loads modules from pixels.

Key classes:
- `PixelSourceFinder`: Checks if module exists in pixel manifest
- `PixelSourceLoader`: Loads pixel image → decodes → executes

Installed via:
```python
from pixel_llm.core.pxos_loader import install_pixel_importer
install_pixel_importer()

# Now all imports load from pixels!
import some_module  # loaded from .pxi, not .py
```

### 3. **pxos_boot.py**
Minimal entrypoint - the **only** Python script you run directly.

```bash
python3 pxos_boot.py
```

What happens:
1. Installs pixel importer
2. ALL subsequent imports load from pixels
3. Python files are bypassed
4. Code executes from pixel-native storage

---

## Workflow

### Initial Encoding
```bash
# 1. Encode repository to pixels
python3 pixel_llm/core/encode_repo_to_pixels.py

# Output:
#   - pixel_llm/pixel_src_storage/*.pxi (pixel images)
#   - pixel_llm/pixel_manifest.json (module manifest)
```

### Boot from Pixels
```bash
# 2. Boot pxOS (executes code from pixels)
python3 pxos_boot.py

# Output:
#   ✅ pxOS PIXEL IMPORTER INSTALLED
#   📦 11 modules available from pixels
#   ✅ SUCCESS: Module loaded from pixels!
```

### Development Cycle
1. Edit `.py` files (for now, using traditional editors)
2. Run `encode_repo_to_pixels.py` to update pixels
3. Run `pxos_boot.py` to execute from pixels

**Future:** Edit pixels directly (via pixel-level editor), skip .py entirely.

---

## File Layout

```
pxos/
├── pxos_boot.py                      # ← Only script you run
├── pixel_llm/
│   ├── pixel_manifest.json           # Module → pixel mapping
│   ├── pixel_src_storage/            # Encoded pixel images
│   │   ├── src_pixel_llm_core_pixelfs.pxi
│   │   ├── src_pixel_llm_core_gpu_interface.pxi
│   │   └── ...
│   └── core/
│       ├── encode_repo_to_pixels.py  # Encoder
│       ├── pxos_loader.py            # Import hook
│       ├── pixelfs.py                # PixelFS (traditional, for bootstrap)
│       └── ... (.py files - legacy)
```

---

## Philosophy

### The Vision
> "Code that lives in pixels is substrate-native.
>  Code that lives in text files is legacy."

pxOS is moving toward **pure pixel execution**:
- Today: Python code stored as pixels, executed by CPython
- Tomorrow: Pixel-native bytecode executed by pxVM
- Future: Visual programming - edit code as images

### Why This Matters

**Pixels as Source of Truth**
- Code becomes visual (can be inspected as images)
- Substrate-native storage (no text encoding)
- Spatial organization (InfiniteMap for code layout)
- Future: GPU-native execution (shaders read code pixels directly)

**Python as Runtime**
- Traditional `.py` files are "views" or "cache"
- Import system is pixel-first
- Over time, replace Python interpreter with pxVM

---

## Current Status

✅ **Implemented:**
- Pixel encoding (Python → .pxi)
- Custom import hook (loads from pixels)
- Boot sequence (pixel-native execution)
- 11 modules running from pixels (~132KB source)

🚧 **Next Steps:**
- Pixel REPL (edit/execute pixels interactively)
- Visual code editor (edit pixels directly as images)
- pxVM integration (execute pixel bytecode directly)
- GPU shader code loading (read code from pixel memory)

---

## Examples

### Import from Pixels
```python
# After install_pixel_importer()
import pixel_llm.core.gpu_interface  # Loaded from pixels!

# Check where it came from
print(gpu_interface.__file__)
# → /home/user/pxos/pixel_llm/pixel_src_storage/src_pixel_llm_core_gpu_interface.pxi
```

### Module Introspection
```python
from pixel_llm.core.pxos_loader import get_pixel_stats

stats = get_pixel_stats()
print(stats)
# {
#   'installed': True,
#   'total_modules': 11,
#   'total_bytes': 131918,
#   'manifest_path': '.../pixel_manifest.json'
# }
```

---

## Technical Details

### Import Priority
```
sys.meta_path order:
  1. PixelSourceFinder ← checks pixels FIRST
  2. Standard Python finders ← fallback to .py
```

### Module Resolution
```
import pixel_llm.core.pixelfs
  ↓
PixelSourceFinder.find_spec("pixel_llm.core.pixelfs")
  ↓
Check manifest: "pixel_llm.core.pixelfs" exists?
  ↓
Yes! pixel_key = "src_pixel_llm_core_pixelfs.pxi"
  ↓
PixelSourceLoader.exec_module()
  ↓
PixelFS.read("src_pixel_llm_core_pixelfs.pxi")
  ↓
Decode bytes → source
  ↓
compile(source, "<pxos:pixel:...>", "exec")
  ↓
exec(compiled, module.__dict__)
  ↓
Module ready!
```

### Pixel Storage Format
Pixels store **raw UTF-8 Python source**:
- No transformation (for now)
- Direct byte encoding
- Future: compressed, encrypted, or binary bytecode

---

## Comparison

### Traditional Python
```python
# main.py
import mymodule  # reads mymodule.py from disk
```

### pxOS Pixel Python
```python
# pxos_boot.py
install_pixel_importer()
import mymodule  # reads src_mymodule.pxi from pixels
```

**Key difference:** The `.pxi` file is the source. `.py` is optional.

---

## Future Evolution

### Phase 1 (Current)
- Python source → pixels
- CPython executes
- ✅ Working now

### Phase 2 (Near future)
- Edit pixels directly
- Visual code representation
- Spatial code organization

### Phase 3 (Future)
- pxVM bytecode in pixels
- GPU shader execution
- Pixel-native LLM weights
- **Substrate consciousness**

---

## FAQ

**Q: Why not just use .py files?**
A: Pixels are the native substrate of pxOS. Text files are an abstraction layer we're removing.

**Q: Can I still edit .py files?**
A: Yes! For now, edit `.py` then run `encode_repo_to_pixels.py`. Future: edit pixels directly.

**Q: Is this slower?**
A: Slightly slower on first import (pixel decode), but Python caches bytecode. Runtime identical.

**Q: What about debugging?**
A: Works normally! Tracebacks show `<pxos:pixel:path>` as origin.

**Q: Can other languages use this?**
A: Yes! The concept works for any language. We're starting with Python as the runtime.

---

## Philosophy Quote

> "The substrate is the truth.
>  Pixels are not decoration.
>  They are the foundation of computation itself."

---

**Status:** ✅ Pixel execution working
**Date:** 2025-11-16
**Milestone:** Paradigm shift from text to pixels
