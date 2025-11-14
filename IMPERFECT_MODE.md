# Imperfect Computing Mode - pxOS

## Philosophy

**Nothing in pxOS is allowed to hard-fail. Everything is best-effort.**

pxOS implements "imperfect computing" - a fault-tolerant design where the system degrades gracefully instead of crashing when encountering errors, malformed data, or missing resources.

## Architecture

The imperfect computing surface lives **above the shader layer**:

```
┌─────────────────────────────────────────┐
│ LLM / User Input (potentially garbage)  │
├─────────────────────────────────────────┤
│ PXSCENE Compiler (imperfect)            │ ← Clamps, defaults, skips
├─────────────────────────────────────────┤
│ PXTERM Terminal (imperfect)             │ ← Try/except, continues
├─────────────────────────────────────────┤
│ GPU Terminal (imperfect)                │ ← CPU fallback, blank frames
├─────────────────────────────────────────┤
│ Frozen WGSL Shader (dumb compositor)    │ ← Just blits pixels
└─────────────────────────────────────────┘
```

The shader is frozen and correct. All error handling happens in Python above it.

## Error Handling by Layer

### 1. PXSCENE Compiler (pxscene_compile.py)

**Never crashes. Always produces valid PXTERM output.**

**Safe value extraction:**
- `safe_int()`: Invalid numbers → default value, optionally clamped
- `safe_color()`: Bad colors → clamped to 0-255 range, defaults to white
- `safe_str()`: Any value → string conversion

**Degradation behaviors:**
- Missing canvas dimensions → 800×600 default
- Out-of-range canvas size → clamped to 1-4096
- Unknown operations → warning comment, skip
- Invalid arguments → use defaults (0 for positions, white for colors)
- Negative rectangle sizes → clamped to 0 (invisible but safe)
- Out-of-range color values → clamped to 0-255
- Broken layout children → warning comment, skip child
- Missing required fields → use sensible defaults

**Example:**
```json
{
  "op": "RECT",
  "w": -50,           // ← clamped to 0
  "h": "bad",         // ← defaults to 0
  "color": [300, -10, "blue", 500]  // ← clamped to [255, 255, 255, 255]
}
```
Compiles to: `RECT 0 0 0 0 255 255 255 255` (invisible but doesn't crash)

### 2. PXTERM Terminal (pxos_llm_terminal.py)

**Never crashes. Logs errors and continues.**

**Fault tolerance:**
- File read errors → log error, exit gracefully
- Missing CANVAS → use 800×600 default
- Invalid command syntax → log line number + error, continue
- Unknown commands → warning, skip
- Drawing exceptions → catch all, log, continue
- Final save errors → log error, don't crash

**Example execution:**
```
Line 5: [ERROR] 'RECT 100 100 bad bad 255 0 0' -> ValueError: invalid literal
Line 6: [WARNING] Unknown command 'FOOBAR' - skipped
Line 7: RECT at (50,50) size 100x100
```
Terminal processes all valid commands, skips bad ones.

### 3. GPU Terminal (pxos_gpu_terminal.py)

**Never crashes. Always returns a frame.**

**Fault tolerance:**
- GPU init fails → silent fallback to CPU
- GPU render fails → fallback to CPU
- CPU render fails → return blank frame (or background layer)
- Missing layers → create on demand
- Out-of-bounds drawing → clipped to canvas
- Save errors → log, don't crash

**Render fallback chain:**
1. Try GPU rendering
2. If GPU fails → try CPU rendering
3. If CPU fails → return background layer
4. If background fails → return black frame

**Example:**
```python
# GPU unavailable
term = PxOSTerminalGPU()  # automatically falls back to CPU

# Render always succeeds
frame = term.draw_frame()  # never None, never crashes
```

## Contract: What Must Never Crash

**PXSCENE Compiler:**
- ✅ Compilation always succeeds
- ✅ Always produces valid PXTERM (may include warning comments)
- ✅ Invalid input → clamped/defaulted values

**PXTERM Terminal:**
- ✅ Execution always completes
- ✅ Bad commands are logged and skipped
- ✅ At least one frame is rendered (even if empty)

**GPU Terminal:**
- ✅ `draw_frame()` always returns a numpy array
- ✅ `save_frame()` attempts save but doesn't crash on failure
- ✅ GPU unavailable → automatic CPU fallback

## What Is Allowed to Degrade

**Visual quality:**
- Missing primitives → rendered as nothing (but scene continues)
- Bad colors → default to white or clamped values
- Invalid dimensions → clamped to 0 (invisible)
- Failed text rendering → silently skipped

**Performance:**
- GPU failure → slower CPU rendering
- Large canvases → clamped to 4096×4096 max

**Functionality:**
- Unknown operations → warning, feature ignored
- Missing resources → defaults used
- Malformed layouts → best-effort positioning

## Benefits

**For LLM-generated content:**
- Hallucinated operations → logged and skipped
- Malformed JSON → safe defaults fill gaps
- Invalid parameters → clamped to valid ranges
- Partial scenes → render what's valid

**For development:**
- Debugging → system stays alive despite bugs
- Experiments → broken code degrades gracefully
- GPU issues → automatic fallback to CPU
- Incomplete features → partial functionality still works

**For reliability:**
- Hostile input → can't crash the system
- Missing dependencies → fallback modes activate
- Corrupted state → system continues with defaults
- Resource exhaustion → graceful degradation

## Testing Imperfect Mode

The file `scene_broken.json` demonstrates imperfect mode with intentional errors:

```bash
python pxscene_run.py scene_broken.json
```

This scene contains:
- Invalid canvas dimensions (`"not_a_number"`, `null`)
- Invalid z-index (`"invalid"`)
- Negative rectangle dimensions
- Out-of-range colors (`[300, -100, "blue", 500]`)
- Unknown operations (`UNKNOWN_OP`)
- Invalid color types (`"not_an_array"`)
- Null children in layouts
- Missing required fields

**Result:** Scene compiles and renders successfully with warnings. System never crashes.

## Configuration

**Quiet mode** (suppress GPU warnings):
```python
term = PxOSTerminalGPU(quiet_errors=True)  # default
```

**Force CPU mode** (skip GPU entirely):
```python
term = PxOSTerminalGPU(use_cpu=True)
```

**Strict mode** (disable imperfect mode, for debugging):
```python
run_pxterm_file("scene.pxterm", imperfect=False)  # raises on errors
```

## Design Principles

1. **Never trust input** - LLMs and users generate garbage
2. **Clamp, don't crash** - Invalid values become valid ones
3. **Log, don't throw** - Errors become console messages
4. **Continue, don't halt** - One bad command doesn't stop the pipeline
5. **Degrade gracefully** - Missing features → partial functionality
6. **Fallback chains** - GPU → CPU → background → blank

## Future Extensions

**Potential additions:**
- Probabilistic rendering (random noise as a feature)
- Fuzzy ALUs in shader (imperfect arithmetic)
- Error accumulation metrics (how "broken" was the input?)
- Self-healing state (detect and repair corrupted data)
- Graceful corruption (partial state corruption doesn't destroy system)

## Implementation Status

**v0.2 Imperfect Mode:**
- ✅ Safe value extraction (safe_int, safe_color, safe_str)
- ✅ Compiler never crashes
- ✅ Terminal catches all exceptions
- ✅ GPU automatic fallback to CPU
- ✅ Drawing operations clip/clamp
- ✅ Frame rendering always succeeds
- ✅ Comprehensive error logging
- ✅ Tested with deliberately broken scenes

## Related Concepts

- **Fault-tolerant computing**: Continue despite hardware failures
- **Graceful degradation**: Reduce functionality instead of failing
- **Defensive programming**: Assume input is hostile
- **Best-effort delivery**: Always produce *something*
- **Error surfaces**: Design where errors can/can't occur

---

*"If it can't be perfect, make it imperfect-by-design."*
