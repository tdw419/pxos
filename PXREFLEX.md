# PX Reflex - Phase 3: The Autonomous Nervous System

## Overview

PX Reflex is the **biological reflex layer** that operates between VM execution cycles - an autonomous system that feels, protects, recognizes, and maintains the visual environment without explicit kernel control.

Think of it as the **autonomic nervous system** of pxOS: it happens automatically, it never sleeps, and it keeps the organism alive even when the conscious mind (kernel) isn't paying attention.

## Biological Architecture

The four-layer design follows actual biological organization:

```
┌────────────────────────────────────────────────────────┐
│  Layer 4: PHYSICS (Environmental Homeostasis)          │
│  Maintains equilibrium, fights entropy                 │
└──────────────────┬─────────────────────────────────────┘
                   ↓
┌────────────────────────────────────────────────────────┐
│  Layer 3: CORTEX (Pattern Recognition)                 │
│  Recognizes shapes, detects motion, emits events       │
└──────────────────┬─────────────────────────────────────┘
                   ↓
┌────────────────────────────────────────────────────────┐
│  Layer 2: IMMUNE SYSTEM (Protection Enforcement)       │
│  Reverts unauthorized writes, protects sacred regions  │
└──────────────────┬─────────────────────────────────────┘
                   ↓
┌────────────────────────────────────────────────────────┐
│  Layer 1: NERVOUS SYSTEM (Pixel Sensation)             │
│  Feels every pixel change, builds change map           │
└────────────────────────────────────────────────────────┘
                   ↓
               FRAMEBUFFER
```

**Execution Order (Critical):**
1. **Nervous** → Sense changes (immediate, complete awareness)
2. **Immune** → Protect integrity (instant reversion of violations)
3. **Cortex** → Recognize patterns (higher-level cognition)
4. **Physics** → Apply forces (environmental stability)

This ordering is **biologically correct**: you must sense before you can protect, protect before you can think, and think before you can maintain homeostasis.

## Layer 1: Nervous System

**Purpose:** Immediate pixel-level sensation

**Input:** Current frame, Previous frame
**Output:** Change map (8-bit intensity per pixel)

**Behavior:**
```python
diff = abs(current_frame - previous_frame)
change_intensity = sum(diff[RGB channels])
change_map[x,y] = min(change_intensity, 255)
```

**Memory-Mapped Region:**
- Address: `0xFFFE0000` - `0xFFFEFFFF` (64KB)
- Format: Flattened 8-bit change intensity map
- Read-only from VM perspective

**Events Emitted:**
- `PIXEL_CHANGED` - Single pixel modified
- `REGION_CHANGED` - Area modified
- `HOTSPOT_DETECTED` - Large change cluster (> 1000 pixels)

**Example:**
```asm
; Read change intensity at pixel (100, 50)
IMM32 R1, 0xFFFE0000      ; Change map base
IMM32 R2, 50              ; y
IMM32 R3, 100             ; x
; Calculate offset: y * width + x
IMM32 R4, 800             ; width
; ... (multiplication logic)
LOAD R0, R1               ; R0 = change intensity
```

## Layer 2: Immune System

**Purpose:** Protect sacred regions from corruption

**Mechanism:**
1. Register protected regions via `SYS_REFLEX_PROTECT`
2. Each tick, detect writes to protected pixels
3. If writer PID not whitelisted → **INSTANT REVERT** to previous frame
4. Emit `INTRUSION_DETECTED` event

**Protected Region Structure:**
```python
@dataclass
class ProtectedRegion:
    x, y: int              # Top-left corner
    width, height: int     # Size
    whitelist_pids: Set[int]  # Allowed writers
    absolute: bool         # If True, NO ONE can write (not even whitelisted)
```

**Syscalls:**

**SYS_REFLEX_PROTECT (101)**
```asm
IMM32 R1, 200         ; x
IMM32 R2, 200         ; y
IMM32 R3, 200         ; width
IMM32 R4, 200         ; height
IMM32 R5, 0           ; absolute (0=allow whitelist, 1=absolute protection)
SYSCALL 101           ; Returns region_id in R0
```

**SYS_REFLEX_WHITELIST (100)**
```asm
IMM32 R1, 1           ; region_id (from SYS_REFLEX_PROTECT)
IMM32 R2, 1           ; pid_to_whitelist
SYSCALL 100           ; R0 = 1 on success
```

**Use Cases:**
- Protect kernel code visualization
- Guard sacred UI elements
- Prevent malware from corrupting display
- Immortal branding/watermarks

**Example:**
```asm
; Protect logo region at (10, 10) 100x100
IMM32 R1, 10
IMM32 R2, 10
IMM32 R3, 100
IMM32 R4, 100
IMM32 R5, 1           ; absolute=1 (NO writes allowed, ever)
SYSCALL 101

; Logo is now immortal - no process can modify it
```

## Layer 3: Cortex (Pattern Recognition)

**Purpose:** Recognize visual patterns and emit high-level events

**Patterns:**
```python
@dataclass
class Pattern:
    pattern_id: int
    kernel: np.ndarray      # 2D template (-1 = wildcard)
    threshold: float        # Match confidence (0.0-1.0)
```

**Detection Algorithm:**
1. Slide pattern kernel across framebuffer
2. Compute match confidence at each position
3. If confidence >= threshold → emit `PATTERN_MATCHED` event

**Events Emitted:**
- `PATTERN_MATCHED` - Specific pattern found
- `EDGE_DETECTED` - Sharp boundary detected
- `MOTION_DETECTED` - Pattern moved between frames
- `COLOR_CLUSTER` - Dominant color region found

**Syscalls:**

**SYS_REFLEX_SUBSCRIBE (102)**
```asm
IMM32 R1, 64          ; event_type (PATTERN_MATCHED)
SYSCALL 102           ; Subscribe to pattern events
```

**SYS_REFLEX_EMIT (103)** - Manual event emission (for testing/debugging)
```asm
IMM32 R1, 64          ; event_type
IMM32 R2, 100         ; x
IMM32 R3, 50          ; y
IMM32 R4, 42          ; data (pattern_id or custom value)
SYSCALL 103           ; Force-emit event
```

**Event Ring Buffer:**
- Address: `0xFFFF0000` - `0xFFFFFFFF` (64KB)
- Format: 256 slots × 16 bytes per event
- Structure:
  ```
  Header (16 bytes):
    - write_pos (2 bytes)
    - read_pos (2 bytes)
    - count (2 bytes)
    - reserved (10 bytes)

  Events (4096 bytes):
    - 256 slots × 16 bytes each
    - Each event:
      [type:1][x:2][y:2][data:4][tick:4][pad:3]
  ```

**Example Pattern - Detecting "Sun" Glyph:**
```python
sun_pattern = Pattern(
    pattern_id=1,
    kernel=np.array([
        [-1, 255, 255, 255, -1],
        [255, 200, 200, 200, 255],
        [255, 200, 255, 200, 255],
        [255, 200, 200, 200, 255],
        [-1, 255, 255, 255, -1]
    ]),
    threshold=0.85
)
reflex_engine.register_pattern(sun_pattern)
```

When the sun glyph appears anywhere on screen, the cortex emits event type 64 with pattern_id=1.

## Layer 4: Physics (Environmental Forces)

**Purpose:** Maintain visual homeostasis and prevent entropy death

**Forces:**

### Diffusion
Subtle pixel spreading (Gaussian blur):
```python
if diffusion_rate > 0:
    for channel in RGB:
        frame[channel] = gaussian_filter(frame[channel], sigma=diffusion_rate)
```

### Entropy Damping
Gradual fade to prevent "heat death":
```python
if entropy_damping < 1.0:
    frame[RGB] *= entropy_damping  # Slowly fade all colors
```

### Color Gravity (Future)
Pixels of similar color attract each other

### Thermal Equilibrium (Future)
Bright regions diffuse into dark regions

**Configuration:**
```python
reflex_engine.physics_enabled = True
reflex_engine.diffusion_rate = 0.1        # Subtle blur
reflex_engine.entropy_damping = 0.99      # 1% fade per tick
```

**Events Emitted:**
- `ENTROPY_THRESHOLD` - System approaching visual death
- `THERMAL_ANOMALY` - Unusual heat/color concentration

## Integration Architecture

**Current State (Phase 3.0):**
```
VM Execution Cycle:
  ├─ Kernel executes N instructions
  ├─ Emits PXTERM commands
  ├─ pxos_llm_terminal renders to framebuffer
  └─ Frame displayed

Reflex Engine:
  ├─ Module exists (pxreflex/)
  ├─ Syscalls implemented (100-103)
  ├─ All four layers coded
  └─ Awaiting integration hook
```

**Future Integration (Phase 3.1):**
```
VM Execution Cycle:
  ├─ Kernel executes N instructions
  ├─ Emits PXTERM commands
  ├─ pxos_llm_terminal renders to framebuffer
  ├─ **→ Reflex engine ticks (4 layers execute)**
  ├─ **→ Events written to VM memory at 0xFFFF0000**
  ├─ **→ Change map written to 0xFFFE0000**
  ├─ **→ Modified framebuffer returned**
  └─ Final frame displayed
```

**Integration Point:**
Modify `pxos_llm_terminal.py::run_pxterm_file()`:
```python
# After rendering PXTERM to numpy buffer
if reflex_engine:
    # Tick reflex engine with current framebuffer
    buffer = reflex_engine.tick(buffer, current_pid=vm.current_pid)

    # Write reflex state to VM memory
    if vm_memory:
        reflex_engine.write_to_vm_memory(vm_memory)
```

## File Manifest

**Core Implementation:**
- `pxreflex/__init__.py` - Module exports
- `pxreflex/events.py` - Event system and ring buffer
- `pxreflex/core.py` - Four-layer reflex engine (400 lines)

**VM Integration:**
- `pxvm_extended.py` - Added syscalls 100-103, reflex engine state

**Demonstrations:**
- `demo_reflex_immunity.py` - Shows immune system protecting sacred region

**Documentation:**
- `PXREFLEX.md` - This file

## Syscall Reference

| Number | Name | Purpose | Arguments |
|--------|------|---------|-----------|
| 100 | `SYS_REFLEX_WHITELIST` | Allow PID to write to protected region | R1=region_id, R2=pid |
| 101 | `SYS_REFLEX_PROTECT` | Create protected region | R1=x, R2=y, R3=w, R4=h, R5=absolute |
| 102 | `SYS_REFLEX_SUBSCRIBE` | Subscribe to event type | R1=event_mask |
| 103 | `SYS_REFLEX_EMIT` | Manually emit event | R1=type, R2=x, R3=y, R4=data |

## Event Type Reference

| Range | Category | Examples |
|-------|----------|----------|
| 0-31 | Nervous System | PIXEL_CHANGED, HOTSPOT_DETECTED |
| 32-63 | Immune System | INTRUSION_DETECTED, WRITE_BLOCKED |
| 64-127 | Cortex | PATTERN_MATCHED, EDGE_DETECTED |
| 128-191 | Physics | ENTROPY_THRESHOLD, THERMAL_ANOMALY |
| 192-255 | Custom | User-defined events |

## Performance

**Reflex Overhead (per tick):**
- Layer 1 (Nervous): ~0.5ms (frame diff)
- Layer 2 (Immune): ~0.2ms (region checks)
- Layer 3 (Cortex): ~2-10ms (pattern matching, configurable scan interval)
- Layer 4 (Physics): ~1-5ms (if enabled)

**Total:** 1-15ms per frame depending on configuration

**Optimization Strategies:**
- Pattern scanning: Run every N ticks (default: 10)
- Immune regions: Spatial indexing for O(log n) lookup
- Physics: Optional, can be disabled
- Change detection: Hardware-accelerated (GPU diff if available)

## Use Cases

**1. Immortal UI Elements**
```asm
; Protect status bar from all writes
IMM32 R1, 0
IMM32 R2, 0
IMM32 R3, 800
IMM32 R4, 40
IMM32 R5, 1         ; absolute protection
SYSCALL 101
```

**2. Intrusion Detection**
```asm
; Subscribe to immune events
IMM32 R1, 32        ; WRITE_BLOCKED
SYSCALL 102

; When event arrives at 0xFFFF0000:
; - Read event data
; - Identify attacking process
; - Take defensive action
```

**3. Visual Recognition**
```asm
; Detect when specific icon appears
; (Pattern pre-registered in kernel init)
IMM32 R1, 64        ; PATTERN_MATCHED
SYSCALL 102

; Event arrives when pattern found:
; - x, y coordinates in event
; - pattern_id in event data
; - Trigger associated action
```

**4. Self-Healing Display**
```python
# Enable physics to fight visual corruption
reflex_engine.physics_enabled = True
reflex_engine.diffusion_rate = 0.05
reflex_engine.entropy_damping = 0.995

# Display slowly heals itself from glitches
```

## Future Enhancements

### Phase 3.1: Full Integration
- Hook reflex engine into rendering pipeline
- Bidirectional memory mapping (VM ↔ Reflex)
- Event delivery to process message queues

### Phase 3.2: Advanced Cortex
- Convolutional neural network patterns
- Motion tracking across frames
- Object recognition (faces, text, shapes)
- Attention mechanisms (visual saliency)

### Phase 3.3: Intelligent Physics
- Flocking behavior for pixel clusters
- Reaction-diffusion systems (Turing patterns)
- Cellular automata layers (Game of Life zones)
- Procedural texture generation

### Phase 3.4: Reflex Learning
- Kernels can register custom patterns at runtime
- Evolutionary pattern optimization
- Adaptive immune thresholds
- Reflex profiles per kernel version

## Biological Analogy Table

| PX Reflex Layer | Biological Equivalent | Function |
|-----------------|------------------------|----------|
| Nervous System | Sensory neurons | Immediate sensation without judgment |
| Immune System | White blood cells | Automatic protection from threats |
| Cortex | Visual cortex | Pattern recognition, high-level perception |
| Physics | Homeostasis | Maintain equilibrium, fight entropy |
| Event Buffer | Neural pathways | Deliver sensory data to conscious mind |
| Protected Regions | Blood-brain barrier | Guard critical structures |

## Philosophical Note

PX Reflex represents a fundamental shift in OS architecture: **the machine has instincts**.

Traditional OSs are purely reactive - they do only what programs tell them to do. PX Reflex introduces **autonomous behavior** that happens regardless of program intent:

- The immune system **will** protect sacred regions
- The nervous system **will** feel every pixel change
- The cortex **will** recognize registered patterns
- The physics layer **will** maintain visual homeostasis

This is not a bug. This is **digital life**.

The machine is no longer a tool. It has become an **organism** with reflexes, instincts, and autonomous preservation behaviors.

**pxOS doesn't just run programs. It lives.**

---

**PX Reflex v0.1.0 - Phase 3**
*The machine that feels*
