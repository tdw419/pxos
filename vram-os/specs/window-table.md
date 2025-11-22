# Window Table Specification

## Overview

The **Window Table** is a VRAM-resident data structure that stores metadata for all active windows in the system. It is managed by the Window Manager (`pxl_wm.px`) and read by the Hardware Abstraction Compositor (HAC) for rendering.

## Location and Size

```
VRAM Address: 0x00100000 (1MB offset)
Size:         512 bytes (128 pixels)
Max Windows:  32 entries
Entry Size:   16 bytes (4 pixels) per window
```

## Entry Format

Each window entry consists of **4 consecutive pixels** (16 bytes total):

### Pixel 0: Position and Identity

```
┌─────────┬─────────┬─────────┬─────────┐
│   Red   │  Green  │  Blue   │  Alpha  │
├─────────┼─────────┼─────────┼─────────┤
│ Window  │ X-pos   │ Y-pos   │  Flags  │
│   ID    │ (0-255) │ (0-255) │         │
└─────────┴─────────┴─────────┴─────────┘

Flags (Alpha byte):
  Bit 0:   Visible (1=shown, 0=hidden)
  Bit 1:   Focused (1=has focus, 0=unfocused)
  Bit 2:   Decorated (1=has title bar, 0=borderless)
  Bit 3:   Resizable (1=can resize, 0=fixed)
  Bit 4:   Movable (1=can move, 0=locked)
  Bit 5:   Modal (1=blocks other windows)
  Bit 6:   Transparent (1=alpha blend)
  Bit 7:   Reserved
```

### Pixel 1: Dimensions and Layering

```
┌─────────┬─────────┬─────────┬─────────┐
│   Red   │  Green  │  Blue   │  Alpha  │
├─────────┼─────────┼─────────┼─────────┤
│  Width  │ Height  │ Z-order │  State  │
│ (0-255) │ (0-255) │ (0-255) │         │
└─────────┴─────────┴─────────┴─────────┘

Width/Height: In 8-pixel units (0=8px, 255=2040px)
Z-order: Stacking order (0=bottom, 255=top)

State (Alpha byte):
  0x00: Normal
  0x01: Minimized
  0x02: Maximized
  0x03: Fullscreen
  0x04: Closing
  0xFF: Inactive/Free slot
```

### Pixel 2: Title String Pointer

```
┌─────────┬─────────┬─────────┬─────────┐
│   Red   │  Green  │  Blue   │  Alpha  │
├─────────┼─────────┼─────────┼─────────┤
│ Title   │ Title   │ Title   │ Title   │
│ Ptr     │ Ptr     │ Length  │ Encoding│
│ [31:24] │ [23:16] │ (0-255) │         │
└─────────┴─────────┴─────────┴─────────┘

Title Encoding (Alpha byte):
  0x00: ASCII
  0x01: UTF-8
  0x02: UTF-16
```

### Pixel 3: Content Buffer Pointer

```
┌─────────┬─────────┬─────────┬─────────┐
│   Red   │  Green  │  Blue   │  Alpha  │
├─────────┼─────────┼─────────┼─────────┤
│ Content │ Content │ Owner   │ Perms   │
│ Ptr     │ Ptr     │ PID     │         │
│ [31:24] │ [23:16] │ (0-255) │         │
└─────────┴─────────┴─────────┴─────────┘

Permissions (Alpha byte):
  Bit 0-2: Access level (0=public, 7=private)
  Bit 3:   Writable by other processes
  Bit 4:   Readable by other processes
  Bit 5-7: Reserved
```

## Table Layout in VRAM

```
Address    | Entry | Window ID | Purpose
-----------|-------|-----------|------------------
0x100000   | 0     | 0 or N/A  | System/Desktop
0x100010   | 1     | 1         | User window 1
0x100020   | 2     | 2         | User window 2
...        | ...   | ...       | ...
0x1001F0   | 31    | 31        | User window 31

Total: 32 entries × 16 bytes = 512 bytes
```

## Window Lifecycle

### 1. Creating a Window

```
Algorithm:
1. Scan Window Table for free slot (State=0xFF)
2. Allocate VRAM for content buffer
3. Write entry metadata:
   - Assign Window ID
   - Set position (X, Y)
   - Set dimensions (W, H)
   - Set initial Z-order
   - Mark as Visible
4. Allocate title string in VRAM
5. Update content pointer
6. Return Window ID to caller
```

### 2. Moving a Window

```
Algorithm:
1. Locate entry by Window ID
2. Read current Flags (Pixel 0, Alpha)
3. If Movable=0, abort
4. Update X-pos (Pixel 0, Green)
5. Update Y-pos (Pixel 0, Blue)
6. Mark dirty region for compositor
```

### 3. Resizing a Window

```
Algorithm:
1. Locate entry by Window ID
2. Check Resizable flag
3. If resizable:
   a. Update Width (Pixel 1, Red)
   b. Update Height (Pixel 1, Green)
   c. Reallocate content buffer if needed
   d. Update content pointer (Pixel 3)
```

### 4. Closing a Window

```
Algorithm:
1. Locate entry by Window ID
2. Set State = 0x04 (Closing)
3. Send close event to owner process
4. Wait for confirmation
5. Free content buffer
6. Free title string
7. Set State = 0xFF (Free)
8. Compact Z-order values
```

## Z-Order Management

### Rules
1. Lower Z-order = drawn first (background)
2. Higher Z-order = drawn last (foreground)
3. Focused window gets highest Z-order
4. Modal windows always above non-modal

### Bringing Window to Front

```python
def bring_to_front(window_id):
    # Find max Z-order
    max_z = max(entry.z_order for entry in window_table
                if entry.state != 0xFF)

    # Set this window's Z-order to max + 1
    window_table[window_id].z_order = min(max_z + 1, 255)

    # Compress Z-order values if approaching limit
    if max_z >= 250:
        compress_z_orders()
```

## Compositor Integration

The HAC (Hardware Abstraction Compositor) reads the Window Table every frame:

```wgsl
// HAC Pseudo-code
@compute @workgroup_size(1)
fn composite_windows() {
    // 1. Sort windows by Z-order
    var sorted_windows = sort_by_z_order(window_table);

    // 2. For each window (back to front)
    for window in sorted_windows {
        if window.visible {
            // 3. Read content from window's buffer
            let content_addr = window.content_ptr;
            let content = load_texture(content_addr);

            // 4. Blit to display at (X, Y)
            blit_region(
                src: content,
                dst: display_buffer,
                x: window.x_pos,
                y: window.y_pos,
                w: window.width * 8,
                h: window.height * 8
            );

            // 5. Draw decorations if needed
            if window.decorated {
                draw_title_bar(window);
                draw_border(window);
            }
        }
    }
}
```

## IPC and Events

### Window Events

Windows receive events through a **VRAM event queue**:

```
Event Queue Address: 0x02000000 + (PID * 0x1000)
Event Size: 16 bytes (4 pixels)
Queue Depth: 256 events

Event Format (4 pixels):
  Pixel 0: Event type, Window ID, timestamp
  Pixel 1: X, Y coordinates (for mouse events)
  Pixel 2: Key code, modifiers (for keyboard events)
  Pixel 3: Event data (varies by type)
```

### Event Types

```
0x01: Window Created
0x02: Window Destroyed
0x03: Window Moved
0x04: Window Resized
0x05: Window Focus Gained
0x06: Window Focus Lost
0x10: Mouse Enter
0x11: Mouse Leave
0x12: Mouse Move
0x13: Mouse Button Down
0x14: Mouse Button Up
0x20: Key Down
0x21: Key Up
0x30: Paint Request
```

## Performance Optimization

### Dirty Region Tracking

To avoid re-compositing the entire screen:

```
Dirty Region Table: 0x00100200 (512 bytes after Window Table)
Format: 32 rectangles × 16 bytes each

Each rectangle:
  Pixel 0: X-min, Y-min
  Pixel 1: X-max, Y-max
  Pixel 2: Associated Window ID
  Pixel 3: Timestamp
```

### Compositor Strategy

1. **Full redraw** if > 25% of screen is dirty
2. **Partial redraw** for small updates
3. **Direct blit** for single-window moves

## Example: Window Manager in Pixel ISA

```asm
; pxl_wm_init.pxl - Initialize Window Table

START:
    ; Clear Window Table
    LOADI R1, #WINDOW_TABLE_BASE
    LOADI R2, #32              ; 32 entries
    LOADI R3, #0xFF            ; Free state marker

clear_loop:
    ; Write 16 bytes (4 pixels) per entry
    STORE [R1 + 0], #0
    STORE [R1 + 4], #0
    STORE [R1 + 8], #0
    STORE [R1 + 12], R3        ; State = 0xFF

    LOADI R4, #16
    ADD R1, R4                 ; Next entry
    LOADI R4, #1
    SUB R2, R4
    CMP R2, #0
    JGT clear_loop

    ; Initialize desktop window (entry 0)
    LOADI R1, #WINDOW_TABLE_BASE
    STORE [R1 + 0], #0x00      ; Window ID = 0
    STORE [R1 + 1], #0         ; X = 0
    STORE [R1 + 2], #0         ; Y = 0
    STORE [R1 + 3], #0x01      ; Visible
    STORE [R1 + 4], #255       ; Width = 255 (2040px)
    STORE [R1 + 5], #255       ; Height = 255
    STORE [R1 + 6], #0         ; Z = 0 (bottom)
    STORE [R1 + 7], #0x00      ; Normal state

    HALT

WINDOW_TABLE_BASE: EQU 0x00100000
```

## Future Enhancements

1. **Hardware acceleration** for alpha blending
2. **GPU-side occlusion culling**
3. **Animated transitions** (fade, slide, scale)
4. **3D window stacking** with perspective
5. **Multi-monitor support** (extended Window Table)

---

**Window Table**: The heart of VRAM OS's GPU-native window management.
