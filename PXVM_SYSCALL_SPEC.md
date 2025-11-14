# PXVM Syscalls v0.1

Goal: give pxVM a **minimal, safe interface** to the PXOS UI without dragging it into graphics details.

For v0.1, syscalls are **integer-based** and generate **PXTERM lines**.
The host collects these lines and runs them through the existing terminal pipeline:

```text
pxVM bytecode
    ↓ (syscalls)
PXTERM lines
    ↓
pxos_llm_terminal.py
    ↓
PxOSTerminalGPU (layers/text/console)
    ↓
WGSL compositor → GPU / PNG
```

We keep v0.1 deliberately simple:

* no raw pointers
* no arbitrary strings
* IDs instead of direct text/colors

Higher versions (v0.2+) can add memory/string-based syscalls.

---

## 1. Syscall Mechanism

### 1.1 New opcode: SYSCALL

* Opcode name: `SYSCALL`
* Opcode byte: **0xF0** (or the next free opcode in pxVM)
* Encoding:

```text
F0 nn
```

where:

* `F0` = SYSCALL opcode
* `nn` = 1-byte syscall number (0–255)

### 1.2 Calling convention

When pxVM executes `SYSCALL nn`, the host reads:

* `R0` – syscall number (mirror of `nn`, for consistency)
* `R1..R7` – integer arguments (meaning depends on syscall)

The host then:

1. Interprets `(nn, R1..R7)` according to the syscall table.
2. Generates **0 or more PXTERM lines** into a list `sysout`.
3. Returns control to the VM (no values returned in v0.1).

If an unknown syscall is invoked in Imperfect Mode:

* host MUST NOT crash,
* host SHOULD append a PXTERM comment like:

```text
# WARNING: unknown syscall nn with args R1..R7
```

---

## 2. Syscall Table v0.1

### 2.1 SYS_PRINT_ID (1)

**Purpose:** write a predefined message into the PXOS console.

* `nn = 1`
* Args:

  * `R1` = `message_id` (integer)

Host behavior:

1. Look up `message_id` in a host-side mapping:

   ```python
   SYS_MESSAGES = {
       1: "PXVM booting...",
       2: "PXVM ready.",
       3: "Task complete.",
       # extendable
   }
   ```

2. If found, emit:

   ```text
   PRINT PXVM: PXVM booting...
   ```

3. If not found, emit:

   ```text
   PRINT [vm warn] unknown message_id 42
   ```

No crash in Imperfect Mode.

---

### 2.2 SYS_RECT_ID (2)

**Purpose:** draw a rectangle in the current layer using a predefined color.

* `nn = 2`
* Args:

  * `R1` = x
  * `R2` = y
  * `R3` = w
  * `R4` = h
  * `R5` = `color_id`

Host behavior:

1. Look up `color_id`:

   ```python
   SYS_COLORS = {
       1: (40, 40, 100, 255),   # window frame
       2: (20, 20, 60, 255),    # title bar
       3: (0, 0, 40, 255),      # background
       # etc.
   }
   ```

2. Emit:

   ```text
   RECT {R1} {R2} {R3} {R4} r g b a
   ```

   using the mapped RGBA; if `color_id` unknown, use a fallback (e.g. magenta) and a comment:

   ```text
   # WARNING: unknown color_id 9, using fallback
   ```

---

### 2.3 SYS_TEXT_ID (3)

**Purpose:** draw a short label at a given position, using a color and a predefined message.

* `nn = 3`
* Args:

  * `R1` = x
  * `R2` = y
  * `R3` = `color_id`
  * `R4` = `message_id`

Host behavior:

1. Look up `color_id` in `SYS_COLORS` and `message_id` in `SYS_MESSAGES`.

2. Emit:

   ```text
   TEXT {R1} {R2} r g b a PXVM booting...
   ```

3. If IDs invalid, log via PRINT or comment, but do not crash.

---

### 2.4 SYS_LAYER_USE_ID (4) – optional but useful

**Purpose:** switch the current drawing layer before other syscalls.

* `nn = 4`
* Args:

  * `R1` = `layer_id` (integer)

Host behavior:

1. Map `layer_id` to a layer name:

   ```python
   SYS_LAYERS = {
       1: "background",
       2: "ui",
       3: "vm",
       4: "overlay",
   }
   ```

2. Emit:

   ```text
   SELECT vm
   ```

   for `layer_id = 3`.

3. If unknown, emit:

   ```text
   PRINT [vm warn] unknown layer_id 7
   ```

---

## 3. Host Output Format

Syscalls **do not** directly call GPU APIs. They only append to a list:

```python
sysout: list[str]  # PXTERM lines
```

Example after a pxVM run:

```text
SELECT vm
RECT 120 140 200 40 40 80 40 255
TEXT 130 150 255 255 255 PXVM booting...
PRINT PXVM: Kernel init done.
```

The host can then:

1. Save `sysout` to `program.pxterm` and call `pxos_llm_terminal.py program.pxterm`, **or**
2. Directly stream each line into the existing PXTERM executor in-process.

---

## 4. Imperfect Mode Behavior

* Unknown syscall IDs: generate a comment or PRINT line, never crash.
* Out-of-range coords/widths/heights: will be clipped at the drawing layer (per Imperfect Mode spec).
* Bad message/color IDs: log a warning, fall back to safe defaults.

Syscalls obey the same **Imperfect Computing** philosophy:
"**Always prefer degraded output + logs over failure.**"

---

## 5. Example VM Program

Pseudo-assembly showing syscall usage:

```asm
; Switch to VM layer
IMM32 R1, 3        ; layer_id = 3 ("vm")
SYSCALL 4          ; SYS_LAYER_USE_ID

; Draw window frame
IMM32 R1, 150      ; x
IMM32 R2, 150      ; y
IMM32 R3, 300      ; w
IMM32 R4, 80       ; h
IMM32 R5, 1        ; color_id = 1 (window frame)
SYSCALL 2          ; SYS_RECT_ID

; Draw title bar
IMM32 R1, 150
IMM32 R2, 150
IMM32 R3, 300
IMM32 R4, 30
IMM32 R5, 2        ; color_id = 2 (title bar)
SYSCALL 2

; Write title text
IMM32 R1, 170      ; x
IMM32 R2, 160      ; y
IMM32 R3, 1        ; color_id (white/light)
IMM32 R4, 2        ; message_id = "PXVM ready."
SYSCALL 3          ; SYS_TEXT_ID

; Console message
IMM32 R1, 1        ; message_id = "PXVM booting..."
SYSCALL 1          ; SYS_PRINT_ID

HALT
```

---

## 6. Future Extensions (v0.2+)

* **SYS_PRINT_STR**: Print arbitrary string from VM memory
* **SYS_TEXT_STR**: Render text from VM memory
* **SYS_LAYER_CREATE**: Dynamically create named layers
* **SYS_DRAW**: Trigger frame render/flush
* **SYS_INPUT**: Read keyboard/mouse (for interactive VMs)
* **SYS_PROCESS**: Process management syscalls

For v0.1, ID-based syscalls provide a safe, simple foundation.
