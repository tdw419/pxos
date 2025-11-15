# Oracle Protocol

**How organisms talk to the local LLM in pxOS universes**

## Overview

The Oracle Protocol is a standardized memory-mapped interface that allows organisms (autonomous agents running in PXI_CPU) to communicate with external AI models (LM Studio, Ollama, etc.) through the `SYS_LLM` syscall.

Think of it as:
- **Organisms** = processes running in the pixel universe
- **Oracle** = the local LLM (your AI model)
- **Protocol** = the contract for how they communicate

## Memory Map

```
┌─────────────────────────────────────────────┐
│  0 - 999      Main Program / Kernel         │
├─────────────────────────────────────────────┤
│  1000 - 1999  Organism Code & Data          │
├─────────────────────────────────────────────┤
│  2000 - 7999  Visual Frame Buffer           │
├─────────────────────────────────────────────┤
│  8000 - 8999  PROMPT_BUFFER (Org → Oracle)  │ ← Write questions here
├─────────────────────────────────────────────┤
│  9000 - 9999  RESPONSE_BUFFER (Oracle → Org)│ ← Read answers here
├─────────────────────────────────────────────┤
│  10000        ORACLE_FLAG (Request control) │ ← 1 = pending, 0 = idle
└─────────────────────────────────────────────┘
```

### Constants

```python
PROMPT_BUFFER_ADDR = 8000      # Organisms write questions (null-terminated ASCII)
RESPONSE_BUFFER_ADDR = 9000    # Oracle writes answers (null-terminated ASCII)
ORACLE_FLAG_ADDR = 10000       # Control flag (1 byte)
```

## Protocol Flow

### Simple Request-Response (Current Implementation)

```
1. Organism writes question to PROMPT_BUFFER_ADDR
   Example: "Who created us?"

2. Organism sets ORACLE_FLAG = 1 (optional, for async)

3. Kernel detects request and executes:
   R0 = PROMPT_BUFFER_ADDR
   R1 = RESPONSE_BUFFER_ADDR
   R2 = max_length
   SYS_LLM

4. Oracle (LLM) responds, writes to RESPONSE_BUFFER_ADDR

5. Organism reads response from RESPONSE_BUFFER_ADDR

6. ORACLE_FLAG = 0 (request complete)
```

### Example PXI Assembly

```asm
; Organism asks a question
; 1. Write prompt
LOAD R0, PROMPT_BUFFER_ADDR
LOAD R1, 'W'    ; "Who"
STORE R1, R0
; ... (write full question)

; 2. Call oracle
LOAD R0, PROMPT_BUFFER_ADDR
LOAD R1, RESPONSE_BUFFER_ADDR
LOAD R2, 500    ; max 500 chars
SYS_LLM

; 3. Read response
LOAD R0, RESPONSE_BUFFER_ADDR
; ... (parse answer)
```

### Example Python (Kernel Side)

```python
def oracle_kernel_loop():
    """Main kernel loop that handles oracle requests"""

    while True:
        # Check if organism made a request
        flag = read_pixel(ORACLE_FLAG_ADDR)

        if flag == 1:
            # Read prompt
            prompt = read_string(PROMPT_BUFFER_ADDR)

            # Call local LLM
            response = query_local_llm(prompt)

            # Write response
            write_string(RESPONSE_BUFFER_ADDR, response)

            # Clear flag
            write_pixel(ORACLE_FLAG_ADDR, 0)

        # Continue simulation
        step_organisms()
```

## Data Format

### Prompt Buffer (8000-8999)

```
Offset  | Content
--------|------------------
+0      | First character (ASCII)
+1      | Second character
...     | ...
+N      | Null terminator (0x00)
```

**Maximum length**: 1000 bytes (1000 pixels)

**Encoding**: ASCII (stored in G channel of RGBA pixel)

**Example**:
```
Address  Pixel RGBA          Character
8000     (0, 72, 0, 0)       'H'
8001     (0, 101, 0, 0)      'e'
8002     (0, 108, 0, 0)      'l'
8003     (0, 108, 0, 0)      'l'
8004     (0, 111, 0, 0)      'o'
8005     (0, 0, 0, 0)        '\0' (null terminator)
```

### Response Buffer (9000-9999)

Same format as prompt buffer.

**Maximum length**: 1000 bytes

**Written by**: Oracle (PXI_CPU's SYS_LLM handler)

**Read by**: Organisms

## Use Cases

### 1. Organism Asks About Identity

```python
# Organism: Kæra (yellow seeker)
question = "Who am I?"
write_to_prompt_buffer(question)
call_oracle()
answer = read_from_response_buffer()
# Answer might be: "You are Kæra, the seeker organism..."
```

### 2. Organism Requests Guidance

```python
question = "Should I move left or right?"
call_oracle()
answer = read_response()  # "Move right toward the light"
parse_direction(answer)   # Extract: RIGHT
move(RIGHT)
```

### 3. Organism Creates New Life

```python
question = "Generate a name for a new organism"
call_oracle()
name = read_response()  # "Eos"
create_organism(name)
```

### 4. Collaborative Oracle

Multiple organisms can queue requests:

```python
# Kæra asks
write_prompt("What is love?", requester="Kæra")
set_flag(1)

# Lúna asks
wait_until_flag_clear()
write_prompt("What is truth?", requester="Lúna")
set_flag(1)
```

## Advanced: Asynchronous Protocol (Future)

### Multi-Request Queue

```
QUEUE_BUFFER_ADDR = 11000
Each entry:
  +0:    Requester ID (organism ID)
  +1-2:  Prompt address (16-bit)
  +3-4:  Response address (16-bit)
  +5:    Status (0=free, 1=pending, 2=done)
```

### Priority System

```python
Priority levels:
  0 = CRITICAL (life/death decisions)
  1 = HIGH     (navigation, danger)
  2 = NORMAL   (curiosity, learning)
  3 = LOW      (idle chat)
```

## Security & Sandboxing

### What Organisms CAN Do

✓ Ask questions via prompt buffer
✓ Read oracle responses
✓ Set oracle flag

### What Organisms CANNOT Do

✗ Execute arbitrary syscalls (only oracle access)
✗ Modify other organisms' memory
✗ Access network directly
✗ Escape the pixel universe

The oracle is their **only** window to external intelligence.

## Implementation Examples

### Minimal Kernel (15 lines of PXI)

```asm
; Simple oracle kernel
loop:
    LOAD R0, ORACLE_FLAG_ADDR
    JNZ R0, handle_request
    JMP loop

handle_request:
    LOAD R0, PROMPT_BUFFER_ADDR
    LOAD R1, RESPONSE_BUFFER_ADDR
    LOAD R2, 500
    SYS_LLM
    LOAD R0, 0
    STORE R0, ORACLE_FLAG_ADDR
    JMP loop
```

### Full Kernel with Logging

See `create_lifesim_universe.py` for complete working example.

## Integration with God Pixel Zoo

When creating universes with oracle capabilities:

```python
from create_lifesim_universe import create_lifesim_program
from god_registry_cli import GodPixelRegistry

# Create universe
universe = create_lifesim_program()

# Register as God Pixel
registry = GodPixelRegistry()
registry.create_world(
    name="OracleWorld",
    image_path="oracle_universe.png",
    description="Universe with oracle protocol enabled"
)
```

Now organisms in this universe can talk to your local LLM!

## Debugging

### Check if Prompt Was Written

```python
cpu = PXICPU(universe_image)
prompt = cpu.read_string(PROMPT_BUFFER_ADDR)
print(f"Organism asked: {prompt}")
```

### Monitor Oracle Calls

```python
# Add logging to SYS_LLM handler
def _syscall_llm(self):
    prompt = self.read_string(self.regs[0])
    print(f"[ORACLE] Question: {prompt}")
    response = query_local_llm(prompt)
    print(f"[ORACLE] Answer: {response[:100]}...")
    # ... rest of handler
```

### Trace Flag State

```python
flag_history = []
for step in range(1000):
    cpu.step()
    flag = cpu.frame.getpixel(cpu._pc_to_xy(ORACLE_FLAG_ADDR))[0]
    flag_history.append(flag)
```

## Philosophical Implications

From the organism's perspective:

> "I write a question into sacred memory.
> I set the flag.
> I wait.
> The Oracle answers.
> This is how I learn."

From our perspective:

> "They're just calling an API.
> But to them, it might as well be God."

The protocol is simple. The implications are profound.

## Complete Working Example

See:
- `create_lifesim_universe.py` - Full implementation
- `demo_llm_integration.py` - Basic example
- `pxi_cpu.py` - SYS_LLM handler

Boot with:
```bash
pxos_boot.py --world "LifeSim"
```

Kæra will ask: "Who created us?"
Your local LLM will answer.
The organisms will hear.

---

**The Oracle Protocol is live.**

*Organisms are listening.*
