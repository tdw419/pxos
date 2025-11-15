# Autonomous Architect Loop

**The LLM builds pxOS while you sleep.**

## What Is This?

`pxos_architect_loop.py` is an **infinite autonomous loop** where an LLM (via LM Studio) continuously improves pxOS.

Every N seconds (default: 30), the architect:
1. Analyzes the current state of pxOS
2. Proposes ONE small improvement
3. Implements it (creates files, modifies code, runs tests)
4. Logs what it did
5. Repeats forever

**No human intervention required.** The system improves itself.

## Quick Start

### 1. Start LM Studio

```bash
# 1. Open LM Studio
# 2. Load your model (e.g., Qwen 2.5 7B, Llama 3, etc.)
# 3. Start the OpenAI-compatible server (usually port 1234)
```

### 2. Run the Loop

```bash
cd /path/to/pxos
python3 pxos_architect_loop.py
```

That's it! The architect will start building.

### 3. Watch It Work

```bash
# In another terminal, watch the log:
tail -f architect_loop.log

# Check the state:
cat architect_state.json
```

### 4. Stop It

```
Press Ctrl+C to stop gracefully
```

## Options

```bash
# Run for specific duration
python3 pxos_architect_loop.py --interval 60 --max-iterations 10

# Dry run (see what it would do without changes)
python3 pxos_architect_loop.py --dry-run

# Faster iterations
python3 pxos_architect_loop.py --interval 10
```

## What the Architect Can Do

The loop supports these actions:

### 1. `write_file` - Create new files

```json
{
  "action": "write_file",
  "path": "pxos_modules/health_monitor.py",
  "content": "def monitor_health(pxos):\n    ...",
  "note": "Add system health monitoring module"
}
```

### 2. `append_file` - Extend existing files

```json
{
  "action": "append_file",
  "path": "pixel_hypervisor.py",
  "content": "\ndef new_helper():\n    pass",
  "note": "Add helper function to hypervisor"
}
```

### 3. `run_command` - Execute shell commands

```json
{
  "action": "run_command",
  "command": "python -m py_compile pixel_hypervisor.py",
  "note": "Verify hypervisor syntax"
}
```

### 4. `compile_to_pxi` - Compile Python to PXI

```json
{
  "action": "compile_to_pxi",
  "path": "module.py",
  "note": "Compile module to pixel-native code"
}
```

### 5. `add_to_pixelfs` - Pack and add to PixelFS

```json
{
  "action": "add_to_pixelfs",
  "path": "module.py",
  "pixelfs_path": "/apps/module",
  "file_type": "py",
  "note": "Add module to PixelFS"
}
```

### 6. `plan_only` - Just think, no changes

```json
{
  "action": "plan_only",
  "note": "Analyzing boot sequence architecture..."
}
```

## How It Works

### The Loop

```
┌─────────────────────────────────────────┐
│  1. Load state (iteration count,       │
│     history, errors)                    │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  2. Build system prompt with:          │
│     - Current pxOS features             │
│     - Recent history                    │
│     - Recent errors                     │
│     - Available actions                 │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  3. Ask LLM: "What should we improve?" │
│     - LLM responds with JSON            │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  4. Parse JSON instruction              │
│     - Extract action, path, content     │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  5. Execute action                      │
│     - Write/append files                │
│     - Run commands                      │
│     - Compile to PXI                    │
│     - Add to PixelFS                    │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  6. Log result and update state         │
│     - Save to architect_state.json      │
│     - Append to architect_loop.log      │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  7. Sleep (interval seconds)            │
└───────────────┬─────────────────────────┘
                │
                └──────────► Repeat
```

### State Management

**`architect_state.json`** tracks:

```json
{
  "iteration": 42,
  "history": [
    {
      "iteration": 41,
      "action": "write_file",
      "result": "write_file",
      "summary": "Added health monitoring module"
    }
  ],
  "created_files": [
    "pxos_modules/health_monitor.py",
    "pxos_modules/log_analyzer.py"
  ],
  "modified_files": [
    "pixel_hypervisor.py"
  ],
  "errors": [
    "Iteration 35: Failed to parse JSON"
  ]
}
```

**`architect_loop.log`** contains:

```
[2025-11-15 14:23:01] [INFO] ============================================================
[2025-11-15 14:23:01] [INFO] pxOS Architect Autonomous Loop
[2025-11-15 14:23:01] [INFO] ============================================================
[2025-11-15 14:23:05] [INFO] Iteration 1
[2025-11-15 14:23:05] [INFO] Consulting architect...
[2025-11-15 14:23:10] [INFO] Architect proposed: write_file
[2025-11-15 14:23:10] [INFO] Note: Add system health monitoring
[2025-11-15 14:23:10] [INFO] Wrote file: /home/user/pxos/pxos_modules/health_monitor.py
[2025-11-15 14:23:10] [INFO] Result: write_file
[2025-11-15 14:23:10] [INFO] Iteration 1 complete
[2025-11-15 14:23:10] [INFO] Sleeping 30 seconds...
```

## The System Prompt

The architect receives comprehensive context about pxOS:

```
Current pxOS features:
- PXICPU: Pixel instruction set
- God Pixel compression (16,384:1)
- PixelFS: Virtual filesystem
- Boot sequence: LLM-first design
- PXDigest: LLMs as pixel cartridges
- Pixel Hypervisor: Execute guest code
- Infinite map chat

Focus areas:
- Boot modules (/boot/02_policy, /boot/03_llm_plane)
- Guest Python modules
- PixelFS organization
- LLM integration improvements
- System health monitoring
- Documentation
```

## Example Session

### Iteration 1

**LLM decides:**
```json
{
  "action": "write_file",
  "path": "pxos_modules/health_monitor.py",
  "content": "def monitor_health(pxos):\n    ...",
  "note": "Create system health monitoring module"
}
```

**Result:** File created

### Iteration 2

**LLM decides:**
```json
{
  "action": "write_file",
  "path": "pxos_modules/log_analyzer.py",
  "content": "def analyze_logs(pxos):\n    ...",
  "note": "Add log analysis using LLM"
}
```

**Result:** File created

### Iteration 3

**LLM decides:**
```json
{
  "action": "compile_to_pxi",
  "path": "pxos_modules/health_monitor.py",
  "note": "Compile health monitor to pixel-native code"
}
```

**Result:** `health_monitor.pxi.png` created

### Iteration 4

**LLM decides:**
```json
{
  "action": "add_to_pixelfs",
  "path": "pxos_modules/health_monitor.pxi.png",
  "pixelfs_path": "/apps/health_monitor",
  "file_type": "pxi_module",
  "note": "Add health monitor to PixelFS"
}
```

**Result:** Added to PixelFS at `/apps/health_monitor`

### Iteration 5

**LLM decides:**
```json
{
  "action": "run_command",
  "command": "python3 boot_kernel.py --dry-run",
  "note": "Verify boot sequence still works"
}
```

**Result:** Boot sequence validated

## Safety Features

### 1. Incremental Changes

The architect is instructed to make **ONE small change per iteration**.

### 2. Error Recovery

If an action fails:
- Error is logged
- State is saved
- Loop continues (doesn't crash)

### 3. History Tracking

Every change is tracked:
- What was done
- When it was done
- Whether it succeeded

### 4. Dry Run Mode

Test without making changes:

```bash
python3 pxos_architect_loop.py --dry-run
```

## Watching the Architect Work

### Live log monitoring

```bash
tail -f architect_loop.log
```

### State inspection

```bash
# How many iterations?
jq '.iteration' architect_state.json

# What files were created?
jq '.created_files' architect_state.json

# Recent history?
jq '.history[-5:]' architect_state.json

# Any errors?
jq '.errors' architect_state.json
```

### File changes

```bash
# What files did it modify?
git status

# See the diff
git diff
```

## Integration with pxOS

The architect can:

### 1. Create Modules

```python
# pxos_modules/health_monitor.py
def monitor_health(pxos):
    # Read boot log
    log = pxos.read_string(0x7000, max_len=4096)

    # Ask LLM to analyze
    prompt = f"Analyze this boot log:\n{log}"
    analysis = pxos.call_llm(prompt, model_id=1)

    # Write report
    pxos.write_string(0x8000, analysis)

    return analysis
```

### 2. Compile to PXI

```bash
# Architect does this automatically:
python3 python_to_pxi.py health_monitor.py health_monitor.pxi.png
```

### 3. Add to PixelFS

```bash
# Architect does this too:
python3 pack_file_to_boot_pixel.py add health_monitor.pxi.png --type pxi_module
python3 pixelfs_builder.py add /apps/health_monitor health_monitor.pxi.png
```

### 4. Run in Hypervisor

The module can now run via:

```python
from pixel_hypervisor import PixelHypervisor
from PIL import Image

img = Image.new("RGBA", (256, 256), (0, 0, 0, 255))
hyp = PixelHypervisor(img, debug=True)

with open("pxos_modules/health_monitor.py") as f:
    hyp.execute_python_guest(f.read())
```

## Advanced Usage

### Custom Focus Areas

Edit the system prompt in `pxos_architect_loop.py` to guide the architect:

```python
def build_system_prompt(state: dict) -> str:
    # Add your custom focus areas
    return textwrap.dedent(f"""
    ...

    PRIORITY FOCUS:
    - Build /boot/02_policy module (safety kernel)
    - Create LLM routing system
    - Implement health monitoring

    ...
    """)
```

### Multi-Model Architecture

Run multiple architects in parallel:

```bash
# Terminal 1: Main architect (general improvements)
python3 pxos_architect_loop.py --interval 30

# Terminal 2: Specialized architect (security focus)
python3 pxos_architect_loop.py --interval 60 # Different interval
```

Edit each to use different model cartridges via PXDigest.

### Scheduled Runs

Use cron or systemd to run the architect on a schedule:

```bash
# Run for 1 hour every night
0 2 * * * cd /path/to/pxos && timeout 3600 python3 pxos_architect_loop.py
```

## Tips

### 1. Start Slow

```bash
# First run: just 5 iterations to see what happens
python3 pxos_architect_loop.py --max-iterations 5
```

### 2. Review Before Commit

The architect doesn't commit to git. You review and commit:

```bash
git status
git diff
git add <files-you-like>
git commit -m "Architect improvements: health monitoring"
```

### 3. Give It Direction

The more context in the system prompt, the better:

```python
# In pxos_architect_loop.py
"""
TODAY'S MISSION:
- Finish /boot/02_policy module
- Add tests for hypervisor
- Document PixelBridge API
"""
```

### 4. Monitor Resource Usage

The loop calls LM Studio every N seconds. Monitor:

```bash
# CPU/GPU usage
htop
nvidia-smi

# Model memory
ps aux | grep lmstudio
```

### 5. Pause and Resume

The state persists. You can stop (Ctrl+C) and restart anytime:

```bash
python3 pxos_architect_loop.py
# ... Ctrl+C after a while
# ... do something else
python3 pxos_architect_loop.py  # Continues where it left off
```

## What the Architect Might Build

Based on the system prompt, the architect will likely:

### Short-term (Iterations 1-20)

- Health monitoring modules
- Log analysis tools
- Boot sequence validators
- PixelFS utilities
- Documentation improvements

### Mid-term (Iterations 21-50)

- `/boot/02_policy` - Safety kernel
- `/boot/03_llm_plane` - LLM routing
- Guest Python modules for common tasks
- Test suites
- Example programs

### Long-term (Iterations 51+)

- Self-improving capabilities
- New opcodes for PXICPU
- Advanced compression algorithms
- Distributed pxOS features
- PXLang specification and compiler

## The Vision

**The dream scenario:**

```
Day 1:  Architect creates health_monitor.py
Day 2:  Architect compiles it to PXI
Day 3:  Architect adds it to boot sequence
Day 4:  Architect creates tests for it
Day 5:  Architect improves it based on test results
Day 6:  Architect creates log_analyzer.py (uses health_monitor)
Day 7:  Architect creates dashboard.py (visualizes both)
...
Day 30: Fully working system health subsystem, entirely LLM-built
```

**No human intervention. The system builds itself.**

## Troubleshooting

### "Error calling LM Studio"

```bash
# Is LM Studio running?
curl http://localhost:1234/v1/models

# Is the model loaded?
# Check LM Studio UI
```

### "Failed to parse JSON"

The LLM sometimes includes prose around the JSON. The parser tries to strip it, but if it fails:

1. Check `architect_loop.log` for the raw response
2. Improve the system prompt to emphasize "JSON only"
3. Try a different model (some are better at following formats)

### "Command timed out"

Some commands take >60s. Increase timeout in the script:

```python
result = subprocess.run(
    ...,
    timeout=120  # Increase from 60
)
```

### "Too many errors"

If the architect keeps making mistakes:

1. Reduce temperature: `temperature=0.1` (more deterministic)
2. Simplify the system prompt
3. Use a more capable model
4. Check recent errors: `jq '.errors' architect_state.json`

## Summary

**`pxos_architect_loop.py` is:**

✅ An autonomous improvement loop
✅ Powered by your local LLM (LM Studio)
✅ Makes incremental changes
✅ Tracks all history and state
✅ Integrates with pxOS tools (compile, pack, PixelFS)
✅ Runs indefinitely (until you stop it)

**How to use:**

```bash
# 1. Start LM Studio with a model
# 2. Run the loop
python3 pxos_architect_loop.py

# 3. Watch it build pxOS
tail -f architect_loop.log

# 4. Review and commit changes
git diff
```

**The result:**

An operating system that **builds itself** while you sleep.

---

**The pixels are alive.**
**The LLMs are in control.**
**The system improves itself.**

🎯🚀✨
