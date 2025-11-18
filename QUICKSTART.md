# pxOS AI Build System - Quick Start Guide

**Build an operating system with AI in 5 minutes!**

---

## Step 1: Install Dependencies

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Install QEMU (optional, for testing)
sudo apt install qemu-system-x86  # Ubuntu/Debian
# or
brew install qemu  # macOS
```

---

## Step 2: Start LM Studio

1. **Download LM Studio**: https://lmstudio.ai/
2. **Load a model**: CodeLlama, DeepSeek Coder, or any code-capable model
3. **Start server**: Click "Start Server" - should run on `localhost:1234`

**Verify it's running:**
```bash
curl http://localhost:1234/v1/models
```

---

## Step 3: Run the AI Builder

```bash
# Make script executable
chmod +x run_ai_build.sh

# Run with menu
./run_ai_build.sh
```

**Or choose directly:**

```bash
# Demo the learning system
./run_ai_build.sh --demo

# Full automation
./run_ai_build.sh --auto
```

---

## What Happens Next?

The AI will:
1. 🤖 **Analyze** current pxOS state
2. 📋 **Plan** implementation steps
3. ⚙️ **Generate** primitive commands (x86 assembly as WRITE/DEFINE)
4. 🔨 **Build** the bootable binary
5. 🧪 **Test** in QEMU (if available)
6. 💾 **Learn** from results

Each build makes the AI smarter!

---

## Example Output

```
🚀 STARTING AUTOMATED BUILD CYCLE
═══════════════════════════════════════════════════════════

GOALS:
  1. Add backspace support to shell
  2. Implement command parser for basic commands
  3. Add 'help' command
  4. Add 'clear' command

🔍 Analyzing current pxOS state...
   Commands: 89 lines
   Binary: 512 bytes

🎯 Generating build plan...
   Generated 4 build steps

🔧 EXECUTING BUILD PLAN
═══════════════════════════════════════════════════════════

=== Iteration 1/10 ===

⚙️  Implementing: Add backspace support to shell
🤖 Generating primitives...
✅ Generated 15 primitive commands
💾 Appending to pixel network...
   ✅ Network expanded: 250 → 268 rows (+18)

🔨 Building binary...
   ✅ Build successful

🧪 Testing in QEMU...
   ✅ QEMU test passed

✅ Step completed: Add backspace support to shell

...

📊 BUILD CYCLE COMPLETE
═══════════════════════════════════════════════════════════

Initial commands: 89
Final commands: 143
Final binary size: 612 bytes

Completed 4/4 steps

🎉 Automated build cycle completed successfully!
```

---

## Test Your pxOS

```bash
# Boot in QEMU
cd pxos-v1.0
./tests/boot_qemu.sh

# Or manually
qemu-system-i386 -fda pxos.bin
```

---

## What Just Happened?

You just:
- ✅ Used **LM Studio** (local AI) to generate OS code
- ✅ Created a **pixel network** that learns from builds
- ✅ Built a **bootable x86 binary** from AI-generated primitives
- ✅ Tested on **real hardware** (via QEMU emulation)

**The AI learned from this build and will do better next time!**

---

## Next Steps

### Customize Goals

Edit the goals in `tools/auto_build_pxos.py` or run:

```bash
python3 tools/auto_build_pxos.py --goals "FAT12 filesystem" "multi-tasking"
```

### Interactive Code Generation

```bash
./run_ai_build.sh --interactive

💡 Feature to implement: Add reboot command
🤖 Generating primitives...
```

### View the Learning Network

```bash
# The AI's knowledge is stored as a PNG!
eog pxvm/networks/pxos_autobuild.png  # Linux
open pxvm/networks/pxos_autobuild.png # macOS
```

Each Q&A is rendered as pixels and appended. The network grows with experience!

---

## Troubleshooting

### "Cannot connect to LM Studio"

- Ensure LM Studio is running
- Check it's on port 1234: http://localhost:1234
- Verify a model is loaded

### "Build failed"

- Check `build_report.json` for details
- Review `pxos-v1.0/pxos_commands.txt` for syntax errors
- The AI learns from failures - try running again

### "ModuleNotFoundError"

```bash
pip3 install -r requirements.txt
```

---

## Architecture Overview

```
┌──────────────┐
│  LM Studio   │  ← Local AI (your computer)
│  (localhost) │
└──────┬───────┘
       │
       │ HTTP API
       │
┌──────▼────────────────────────────────────────┐
│  AI Build System                              │
│                                               │
│  ┌─────────────────┐    ┌─────────────────┐  │
│  │ LM Studio Bridge│◄──►│ Pixel Network   │  │
│  │  (queries AI)   │    │  (learns)       │  │
│  └────────┬────────┘    └─────────────────┘  │
│           │                                   │
│  ┌────────▼────────┐    ┌─────────────────┐  │
│  │Primitive Gen    │───►│ pxos_commands   │  │
│  │(WRITE/DEFINE)   │    │     .txt        │  │
│  └─────────────────┘    └────────┬────────┘  │
│                                  │           │
│                         ┌────────▼────────┐  │
│                         │  build_pxos.py  │  │
│                         │  (assembler)    │  │
│                         └────────┬────────┘  │
│                                  │           │
│                         ┌────────▼────────┐  │
│                         │   pxos.bin      │  │
│                         │  (bootable!)    │  │
│                         └─────────────────┘  │
└───────────────────────────────────────────────┘
```

---

## Files Generated

- `pxos-v1.0/pxos.bin` - Bootable OS binary
- `pxvm/networks/pxos_autobuild.png` - AI knowledge base
- `build_report.json` - Detailed build log
- `pxos-v1.0/pxos_commands.txt` - Updated primitives

---

## More Information

- **Full Documentation**: `AI_BUILD_SYSTEM.md`
- **pxOS README**: `README.md`
- **Primitive Reference**: `pxos-v1.0/docs/primitives.md`

---

**Welcome to the future of OS development!** 🚀

The system that builds itself, using AI that learns from each build.
