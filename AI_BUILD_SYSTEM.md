# pxOS AI Build System

**Automated OS Development with LM Studio + Self-Expanding Pixel Networks**

This system uses AI to automatically generate, build, and improve pxOS by combining:
- **LM Studio**: Local LLM for code generation
- **Pixel Networks**: Self-expanding knowledge base that learns from each build
- **Primitive Generator**: AI that writes x86 assembly as WRITE/DEFINE commands
- **Automated Build Loop**: Iterative improvement system

---

## Quick Start

### Prerequisites

1. **LM Studio** running on `localhost:1234` with a model loaded
   - Download: https://lmstudio.ai/
   - Load any code-capable model (e.g., CodeLlama, DeepSeek Coder)

2. **Python 3.6+** with dependencies:
   ```bash
   pip3 install requests numpy pillow
   ```

3. **QEMU** (optional, for testing):
   ```bash
   sudo apt install qemu-system-x86
   ```

### Run Automated Build

```bash
# Simple menu interface
./run_ai_build.sh

# Or run directly:
./run_ai_build.sh --auto
```

---

## System Components

### 1. LM Studio Bridge (`pxvm/integration/lm_studio_bridge.py`)

Self-expanding AI learning system that:
- Queries LM Studio with accumulated context
- Stores Q&A interactions as pixels in PNG files
- Gets smarter with each interaction

**Usage:**
```bash
# Demo the learning system
python3 pxvm/integration/lm_studio_bridge.py --demo

# Interactive mode
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

### 2. Primitive Generator (`tools/ai_primitive_generator.py`)

AI-powered code generator that converts feature descriptions to pxOS primitives.

**Usage:**
```bash
# Generate primitives for a feature
python3 tools/ai_primitive_generator.py --feature "Add backspace support"

# Interactive generation
python3 tools/ai_primitive_generator.py --interactive
```

**Example:**
```
Input: "Add backspace support"

Output:
COMMENT Add backspace support
DEFINE backspace_handler 0x7E00
WRITE 0x7E00 0xB4    COMMENT mov ah, 0x0E
WRITE 0x7E01 0x0E
WRITE 0x7E02 0xB0    COMMENT mov al, 0x08 (backspace)
WRITE 0x7E03 0x08
...
```

### 3. Automated Builder (`tools/auto_build_pxos.py`)

Full automation system that:
1. Analyzes current pxOS state
2. Generates build plan with AI
3. Implements features step-by-step
4. Builds and tests each iteration
5. Learns from successes and failures

**Usage:**
```bash
# Automated build with default goals
python3 tools/auto_build_pxos.py --auto

# Custom goals
python3 tools/auto_build_pxos.py --goals "backspace" "help command" "clear screen"

# With QEMU testing
python3 tools/auto_build_pxos.py --auto --test
```

### 4. Learning Network (`pxvm/networks/`)

Pixel-based knowledge accumulation:
- Each build session is stored as pixels
- Network grows with experience
- Future builds benefit from past knowledge

---

## How It Works

### The Self-Improving Loop

```
┌─────────────────────────────────────────────────────────┐
│  1. ANALYZE                                             │
│     • Read current pxOS state                           │
│     • Count primitives, measure binary size             │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  2. PLAN (with AI)                                      │
│     • Query LM Studio: "How to implement X?"            │
│     • Generate step-by-step build plan                  │
│     • Use accumulated pixel network knowledge           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  3. GENERATE (with AI)                                  │
│     • For each step, generate primitive commands        │
│     • LM Studio produces WRITE/DEFINE/COMMENT lines     │
│     • Validate x86 opcodes and addresses                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  4. BUILD                                               │
│     • Append primitives to pxos_commands.txt            │
│     • Run build_pxos.py to generate binary              │
│     • Check for errors                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  5. TEST                                                │
│     • Boot in QEMU                                      │
│     • Verify no crashes                                 │
│     • Collect results                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  6. LEARN                                               │
│     • Render results as pixels                          │
│     • Append to knowledge network                       │
│     • Network grows, next iteration is smarter!         │
└─────────────────┴───────────────────────────────────────┘
                  │
                  └──► Repeat until goals achieved
```

---

## Examples

### Example 1: Demo the Learning System

```bash
./run_ai_build.sh --demo
```

This demonstrates how the pixel network learns:
1. Asks LM Studio a question WITHOUT context
2. Teaches the network about pxOS
3. Asks the same question WITH context
4. Shows improved answer!

### Example 2: Interactive Code Generation

```bash
./run_ai_build.sh --interactive
```

Then describe features:
```
💡 Feature to implement: Add a help command that prints available commands

🤖 Generating primitives...
✅ Generated 25 primitive commands

💾 Append to commands file? (y/n): y
```

### Example 3: Full Automation

```bash
./run_ai_build.sh --auto
```

The system will:
- Analyze current pxOS
- Generate build plan for default features
- Implement each feature with AI
- Build and test iteratively
- Save detailed report

**Default goals:**
1. Add backspace support to shell
2. Implement command parser for basic commands
3. Add 'help' command
4. Add 'clear' command

---

## Configuration

### LM Studio Settings

For best results:
- **Model**: CodeLlama 13B, DeepSeek Coder, or similar
- **Temperature**: 0.7
- **Max tokens**: 500-1000
- **Context length**: 4096+

### Network Paths

Knowledge networks are stored in `pxvm/networks/`:
- `pxos_autobuild.png` - Main automation network
- `pxos_dev.png` - Development/testing network
- `learning_network.png` - General learning network

### Build Configuration

Edit `tools/auto_build_pxos.py` to customize:
- Max iterations: `--max-iterations 20`
- Goals: `--goals "feature1" "feature2"`
- pxOS directory: `--pxos-dir pxos-v1.0`

---

## File Structure

```
pxos/
├── run_ai_build.sh              # Main automation script
├── AI_BUILD_SYSTEM.md           # This file
│
├── pxvm/                        # AI/ML components
│   ├── integration/
│   │   └── lm_studio_bridge.py  # LM Studio interface
│   ├── learning/
│   │   └── append.py            # Pixel rendering
│   └── networks/                # Knowledge storage (PNG files)
│       └── pxos_autobuild.png
│
├── tools/                       # Build automation
│   ├── ai_primitive_generator.py   # AI code generator
│   └── auto_build_pxos.py          # Full automation
│
└── pxos-v1.0/                   # pxOS source
    ├── build_pxos.py            # Primitive → binary compiler
    ├── pxos_commands.txt        # Primitive source code
    └── pxos.bin                 # Bootable binary
```

---

## Advanced Usage

### Custom Feature Generation

```bash
python3 tools/ai_primitive_generator.py \
    --feature "Implement FAT12 filesystem driver" \
    --addr 0x8000 \
    --output custom_commands.txt
```

### Batch Processing

```bash
# Generate multiple features
for feature in "backspace" "help" "clear" "reboot"; do
    python3 tools/ai_primitive_generator.py \
        --feature "$feature" \
        --output "features_$feature.txt"
done
```

### Integration Testing

```bash
# Build + test in QEMU with custom timeout
python3 tools/auto_build_pxos.py \
    --auto \
    --test \
    --max-iterations 20 \
    --report detailed_report.json
```

---

## Troubleshooting

### LM Studio Connection Errors

**Problem**: `Cannot connect to LM Studio`

**Solution**:
1. Ensure LM Studio is running
2. Check a model is loaded
3. Verify server is on port 1234
4. Test with: `curl http://localhost:1234/v1/models`

### Primitive Validation Errors

**Problem**: Generated primitives fail validation

**Solution**:
- The AI sometimes generates invalid opcodes
- Review `build_report.json` for specifics
- The system learns from errors - try re-running
- Manually edit `pxos_commands.txt` if needed

### Build Failures

**Problem**: `build_pxos.py` fails

**Solution**:
- Check syntax in `pxos_commands.txt`
- Look for duplicate DEFINE labels
- Verify addresses don't overlap
- Check boot signature at 0x1FE-0x1FF

### QEMU Crashes

**Problem**: Binary crashes in QEMU

**Solution**:
- Verify stack setup (CLI/STI, SS:SP)
- Check for invalid memory access
- Use QEMU debug: `qemu-system-i386 -fda pxos.bin -d int,cpu_reset`

---

## Knowledge Base Seeding

The system comes pre-seeded with:
- x86 opcode reference
- BIOS interrupt specifications
- Boot sector requirements
- Memory map conventions
- Common assembly patterns

This knowledge improves with every build!

---

## Contributing

The AI system learns from:
- ✅ Successful builds
- ❌ Failed builds (learns what NOT to do)
- 💡 User corrections
- 🔧 Manual edits to generated code

Each interaction makes the network smarter!

---

## Architecture Philosophy

**Why Pixel Networks?**

Traditional build systems use text logs. We use **pixels** because:
1. **Append-only**: Perfect for immutable learning
2. **Visual**: Can render diagrams, tables, assembly listings
3. **Compact**: PNG compression is efficient
4. **Semantic**: Can use computer vision for pattern recognition
5. **Novel**: Demonstrates pxOS's pixel-first philosophy

**Why LM Studio?**

- **Local**: No cloud dependencies
- **Private**: Your code never leaves your machine
- **Fast**: Low latency for iterative development
- **Flexible**: Swap models easily

---

## Future Enhancements

Planned features:
- [ ] Multi-model ensembling (query multiple LLMs, pick best)
- [ ] Semantic search in pixel networks (find similar past solutions)
- [ ] Visual diff rendering (show changes as images)
- [ ] Auto-optimization (AI suggests performance improvements)
- [ ] Test case generation (AI writes test scenarios)
- [ ] Documentation generation (auto-generate docs from code)

---

## License

Same as pxOS: MIT License

---

## Credits

Built on:
- **pxOS**: Primitive-based bootloader system
- **LM Studio**: Local LLM runtime
- **pxVM**: Pixel network architecture

**Made with AI assistance, for AI-assisted development!**

---

*"The operating system that builds itself..."*
