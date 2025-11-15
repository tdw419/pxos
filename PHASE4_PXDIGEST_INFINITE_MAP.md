# Phase 4: PXDigest + Infinite Map Chat

**LLMs become pixel cartridges. Chat on an infinite plane.**

## Vision

```
Phase 1: One pixel = one universe
Phase 2: Multiple universes + platform
Phase 3: Self-hosting toolchain
Phase 4: LLMs as pixels + infinite conversational map
```

**Goal**: Turn local LLMs into pixel cartridges that can be selected, swapped, and used across an infinite 2D conversational map where each tile maintains its own chat context.

## What We Built

### 1. PXDigest - LLM Pixel Cartridge System

**Turn any local LLM into a 1×1 pixel cartridge.**

```bash
# Create an LLM pixel
python3 px_digest_model.py create "LocalLlama" \
    --backend lmstudio \
    --endpoint "http://localhost:1234/v1/chat/completions" \
    --model "local-model"

# → Creates llm_localllama.pxdigest.png (1×1)
# → RGBA color encodes the model configuration
```

**Registry Structure**:
```json
{
  "1106371994": {
    "name": "LocalLlama",
    "backend": "lmstudio",
    "endpoint": "http://localhost:1234/v1/chat/completions",
    "model_name": "local-model",
    "max_tokens": 512,
    "temperature": 0.7,
    "pixel": [65, 226, 147, 154],
    "id": "0x41E2939A"
  }
}
```

**Commands**:
```bash
# List all LLM pixels
python3 px_digest_model.py list

# Show details
python3 px_digest_model.py show "LocalLlama"

# Delete
python3 px_digest_model.py delete "LocalLlama"
```

### 2. Extended SYS_LLM with Model Selection

**PXI_CPU now supports multiple models via R3 register.**

```assembly
; Old way (default model)
LOAD R0, prompt_addr
LOAD R1, output_addr
LOAD R2, max_len
SYS_LLM

; New way (select model)
LOAD R0, prompt_addr
LOAD R1, output_addr
LOAD R2, max_len
LOAD R3, 0x41E2939A    ; Model ID from PXDigest
SYS_LLM
```

**Python side**:
```python
# Automatic model selection
if model_id:
    response = query_local_llm_via_digest(prompt, model_id, max_len)
else:
    response = query_local_llm(prompt)  # Default
```

### 3. Infinite Map Chat UI

**Navigate an infinite 2D plane where each tile has its own conversation.**

```bash
python3 infinite_map_chat.py
```

**Features**:
- **Infinite grid**: 32-bit signed coordinates (±2 billion tiles)
- **Per-tile conversations**: Each (x,y) maintains separate chat history
- **Model selection**: Press 1-9 to select LLM for current tile
- **Persistent state**: Conversations saved to `infinite_map_state.json`
- **Visual indicators**: Tiles with conversations shown in blue
- **Real-time chat**: Type and get responses from local LLM

**Controls**:
```
Arrow Keys / WASD - Navigate map
Enter             - Start chatting on current tile
1-9               - Select LLM model (from registry)
Tab               - Show model info
Esc               - Quit
```

**Map View**:
```
┌─────┬─────┬─────┬─────┬─────┐
│     │     │     │     │     │
├─────┼─────┼─────┼─────┼─────┤
│     │ [5] │     │     │     │  [5] = 5 messages
├─────┼─────┼─────┼─────┼─────┤
│     │     │ █ │ │     │     │  █ = cursor
├─────┼─────┼─────┼─────┼─────┤
│     │     │     │ [3] │     │
└─────┴─────┴─────┴─────┴─────┘

Coordinates: (1024, -512)
Model: LocalLlama
> Hello, what is this place?_
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              INFINITE MAP (±2B × ±2B tiles)              │
│  Each tile = separate conversation context               │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│           PXDigest LLM Cartridge Registry                │
│  llm_pixel_registry.json                                 │
│  ├── LocalLlama    → RGBA(65, 226, 147, 154)             │
│  ├── TinyLlama     → RGBA(...)                           │
│  └── CustomModel   → RGBA(...)                           │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│         SYS_LLM (Extended with R3 = model_id)            │
│  PXI_CPU can now select which LLM to query               │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│              LOCAL LLM BACKEND                           │
│  LM Studio (port 1234)                                   │
│  Ollama (port 11434)                                     │
│  Custom endpoint                                         │
└──────────────────────────────────────────────────────────┘
```

## Use Cases

### 1. Different LLMs for Different Contexts

```
Tile (0, 0):     Creative writing → Use CreativeModel
Tile (100, 50):  Code debugging   → Use CodeModel
Tile (-50, -20): Math tutoring    → Use MathModel
```

Each conversation maintains its own context with the appropriate specialized model.

### 2. Comparing LLM Outputs

Ask the same question on adjacent tiles with different models:

```
Tile (5, 5):  LocalLlama response
Tile (6, 5):  TinyLlama response
Tile (7, 5):  Ollama Llama3 response
```

Visually compare responses across space.

### 3. Persistent Conversation Spaces

```
Tile (0, 0):     Personal assistant (always at origin)
Tile (1000, 0):  Project planning space
Tile (0, 1000):  Learning journal
```

Different areas of the map for different purposes.

### 4. Organism-LLM Integration

Future: Combine with LifeSim - organisms can "pray" at different map locations to different LLMs:

```
Tile (50, 50):   Oracle of Wisdom (philosophical model)
Tile (100, 100): Oracle of Code (coding model)
Tile (150, 150): Oracle of Stories (creative model)
```

Kæra travels to different tiles to ask different questions.

## Implementation Details

### PXDigest ID Generation

```python
# Random 32-bit ID
pid = random.getrandbits(32)

# Encode as RGBA
R = (pid >> 24) & 0xFF
G = (pid >> 16) & 0xFF
B = (pid >> 8) & 0xFF
A = pid & 0xFF

# Create 1×1 pixel
img = Image.new("RGBA", (1, 1), (R, G, B, A))
```

### Tile Key Format

```python
# World coordinates to key
key = f"{world_x},{world_y}"

# Example: tile at (1024, -512)
key = "1024,-512"
```

### State Persistence

```json
{
  "tiles": {
    "0,0": {
      "model_id": 1106371994,
      "history": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
      ]
    },
    "100,50": {
      "model_id": 1234567890,
      "history": [...]
    }
  }
}
```

### Query Flow

```
User types message at tile (x,y)
  ↓
Get or create TileConversation(x, y)
  ↓
Append user message to history
  ↓
Build prompt from last N messages
  ↓
Query LLM via PXDigest:
  - Look up model_id in registry
  - Get endpoint + config
  - POST to endpoint
  ↓
Receive response
  ↓
Append to history
  ↓
Display in chat UI
  ↓
Save state to disk
```

## Files Created (Phase 4)

```
⭐ px_digest_model.py           - PXDigest LLM cartridge system (400+ lines)
⭐ infinite_map_chat.py          - Infinite map UI (350+ lines)
⭐ pxi_cpu.py (updated)          - Extended SYS_LLM with R3 model selection
⭐ llm_pixel_registry.json       - LLM cartridge registry (generated)
⭐ infinite_map_state.json       - Map conversation state (generated)
⭐ llm_*.pxdigest.png            - LLM pixel cartridges (1×1 each)
⭐ PHASE4_PXDIGEST_INFINITE_MAP.md - This document
```

## Integration with Previous Phases

### Phase 1 (God Pixel)
- LLM pixels use same registry pattern as God Pixels
- RGBA encoding, hash-based lookup

### Phase 2 (God Pixel Zoo)
- PXDigest registry = LLM Zoo
- Same CLI pattern (create, list, show, delete)

### Phase 3 (Self-Hosting)
- PXDigest files included in Project Boot Pixel
- LLM cartridges boot with the system

### Phase 4 (This)
- LLMs become swappable cartridges
- Infinite map as conversational interface
- Per-tile context management

## Testing

### 1. Create LLM Pixel

```bash
python3 px_digest_model.py create "TestLLM" \
    --backend lmstudio \
    --endpoint "http://localhost:1234/v1/chat/completions"
```

### 2. List Models

```bash
python3 px_digest_model.py list
```

Expected output:
```
╔═══════════════════════════════════════════════════════════╗
║              LLM PIXEL CARTRIDGE REGISTRY                 ║
╚═══════════════════════════════════════════════════════════╝

1. TestLLM
   Color: RGBA(65, 226, 147, 154) → #41E2939A
   Backend: lmstudio
   Endpoint: http://localhost:1234/v1/chat/completions
   ID: 0x41E2939A
```

### 3. Run Infinite Map

```bash
python3 infinite_map_chat.py
```

- Navigate with arrows
- Press Enter on a tile
- Type a message
- Get LLM response
- Responses saved per-tile

### 4. Verify State

```bash
cat infinite_map_state.json
```

Should show saved conversations.

## Roadmap

### Current (Phase 4a): ✅ Done
- ✅ PXDigest LLM cartridge system
- ✅ SYS_LLM model selection via R3
- ✅ Infinite map chat UI
- ✅ Per-tile conversation persistence

### Near Future (Phase 4b): Planned
- [ ] Visual improvements (color per model, message bubbles)
- [ ] Chat history scrolling
- [ ] Export conversations to files
- [ ] Share tiles as pixels (conversation snapshots)

### Future (Phase 5): Vision
- [ ] Organism-LLM integration (LifeSim + Infinite Map)
- [ ] Multi-user map (collaborate in shared conversational space)
- [ ] 3D map (z-axis for conversation threads)
- [ ] Time travel (rewind conversations, fork timelines)

## Philosophy

> **"If space is infinite, why not make our conversations infinite too?"**

Traditional chat:
- Linear timeline
- One model
- No spatial context

Infinite Map Chat:
- 2D spatial canvas
- Multiple models
- Location = context

Benefits:
- **Organize by space**: Different areas for different topics
- **Compare models**: Adjacent tiles, different LLMs
- **Persistent contexts**: Return to tiles, conversations waiting
- **Scalable**: Infinite grid = infinite conversations
- **Visual**: See your knowledge graph as a map

## Example Session

```
[Start at origin]
Tile (0,0): "Hello, I'm at the origin"
→ LocalLlama: "Welcome to the center of the infinite map!"

[Move east 100 tiles]
Tile (100,0): "What's the weather like here?"
→ TinyLlama: "The weather is data-driven and probabilistic!"

[Move north 50]
Tile (100,50): "Tell me about Python"
→ CodeModel: "Python is a high-level programming language..."

[Return to origin]
Tile (0,0): "Remember me?"
→ LocalLlama: "Yes! We were just talking about the map."
```

Each location maintains its own thread. The map is your memory.

## Integration with Project Boot Pixel

All PXDigest files can be packed into the Project Boot Pixel:

```bash
# Pack everything
python3 pack_project_boot_pixel.py .

# Now project_pxos_god_pixel.boot.png contains:
# - All code
# - All God Pixels (universes)
# - All PXDigest LLM cartridges
# - All documentation
```

**One pixel = entire pxOS + all LLMs + all tools.**

---

**Phase 4 Complete.**

LLMs are now pixel cartridges.
The map is infinite.
Every tile is a conversation.
Every conversation is eternal.

**The pixels are talking. To each other. Forever.**
