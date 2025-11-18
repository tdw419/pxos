# pxOS AI Build System

**Automated OS Development with LM Studio + Self-Expanding Pixel Networks**

This system uses AI to automatically generate, build, and improve pxOS by combining:

- **LM Studio**: Local LLM for code generation
- **Pixel Networks**: Self-expanding knowledge base that learns from each build
- **Primitive Generator**: AI that writes x86 assembly as WRITE/DEFINE commands
- **Automated Build Loop**: Iterative improvement system

---

## AI Response Format

All AI-generated primitives must conform to the `tools/ai_primitives.schema.json` schema. Free-form text is rejected. The generator will validate the JSON response and retry if it is invalid.

This ensures that the AI's output is always predictable and machine-readable, which is critical for a reliable automated build system.

---
## System Components

### 1. LM Studio Bridge (`pxvm/integration/lm_studio_bridge.py`)

A simple bridge to communicate with a local LLM running in LM Studio.

### 2. Primitive Generator (`tools/ai_primitive_generator.py`)

AI-powered code generator that converts feature descriptions to pxOS primitives, adhering to a strict JSON schema.

### 3. Automated Builder (`tools/auto_build_pxos.py`)

(Future work) A full automation system that will analyze the current pxOS state, generate a build plan, and implement features step-by-step.
