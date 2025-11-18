# pxOS Concept (Living Document)

## Vision
pxOS is a GPU-native, pixel-driven operating system where pixels are the instruction space.
All computation, storage, and IO are mapped to visual structures in VRAM.

## Current Focus
- Define minimal GPU-side interpreter (shader or VM) that executes simple pixel opcodes.
- Expose a Python-facing terminal that can send programs into VRAM and read back results.

## Constraints
- Keep all bootstraps as small and understandable as possible.
- Prefer clear, LLM-debbugable layers over cleverness.
- All components should be explainable to another AI in < 2 pages of text.

## Open Questions
- How should pxOS represent processes in VRAM?
- Minimal opcode set for v0.1?
- How to snapshot VRAM into a reproducible “px image” format?
