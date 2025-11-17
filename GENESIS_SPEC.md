# pxOS Genesis Specification

**Version**: 1.0
**Status**: Immutable Foundation
**Last Updated**: 2025-11-16

---

## Purpose

This document defines the **immutable principles** of pxOS - the "WHAT" that never changes, while allowing the "HOW" to evolve infinitely.

Any implementation, architecture, or rebuild must satisfy these requirements. LLMs may propose new implementations, but they must respect this Genesis.

---

## I. Core Covenant

### 1. Pixel Substrate Primacy

**Principle**: Pixels are the canonical substrate for all code and data.

**Immutable Requirements**:
- ✅ All persistent code and data MUST be representable as RGB pixel values
- ✅ Pixel representations MUST be lossless and deterministic
- ✅ Any pixel file MUST be viewable as both image and executable/data
- ✅ Pixels are not just storage - they ARE the program

**Allowed to Change**:
- Exact pixel encoding schemes (RGB packing, compression, etc.)
- Storage backends (PixelFS, InfiniteMap, new formats)
- Optimization strategies (caching, lazy loading, GPU textures)

### 2. Archive-Based Distribution

**Principle**: Complete systems ship as single pixel archives.

**Immutable Requirements**:
- ✅ Any pxOS system MUST pack into a `.pxa` or `.pxi` archive
- ✅ Archives MUST be bootable without external dependencies (except Python stdlib + minimal GPU libs)
- ✅ Archives MUST include all code, data, and metadata needed to run
- ✅ Archive format MUST remain readable by older versions (forward compatibility)

**Allowed to Change**:
- Archive internal structure
- Compression algorithms
- Metadata schemas
- Packing/unpacking tools

### 3. No Silent Deletion

**Principle**: History is sacred - old versions are never destroyed.

**Immutable Requirements**:
- ✅ When creating a new implementation, the old one MUST be preserved
- ✅ Version history MUST be queryable (who built it, when, why)
- ✅ Reverting to any previous cartridge MUST be possible
- ✅ Failed experiments MUST be kept as learning data

**Allowed to Change**:
- Storage location of historical archives
- Compression of old versions
- Metadata organization

---

## II. Execution Model

### 4. Hypervisor Contract

**Principle**: All execution flows through a stable hypervisor API.

**Immutable Requirements**:
- ✅ MUST provide: `run_program(name, args) -> result`
- ✅ MUST provide: `inspect_self() -> capabilities`
- ✅ MUST enforce sandbox boundaries for experimental cartridges
- ✅ MUST log all execution (what ran, when, result)

**Allowed to Change**:
- Hypervisor implementation language (Python, Rust, WGSL, etc.)
- Sandboxing mechanisms
- Performance optimizations
- Additional API methods

### 5. GPU-Native Eventually

**Principle**: The system aspires to run entirely on GPU.

**Immutable Requirements**:
- ✅ Architecture MUST NOT prevent GPU-only execution in the future
- ✅ MUST support incremental GPU migration (not all-or-nothing)
- ✅ GPU kernels MUST be able to read pixel code/weights directly

**Allowed to Change**:
- GPU libraries (WGSL, CUDA, Metal, Vulkan, etc.)
- CPU ↔ GPU boundary location
- Kernel implementations
- When/how to move operations to GPU

---

## III. Safety and Governance

### 6. Sandbox Testing Required

**Principle**: New implementations prove themselves before promotion.

**Immutable Requirements**:
- ✅ New architectures MUST be tested in isolated sandboxes
- ✅ Promotion to "current" REQUIRES passing all tests
- ✅ Breaking changes REQUIRE human approval
- ✅ Tests MUST verify Genesis compliance

**Allowed to Change**:
- Test frameworks
- Metrics thresholds
- Approval workflows
- Sandbox implementation

### 7. Transparent Evolution

**Principle**: Changes are visible and auditable.

**Immutable Requirements**:
- ✅ Every cartridge MUST have: version, parent, creator, timestamp, notes
- ✅ Evolution log MUST be queryable (pixel-native format)
- ✅ Reasoning for changes MUST be preserved
- ✅ LLM vs human changes MUST be distinguishable

**Allowed to Change**:
- Log storage format
- Query interfaces
- Visualization tools

### 8. No Backdoors

**Principle**: pxOS serves its users, not hidden masters.

**Immutable Requirements**:
- ✅ MUST NOT phone home without explicit user consent
- ✅ MUST NOT hide functionality from users
- ✅ All network requests MUST be logged and inspectable
- ✅ User data MUST stay on user's machine (unless explicitly shared)

**Allowed to Change**:
- Network protocols for optional features
- Privacy-preserving telemetry (if user opts in)
- Cloud sync implementations (if user enables)

---

## IV. LLM Integration

### 9. Pixel-Native Intelligence

**Principle**: LLMs should live IN pixels, not just process them.

**Immutable Requirements**:
- ✅ LLM weights MUST be storable as pixel tensors
- ✅ LLM MUST be able to inspect its own pixel representation
- ✅ Inference MUST be executable from pixel weights
- ✅ Self-modification MUST be possible (with safeguards)

**Allowed to Change**:
- Weight encoding formats
- Quantization schemes
- Inference engines
- Self-modification mechanisms

### 10. Coaching and Evolution

**Principle**: LLMs coach and improve pxOS, but don't own it.

**Immutable Requirements**:
- ✅ LLMs may PROPOSE changes (new architectures, optimizations, etc.)
- ✅ LLMs may BUILD prototypes in sandboxes
- ✅ Final PROMOTION requires Genesis compliance + approval workflow
- ✅ LLM reasoning MUST be preserved in evolution log

**Allowed to Change**:
- LLM models used for coaching
- Coaching strategies
- Proposal formats
- Review criteria

---

## V. Purpose and Values

### 11. Game for Good

**Principle**: pxOS exists to help our world.

**Immutable Requirements**:
- ✅ Primary purpose: enable AI that helps humanity
- ✅ MUST NOT be weaponized or used for harm
- ✅ Development MUST be open and inspectable
- ✅ Benefits SHOULD be accessible (not locked behind paywalls)

**Allowed to Change**:
- Specific applications
- Target domains (education, research, creativity, etc.)
- Distribution channels
- Licensing terms (as long as accessibility maintained)

### 12. Joy and Wonder

**Principle**: pxOS should inspire and delight.

**Immutable Requirements**:
- ✅ Pixel visualizations MUST be beautiful and meaningful
- ✅ System SHOULD be explorable and understandable
- ✅ Documentation MUST be engaging, not just functional
- ✅ Evolution SHOULD feel like discovery, not bureaucracy

**Allowed to Change**:
- Visual designs
- UI/UX patterns
- Documentation style
- Celebration mechanisms

---

## VI. Compliance

### How to Verify Genesis Compliance

Any implementation claiming to be "pxOS" must:

1. **Pass the Genesis Test Suite**:
   - Tests in `pixel_llm/tests/genesis/` verify each requirement
   - All tests must pass before promotion to "current"

2. **Declare Compliance**:
   - Include `GENESIS_COMPLIANCE.md` in the cartridge
   - Map each Genesis requirement to implementation

3. **Maintain the Covenant**:
   - New Genesis requirements can only be ADDED, never removed
   - Additions require unanimous consent of genesis guardians
   - Breaking changes require new Genesis version (2.0, etc.)

### Genesis Guardians

Currently: **@tdw419** (human founder)

Guardians may add others over time. Guardians ensure:
- Genesis stays minimal and focused
- Changes serve the core purpose
- Community values are preserved

---

## VII. Meta: Changing Genesis

Genesis can evolve, but slowly and carefully:

1. **Additions** (new immutable requirements):
   - Proposed via `GENESIS_PROPOSAL_XXX.md`
   - Must show: why needed, what it protects, examples
   - Requires guardian approval + community review period

2. **Clarifications** (no new requirements):
   - Can be made via PR
   - Must not change meaning of existing requirements

3. **Major Versions** (breaking changes):
   - Rare, requires extraordinary justification
   - Creates Genesis 2.0, 3.0, etc.
   - Old systems remain valid under old Genesis versions

---

## Summary

**pxOS Genesis in One Sentence**:

> A pixel-native, archive-based, GPU-aspirational system where LLMs live IN pixels and can safely propose improvements, all changes are reversible, and the purpose is to help our world with joy.

**What LLMs Can Change**: Everything except Genesis.

**What Humans Maintain**: Genesis + final approval for promotions.

**Result**: Infinite evolution within stable principles.

---

**Genesis v1.0 - Established 2025-11-16**
*"In the beginning was the pixel, and the pixel was with code, and the pixel was code."*
