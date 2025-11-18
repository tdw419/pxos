# PXCARTRIDGE v0 Format Specification

**Version**: 0.1
**Status**: Draft
**Author**: pxOS Project
**Date**: 2025-01-18

---

## Overview

PXCARTRIDGE is a **pixel-based container format** for storing binary programs, metadata, and porting information as images. It enables systematic conversion and porting of foreign software to pxOS.

### Key Benefits

1. **Universal Storage**: Any binary (ELF, PE, raw) → standardized pixel format
2. **Rich Metadata**: Store ISA, ABI, dependencies, licensing in image headers
3. **Visual Archival**: Programs become images that can be versioned, displayed, and stored in pixel networks
4. **Centralized Porting**: One pipeline handles all foreign programs
5. **Data-Driven Translation**: LLMs can reason about programs as structured image data

---

## Format Structure

A PXCARTRIDGE image (`.pxcart.png`) has three sections:

```
┌─────────────────────────────────────────┐
│  HEADER ROWS (Metadata as pixels)      │  ← 16-32 rows
├─────────────────────────────────────────┤
│  PAYLOAD (Binary data as pixels)       │  ← Variable size
├─────────────────────────────────────────┤
│  CHECKSUM ROW (Integrity verification) │  ← 1 row
└─────────────────────────────────────────┘
```

### Image Properties

- **Format**: PNG (lossless, 8-bit RGB or RGBA)
- **Width**: 256, 512, or 1024 pixels (power of 2 for efficiency)
- **Height**: Variable (header + payload + checksum)
- **Encoding**: RGBA where each pixel encodes 1-4 bytes

---

## Section 1: Header Rows

Header stores metadata as **pixel-encoded fields**. Each field occupies a fixed position.

### Header Layout (v0)

**Row 0: Magic Number & Version**

| Pixels 0-3 | Pixels 4-7 | Pixels 8-11 | Pixels 12-15 | Rest |
|------------|------------|-------------|--------------|------|
| Magic: `PXCT` | Version: `0001` | Reserved | Reserved | Padding (black) |

**Row 1: Architecture & ABI**

| Pixels 0-7 | Pixels 8-15 | Rest |
|------------|-------------|------|
| ISA (e.g., `x86_32`, `arm64`) | ABI/OS (e.g., `elf_linux`, `pe_win32`) | Padding |

**Row 2: Entry Point & Size**

| Pixels 0-7 | Pixels 8-15 | Rest |
|------------|-------------|------|
| Entry point address (8 bytes, little-endian) | Binary size (8 bytes) | Padding |

**Row 3: Flags & Compression**

| Pixels 0-3 | Pixels 4-7 | Pixels 8-11 | Rest |
|------------|------------|-------------|------|
| Flags (bitfield) | Compression (e.g., `zlib`, `lz4`, `none`) | Reserved | Padding |

**Flags Bitfield** (32-bit):
- Bit 0: Compressed (1 = yes, 0 = no)
- Bit 1: Contains IR/lifted code (1 = yes, 0 = no)
- Bit 2: Partial port (has pxOS stubs) (1 = yes, 0 = no)
- Bit 3-7: Reserved
- Bit 8-15: Porting status (0 = raw, 1 = analyzed, 2 = lifted, 3 = ported, 4 = tested)
- Bit 16-31: Reserved for future use

**Row 4-7: Dependencies & Requirements**

Encodes required libraries, OS features, etc. as null-terminated strings.

**Row 8-11: Licensing & Attribution**

Encodes license identifier (e.g., `MIT`, `GPL3`, `BSD2`) and author info.

**Row 12-15: Checksums & Hashes**

| Pixels 0-15 | Pixels 16-31 | Rest |
|-------------|--------------|------|
| SHA256 of payload (32 bytes) | CRC32 of header (4 bytes) | Padding |

**Row 16+: Extended Metadata (Optional)**

Can include:
- Disassembly snapshot
- Symbol table
- Relocation info
- Notes from original porter
- Test cases

---

## Section 2: Payload

Binary data encoded as pixels. Each pixel stores up to 4 bytes (RGBA).

### Encoding Scheme

**Standard (RGBA):**
```
Pixel = [R, G, B, A]
  → Byte 0 = R
  → Byte 1 = G
  → Byte 2 = B
  → Byte 3 = A
```

**For 3-byte RGB:**
```
Pixel = [R, G, B]
  → Byte 0 = R
  → Byte 1 = G
  → Byte 2 = B
```

**Example:**
Binary: `48 65 6C 6C 6F 00` (Hello\0)
→ Pixels: `[48, 65, 6C, 6C]`, `[6F, 00, 00, 00]`

### Layout

Payload starts at row 16 (or after extended metadata if present).

Rows filled left-to-right, top-to-bottom. Padding with zeros if binary size is not a multiple of (width × 4).

---

## Section 3: Checksum Row

Final row contains integrity verification data.

| Pixels 0-15 | Rest |
|-------------|------|
| CRC32 of entire image (4 bytes) | Reserved/Padding |

---

## Pixel Encoding Details

### ASCII Strings

Encoded as UTF-8 bytes, null-terminated:

```
"x86_32" → 78 38 36 5F 33 32 00 (7 bytes)
→ Pixels: [78, 38, 36, 5F], [33, 32, 00, 00]
```

### Multi-Byte Integers

Stored **little-endian**:

```
Entry point: 0x08048000 (Linux ELF typical)
→ Bytes: 00 80 04 08 00 00 00 00 (8 bytes, 64-bit)
→ Pixels: [00, 80, 04, 08], [00, 00, 00, 00]
```

### Bitfields

Stored as 32-bit little-endian integers:

```
Flags = 0x00000103 (compressed + ported)
→ Bytes: 03 01 00 00
→ Pixel: [03, 01, 00, 00]
```

---

## Supported ISAs (Row 1, Pixels 0-7)

Identifier strings (null-terminated):

| ISA | String |
|-----|--------|
| x86 32-bit | `x86_32` |
| x86 64-bit | `x86_64` |
| ARM 32-bit | `arm32` |
| ARM 64-bit | `arm64` |
| MIPS | `mips` |
| RISC-V 32 | `riscv32` |
| RISC-V 64 | `riscv64` |
| WebAssembly | `wasm32` |
| Custom/Unknown | `unknown` |

---

## Supported ABIs (Row 1, Pixels 8-15)

| ABI/OS | String |
|--------|--------|
| Linux ELF | `elf_linux` |
| Windows PE | `pe_win32` |
| Windows PE 64 | `pe_win64` |
| macOS Mach-O | `macho` |
| Raw binary | `raw_bin` |
| DOS COM | `dos_com` |
| DOS EXE | `dos_exe` |
| pxOS primitive | `pxos_prim` |
| pxOS native | `pxos_native` |

---

## Compression Types (Row 3, Pixels 4-7)

| Type | String |
|------|--------|
| None | `none` |
| zlib | `zlib` |
| gzip | `gzip` |
| LZ4 | `lz4` |
| LZMA | `lzma` |

---

## Example: Minimal Cartridge

### Input: `hello.bin` (raw x86 binary, 512 bytes)

```
Entry point: 0x7C00
ISA: x86_32
ABI: raw_bin
Size: 512 bytes
Compressed: No
```

### Resulting .pxcart.png

**Dimensions**: 256 × 20 pixels (16 header + 2 payload + 1 checksum + 1 padding)

**Row 0:**
```
Pixels 0-3:   [50, 58, 43, 54]  # "PXCT"
Pixels 4-7:   [30, 30, 30, 31]  # "0001"
Pixels 8+:    [00, 00, 00, 00]  # Padding
```

**Row 1:**
```
Pixels 0-6:   [78, 38, 36, 5F], [33, 32, 00, 00]  # "x86_32\0"
Pixels 7-15:  [72, 61, 77, 5F], [62, 69, 6E, 00]  # "raw_bin\0"
```

**Row 2:**
```
Pixels 0-7:   [00, 7C, 00, 00], [00, 00, 00, 00]  # Entry: 0x7C00
Pixels 8-15:  [00, 02, 00, 00], [00, 00, 00, 00]  # Size: 512
```

**Row 3:**
```
Pixels 0-3:   [00, 00, 00, 00]  # Flags: 0 (no compression)
Pixels 4-7:   [6E, 6F, 6E, 65]  # "none"
```

(Rows 4-15: Padding or optional metadata)

**Rows 16-17:** Binary payload (512 bytes = 128 pixels at 4 bytes each)

**Row 18:** Checksum

---

## Tools

### make_pxcart.py

Create a pixel cartridge from a binary:

```bash
python3 tools/make_pxcart.py hello.bin \
  --isa x86_32 \
  --abi raw_bin \
  --entry 0x7C00 \
  --output hello.pxcart.png
```

### read_pxcart.py

Extract metadata and binary from cartridge:

```bash
# Show metadata
python3 tools/read_pxcart.py hello.pxcart.png --info

# Extract binary
python3 tools/read_pxcart.py hello.pxcart.png --extract hello_extracted.bin

# Verify checksums
python3 tools/read_pxcart.py hello.pxcart.png --verify
```

### port_pxcart.py

Attempt automatic porting:

```bash
# Emulate mode (run under emulator)
python3 tools/port_pxcart.py hello.pxcart.png --mode emulate

# Lift mode (translate to IR)
python3 tools/port_pxcart.py hello.pxcart.png --mode lift --output hello_ir.json

# Recompile mode (IR → pxOS primitives)
python3 tools/port_pxcart.py hello_ir.json --mode recompile --output hello_pxos.txt
```

---

## Porting Workflows

### Workflow 1: Emulation

```
Binary → PXCART → Decoder → Emulator → Runs on pxOS
```

1. Convert binary to `.pxcart.png`
2. pxOS loader decodes header, extracts ISA/ABI
3. Launches appropriate emulator (e.g., x86 emulator for x86 binaries)
4. Binary runs under emulation

**Pros**: Works for any binary, no translation needed
**Cons**: Performance overhead, requires emulator

---

### Workflow 2: Lift & Translate

```
Binary → PXCART → Lifter → IR → Translator → pxOS Primitives → Native
```

1. Convert binary to `.pxcart.png`
2. Lifter disassembles and converts to intermediate representation (IR)
3. Translator maps IR to pxOS primitives or pxVM bytecode
4. Result is pixel-native code

**Pros**: Native performance, portable
**Cons**: Complex, not all code can be lifted

---

### Workflow 3: Hybrid (Selective Patching)

```
Binary → PXCART → Patcher → Modified Binary → Runs with pxOS stubs
```

1. Keep most binary unchanged
2. Identify OS-specific calls (syscalls, I/O, GUI)
3. Replace with pxOS-compatible stubs
4. Store patches in PXCART extended metadata

**Pros**: Easier than full translation
**Cons**: Still needs some emulation

---

## Integration with pxOS

### Cartridge Loader (Future Milestone)

Add to pxOS:

```
tools/pxos_cart_loader.asm
```

Responsibilities:
- Read `.pxcart.png` from disk (FAT12 or raw sectors)
- Decode header pixels → metadata struct
- Decide: emulate, translate, or patch?
- Execute appropriate handler

### Pixel Vault Storage

Store cartridges in **Pixel Vault**:
- Each cartridge is one image
- Indexed by hash (SHA256 from header)
- Visual catalog of all software
- LLM can browse and analyze

---

## Future Extensions (v0.2+)

1. **Differential Cartridges**: Store only diffs from a base cartridge
2. **Multi-Binary Cartridges**: Pack multiple binaries (program + libs)
3. **Execution Traces**: Include sample I/O for testing
4. **Source Code Embedding**: Store original source alongside binary
5. **Neural Metadata**: Embeddings for LLM-powered analysis

---

## Security Considerations

- **Checksums**: Verify integrity before loading
- **Sandboxing**: Emulated programs should be sandboxed
- **Malware Scanning**: Check binaries before creating cartridges
- **Signature**: Future versions may support code signing

---

## References

- [pxOS Primitives](primitives.md)
- [pxOS Architecture](architecture.md)
- [Pixel Vault Spec](#) (TBD)
- [ELF Format](https://refspecs.linuxfoundation.org/elf/elf.pdf)
- [PE Format](https://docs.microsoft.com/en-us/windows/win32/debug/pe-format)

---

## Changelog

### v0.1 (2025-01-18)
- Initial specification
- Basic header layout
- Standard encoding schemes
- Tool outlines

---

**Status**: This is a draft specification. Feedback and contributions welcome!
