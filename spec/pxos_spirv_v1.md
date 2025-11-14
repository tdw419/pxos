# pxOS SPIR-V Instruction Set Specification v0.1

This document specifies the initial, minimal subset of SPIR-V instructions and custom extensions supported by the pxOS CPU bootstrap. The goal of this subset is to be just powerful enough to execute a "Hello, World!" program via a syscall.

## Binary Format

pxOS programs are stored in `.vram` files, which are raw binary dumps of a VRAM image. For the SPIR-V interpreter, the "code" section of this VRAM is interpreted as a sequence of 32-bit SPIR-V words.

- Endianness: Little Endian
- Word Size: 32-bit

## Standard SPIR-V Opcodes (Subset)

The following standard SPIR-V opcodes are supported in v0.1. Each instruction word is composed of a 16-bit word count and a 16-bit opcode.

| Opcode | Name | Description | Word Format |
|---|---|---|---|
| 17 | `OpCapability` | Declares a capability used by the module. We will require `Shader`. | `(1: word_count, 17: opcode), (capability)` |
| 19 | `OpEntryPoint` | Declares an entry point and its execution model. | `(3 + len(name): wc, 19: op), (model), (entry_id), (name_string), ...` |
| 43 | `OpTypeVoid` | Declares the void type. | `(2: wc, 43: op), (result_id)` |
| 44 | `OpTypeFunction` | Declares a function type. | `(4: wc, 44: op), (result_id), (return_type_id), (param_type_id)` |
| 48 | `OpTypePointer` | Declares a pointer type. | `(4: wc, 48: op), (result_id), (storage_class), (type_id)` |
| 49 | `OpConstant` | Declares a scalar constant. | `(3 + size: wc, 49: op), (type_id), (result_id), (value)` |
| 54 | `OpFunction` | Begins a function definition. | `(5: wc, 54: op), (result_type_id), (result_id), (function_control), (function_type_id)` |
| 56 | `OpFunctionEnd` | Ends a function definition. | `(1: wc, 56: op)` |
| 61 | `OpLoad` | Loads a value from memory. | `(4: wc, 61: op), (result_type_id), (result_id), (pointer_id)` |
| 62 | `OpStore` | Stores a value to memory. | `(3: wc, 62: op), (pointer_id), (object_id)` |
| 12 | `OpExtInstImport`| Imports an extended instruction set. | `(2 + len(name): wc, 12: op), (result_id), (name_string)` |
| 13 | `OpExtInst` | Executes an instruction from an extended instruction set. | `(5+: wc, 13: op), (result_type_id), (result_id), (set_id), (instruction_id), ...` |
| 253 | `OpReturn` | Returns from a function. | `(1: wc, 253: op)` |

## pxOS Custom Extended Instruction Set

pxOS defines its own extended instruction set for system calls. This set must be imported before use.

- **Set Name**: `"PXOS.syscalls"`

### Instructions

#### `SYS_PRINT`
- **Instruction ID**: 1
- **Description**: Prints a null-terminated string to the console. The operand must be a pointer to a sequence of `OpConstant` characters.
- **SPIR-V Usage**:
  ```spirv
  %syscall_set = OpExtInstImport "PXOS.syscalls"
  ...
  %string_ptr = OpConstant %u8 "Hello, World!"
  ...
  OpExtInst %void %syscall_set SYS_PRINT %string_ptr
  ```

This minimal set provides the foundation for loading data, performing basic operations, and interacting with the host system via syscalls, all within the standard SPIR-V framework.
