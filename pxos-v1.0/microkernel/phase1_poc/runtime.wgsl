// pxOS GPU Runtime - WebGPU Compute Shader
// Executes pixel-encoded operating system (os.pxi)
// Each thread fetches and executes instructions from texture

// Bindings
@group(0) @binding(0) var os_code: texture_2d<f32>;          // os.pxi loaded as texture
@group(0) @binding(1) var<storage, read_write> cpu_request_mailbox: array<atomic<u32>>;  // CPU ↔ GPU communication
@group(0) @binding(2) var<storage, read_write> memory: array<u32>;         // OS memory
@group(0) @binding(3) var<storage, read_write> vga_buffer: array<u32>;     // VGA text mode buffer
@group(0) @binding(4) var<storage, read_write> debug_output: array<u32>;   // Debug output buffer
@group(0) @binding(5) var<storage, read_write> registers: array<u32>;      // Register file (per-thread)

// Opcodes (must match create_os_pxi.py)
const OP_NOP: u32              = 0x00u;
const OP_PRINT_CHAR: u32       = 0x01u;
const OP_PRINT_STR: u32        = 0x02u;
const OP_SET_CURSOR: u32       = 0x03u;
const OP_CLEAR_SCREEN: u32     = 0x04u;
const OP_LOAD: u32             = 0x10u;
const OP_STORE: u32            = 0x11u;
const OP_MOVE: u32             = 0x12u;
const OP_ADD: u32              = 0x20u;
const OP_SUB: u32              = 0x21u;
const OP_INC: u32              = 0x22u;
const OP_DEC: u32              = 0x23u;
const OP_JMP: u32              = 0x30u;
const OP_JZ: u32               = 0x31u;
const OP_JNZ: u32              = 0x32u;
const OP_CALL: u32             = 0x33u;
const OP_RET: u32              = 0x34u;
const OP_CPU_REQ: u32          = 0xF0u;
const OP_YIELD: u32            = 0xF1u;
const OP_MMIO_WRITE_UART: u32  = 0x80u;
const OP_CPU_HALT: u32         = 0x8Fu;
const OP_HALT: u32             = 0xFFu;

// Mailbox request types
const REQ_NONE: u32            = 0u;
const REQ_MMIO_WRITE: u32      = 1u;
const REQ_MMIO_READ: u32       = 2u;
const REQ_HALT: u32            = 3u;

// VGA text mode constants
const VGA_WIDTH: u32 = 80u;
const VGA_HEIGHT: u32 = 25u;
const VGA_SIZE: u32 = 2000u;  // 80 * 25

// Debug buffer
const DEBUG_BUFFER_SIZE: u32 = 4096u;

struct Instruction {
    opcode: u32,
    arg1: u32,
    arg2: u32,
    arg3: u32,
}

// Decode RGBA pixel into instruction
fn decode_pixel(pixel: vec4<f32>) -> Instruction {
    // RGBA values are 0.0-1.0, convert to 0-255
    let r = u32(pixel.r * 255.0);
    let g = u32(pixel.g * 255.0);
    let b = u32(pixel.b * 255.0);
    let a = u32(pixel.a * 255.0);

    return Instruction(r, g, b, a);
}

// Fetch instruction from os.pxi texture
fn fetch_instruction(pc: u32, width: u32) -> Instruction {
    let x = i32(pc % width);
    let y = i32(pc / width);
    let pixel = textureLoad(os_code, vec2<i32>(x, y), 0);
    return decode_pixel(pixel);
}

// Get register index for this thread
fn get_register_base(thread_id: u32) -> u32 {
    return thread_id * 8u;  // 8 registers per thread
}

// Read register
fn read_reg(thread_id: u32, reg: u32) -> u32 {
    let base = get_register_base(thread_id);
    return registers[base + (reg & 7u)];
}

// Write register
fn write_reg(thread_id: u32, reg: u32, value: u32) {
    let base = get_register_base(thread_id);
    registers[base + (reg & 7u)] = value;
}

// VGA operations
var<private> vga_cursor_x: u32 = 0u;
var<private> vga_cursor_y: u32 = 0u;

fn vga_put_char(char: u32, color: u32) {
    if (vga_cursor_x >= VGA_WIDTH) {
        vga_cursor_x = 0u;
        vga_cursor_y += 1u;
    }
    if (vga_cursor_y >= VGA_HEIGHT) {
        vga_cursor_y = 0u;  // Wrap around (TODO: scroll)
    }

    let offset = vga_cursor_y * VGA_WIDTH + vga_cursor_x;
    // VGA format: low byte = char, high byte = color
    vga_buffer[offset] = (color << 8u) | char;

    vga_cursor_x += 1u;
}

fn vga_clear_screen(color: u32) {
    for (var i = 0u; i < VGA_SIZE; i++) {
        vga_buffer[i] = (color << 8u) | u32(' ');
    }
    vga_cursor_x = 0u;
    vga_cursor_y = 0u;
}

fn vga_set_cursor(x: u32, y: u32) {
    vga_cursor_x = x;
    vga_cursor_y = y;
}

// CPU request mailbox
// Format: [request_type, arg1, arg2, arg3, ...]
fn request_cpu_mmio_write(port: u32, value: u32) {
    // Find free mailbox slot
    for (var i = 0u; i < 64u; i += 4u) {
        let req_type = atomicLoad(&cpu_request_mailbox[i]);
        if (req_type == REQ_NONE) {
            // Claim this slot
            atomicStore(&cpu_request_mailbox[i], REQ_MMIO_WRITE);
            atomicStore(&cpu_request_mailbox[i + 1u], port);
            atomicStore(&cpu_request_mailbox[i + 2u], value);
            atomicStore(&cpu_request_mailbox[i + 3u], 0u);
            return;
        }
    }
    // Mailbox full - drop request (TODO: better handling)
}

fn request_cpu_halt() {
    atomicStore(&cpu_request_mailbox[0], REQ_HALT);
}

// Debug output
var<private> debug_write_pos: u32 = 0u;

fn debug_write_char(char: u32) {
    if (debug_write_pos < DEBUG_BUFFER_SIZE) {
        debug_output[debug_write_pos] = char;
        debug_write_pos += 1u;
    }
}

fn debug_write_string(str_ptr: u32, len: u32) {
    for (var i = 0u; i < len; i++) {
        debug_write_char(memory[str_ptr + i]);
    }
}

// Execute one instruction
fn execute_instruction(inst: Instruction, thread_id: u32) -> u32 {
    // Returns next PC (current PC + 1, or jump target)

    switch (inst.opcode) {
        case OP_NOP: {
            // Do nothing
        }

        case OP_PRINT_CHAR: {
            // arg1 = char, arg2 = color, arg3 = flags
            vga_put_char(inst.arg1, inst.arg2);
        }

        case OP_PRINT_STR: {
            // arg1 = offset high, arg2 = offset low, arg3 = color
            let offset = (inst.arg1 << 8u) | inst.arg2;
            // TODO: implement string printing from memory
        }

        case OP_SET_CURSOR: {
            // arg1 = x, arg2 = y
            vga_set_cursor(inst.arg1, inst.arg2);
        }

        case OP_CLEAR_SCREEN: {
            // arg1 = color
            vga_clear_screen(inst.arg1);
        }

        case OP_LOAD: {
            // arg1 = dest_reg, arg2 = addr_hi, arg3 = addr_lo
            let addr = (inst.arg2 << 8u) | inst.arg3;
            let value = memory[addr];
            write_reg(thread_id, inst.arg1, value);
        }

        case OP_STORE: {
            // arg1 = src_reg, arg2 = addr_hi, arg3 = addr_lo
            let addr = (inst.arg2 << 8u) | inst.arg3;
            let value = read_reg(thread_id, inst.arg1);
            memory[addr] = value;
        }

        case OP_MOVE: {
            // arg1 = dest_reg, arg2 = src_reg
            let value = read_reg(thread_id, inst.arg2);
            write_reg(thread_id, inst.arg1, value);
        }

        case OP_ADD: {
            // arg1 = dest, arg2 = src1, arg3 = src2
            let val1 = read_reg(thread_id, inst.arg2);
            let val2 = read_reg(thread_id, inst.arg3);
            write_reg(thread_id, inst.arg1, val1 + val2);
        }

        case OP_SUB: {
            // arg1 = dest, arg2 = src1, arg3 = src2
            let val1 = read_reg(thread_id, inst.arg2);
            let val2 = read_reg(thread_id, inst.arg3);
            write_reg(thread_id, inst.arg1, val1 - val2);
        }

        case OP_INC: {
            // arg1 = reg
            let value = read_reg(thread_id, inst.arg1);
            write_reg(thread_id, inst.arg1, value + 1u);
        }

        case OP_DEC: {
            // arg1 = reg
            let value = read_reg(thread_id, inst.arg1);
            write_reg(thread_id, inst.arg1, value - 1u);
        }

        case OP_JMP: {
            // arg1 = target_hi, arg2 = target_lo
            let target = (inst.arg1 << 8u) | inst.arg2;
            return target;  // Jump!
        }

        case OP_JZ: {
            // arg1 = reg, arg2 = target_hi, arg3 = target_lo
            let value = read_reg(thread_id, inst.arg1);
            if (value == 0u) {
                let target = (inst.arg2 << 8u) | inst.arg3;
                return target;
            }
        }

        case OP_JNZ: {
            // arg1 = reg, arg2 = target_hi, arg3 = target_lo
            let value = read_reg(thread_id, inst.arg1);
            if (value != 0u) {
                let target = (inst.arg2 << 8u) | inst.arg3;
                return target;
            }
        }

        case OP_MMIO_WRITE_UART: {
            // arg1 = char, arg2 = port_hi, arg3 = port_lo
            let port = (inst.arg2 << 8u) | inst.arg3;
            request_cpu_mmio_write(port, inst.arg1);

            // Also write to debug buffer
            debug_write_char(inst.arg1);
        }

        case OP_CPU_HALT: {
            request_cpu_halt();
        }

        case OP_HALT: {
            // Halt this thread
            return 0xFFFFFFFFu;  // Special value to stop execution
        }

        default: {
            // Unknown opcode - treat as NOP
        }
    }

    return 0xFFFFFFFFu;  // Default: don't override PC
}

// Main compute shader entry point
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;

    // Get texture dimensions
    let dimensions = textureDimensions(os_code);
    let width = u32(dimensions.x);
    let height = u32(dimensions.y);
    let max_instructions = width * height;

    // Each thread starts at its own offset
    // Thread 0 executes from PC=0, Thread 1 from PC=1, etc.
    // This allows parallel execution of different parts of the OS
    var pc = thread_id;

    // Execution loop (simplified for Phase 1 POC)
    // In Phase 2, this will be a persistent loop with synchronization
    for (var cycle = 0u; cycle < 1000u; cycle++) {
        if (pc >= max_instructions) {
            break;  // PC out of bounds
        }

        // Fetch
        let inst = fetch_instruction(pc, width);

        // Decode and Execute
        let next_pc = execute_instruction(inst, thread_id);

        // Update PC
        if (next_pc == 0xFFFFFFFFu) {
            // HALT or normal increment
            if (inst.opcode == OP_HALT) {
                break;  // Stop this thread
            }
            pc += 1u;  // Next instruction
        } else {
            pc = next_pc;  // Jump target
        }
    }
}

// Alternative: Single-threaded execution mode (for debugging)
@compute @workgroup_size(1)
fn main_single_thread(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Only thread 0 executes
    if (global_id.x != 0u) {
        return;
    }

    let dimensions = textureDimensions(os_code);
    let width = u32(dimensions.x);
    let height = u32(dimensions.y);
    let max_instructions = width * height;

    var pc = 0u;

    // Execute sequentially
    loop {
        if (pc >= max_instructions) {
            break;
        }

        let inst = fetch_instruction(pc, width);

        if (inst.opcode == OP_HALT) {
            break;
        }

        let next_pc = execute_instruction(inst, 0u);

        if (next_pc == 0xFFFFFFFFu) {
            pc += 1u;
        } else {
            pc = next_pc;
        }
    }

    // Signal completion
    request_cpu_halt();
}
