// pxOS GPU Runtime - Mailbox Protocol Handler
//
// This is the GPU-side implementation of the hardware mailbox protocol.
// The GPU acts as the primary execution engine and privilege broker.
//
// Architecture:
//   CPU (ring 3, unprivileged) → writes mailbox → GPU (ring 0, privileged)
//   GPU processes commands and updates status/response registers
//
// Register Layout (BAR0 + offsets):
//   0x0000: CMD_REG     (CPU writes)
//   0x0004: STATUS_REG  (GPU writes)
//   0x0008: RESP_REG    (GPU writes)
//   0x000C: DOORBELL    (CPU writes to trigger)
//
// Command Format (32-bit):
//   Bits 31-24: Opcode
//   Bits 23-16: Thread ID
//   Bits 15-0:  Payload (parameter/data)

// =============================================================================
// MEMORY BUFFERS
// =============================================================================

// Mailbox registers (mapped to BAR0 MMIO region)
@group(0) @binding(0) var<storage, read_write> mailbox_cmd: u32;
@group(0) @binding(1) var<storage, read_write> mailbox_status: u32;
@group(0) @binding(2) var<storage, read_write> mailbox_resp: u32;
@group(0) @binding(3) var<storage, read_write> mailbox_doorbell: u32;

// Serial port output buffer (for OP_UART_WRITE)
@group(0) @binding(4) var<storage, read_write> uart_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> uart_write_idx: atomic<u32>;

// System memory (for privileged operations)
@group(0) @binding(6) var<storage, read_write> system_memory: array<u32>;

// Performance counters
@group(0) @binding(7) var<storage, read_write> perf_counters: array<u32>;

// =============================================================================
// CONSTANTS
// =============================================================================

// Status bits
const STATUS_READY: u32 = 0x00000001u;
const STATUS_BUSY: u32 = 0x00000002u;
const STATUS_COMPLETE: u32 = 0x00000004u;
const STATUS_ERROR: u32 = 0x80000000u;

// Opcodes (must match mailbox_protocol.asm)
const OP_UART_WRITE: u32 = 0x80u;
const OP_UART_READ: u32 = 0x81u;
const OP_GPU_EXECUTE: u32 = 0x82u;
const OP_MMIO_READ: u32 = 0x83u;
const OP_MMIO_WRITE: u32 = 0x84u;
const OP_SYSCALL: u32 = 0x85u;
const OP_ALLOC_MEMORY: u32 = 0x86u;
const OP_FREE_MEMORY: u32 = 0x87u;
const OP_HALT: u32 = 0x8Fu;

// Performance counter indices
const PERF_COMMANDS_RECEIVED: u32 = 0u;
const PERF_COMMANDS_COMPLETED: u32 = 1u;
const PERF_COMMANDS_FAILED: u32 = 2u;
const PERF_TOTAL_CYCLES: u32 = 3u;

// =============================================================================
// MAIN MAILBOX HANDLER
// =============================================================================

@compute @workgroup_size(1, 1, 1)
fn mailbox_handler(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Check if doorbell has been rung
    let doorbell = mailbox_doorbell;
    if (doorbell == 0u) {
        return;  // No command pending
    }

    // Set status to BUSY
    mailbox_status = STATUS_BUSY;
    workgroupBarrier();

    // Read command
    let cmd = mailbox_cmd;
    let opcode = (cmd >> 24u) & 0xFFu;
    let tid = (cmd >> 16u) & 0xFFu;
    let payload = cmd & 0xFFFFu;

    // Increment performance counter
    atomicAdd(&perf_counters[PERF_COMMANDS_RECEIVED], 1u);

    // Dispatch based on opcode
    var result: u32 = 0u;
    var error: bool = false;

    switch (opcode) {
        case OP_UART_WRITE: {
            result = uart_write(payload);
        }
        case OP_UART_READ: {
            result = uart_read();
        }
        case OP_GPU_EXECUTE: {
            result = gpu_execute_kernel(payload);
        }
        case OP_MMIO_READ: {
            result = mmio_read(payload);
        }
        case OP_MMIO_WRITE: {
            mmio_write(payload & 0xFFu, (payload >> 8u) & 0xFFu);
            result = 0u;
        }
        case OP_SYSCALL: {
            result = handle_syscall(tid, payload);
        }
        case OP_ALLOC_MEMORY: {
            result = alloc_memory(payload);
        }
        case OP_FREE_MEMORY: {
            free_memory(payload);
            result = 0u;
        }
        case OP_HALT: {
            halt_system();
            result = 0u;
        }
        default: {
            // Unknown opcode
            error = true;
            result = 0xFFFFFFFFu;
        }
    }

    // Write response
    mailbox_resp = result;

    // Update status
    if (error) {
        mailbox_status = STATUS_ERROR;
        atomicAdd(&perf_counters[PERF_COMMANDS_FAILED], 1u);
    } else {
        mailbox_status = STATUS_COMPLETE;
        atomicAdd(&perf_counters[PERF_COMMANDS_COMPLETED], 1u);
    }

    // Clear doorbell
    mailbox_doorbell = 0u;
}

// =============================================================================
// OPCODE IMPLEMENTATIONS
// =============================================================================

fn uart_write(char_code: u32) -> u32 {
    // Write character to UART buffer
    let idx = atomicAdd(&uart_write_idx, 1u);
    uart_buffer[idx] = char_code & 0xFFu;
    return 0u;  // Success
}

fn uart_read() -> u32 {
    // Read character from UART (stub - would interface with hardware)
    return 0x00u;
}

fn gpu_execute_kernel(kernel_id: u32) -> u32 {
    // Execute a GPU kernel by ID
    // This would dispatch to other compute shaders
    // For now, just return success
    return kernel_id;
}

fn mmio_read(offset: u32) -> u32 {
    // Read from MMIO region (privileged operation)
    if (offset >= 1024u) {
        return 0xFFFFFFFFu;  // Out of bounds
    }
    return system_memory[offset];
}

fn mmio_write(offset: u32, value: u32) {
    // Write to MMIO region (privileged operation)
    if (offset < 1024u) {
        system_memory[offset] = value;
    }
}

fn handle_syscall(syscall_num: u32, arg: u32) -> u32 {
    // Handle system calls (CPU requesting privileged operations from GPU)
    switch (syscall_num) {
        case 0u: {  // sys_debug_print
            return uart_write(arg);
        }
        case 1u: {  // sys_get_time
            return get_timestamp();
        }
        default: {
            return 0xFFFFFFFFu;  // Unknown syscall
        }
    }
}

fn alloc_memory(size: u32) -> u32 {
    // Allocate memory (GPU manages all memory)
    // This is a stub - would interface with GPU memory allocator
    // For now, return a fake address
    return 0x10000000u + (size * 4096u);
}

fn free_memory(addr: u32) {
    // Free memory
    // Stub - would mark memory as free in allocator
}

fn halt_system() {
    // Halt the entire system
    // Set a flag that the host can check
    mailbox_status = STATUS_ERROR | 0x00000100u;  // Halt flag
}

fn get_timestamp() -> u32 {
    // Return GPU timestamp
    // In real implementation, would read GPU clock
    return perf_counters[PERF_TOTAL_CYCLES];
}

// =============================================================================
// CONTINUOUS POLLING LOOP
// =============================================================================

// This shader runs continuously, polling the doorbell
@compute @workgroup_size(1, 1, 1)
fn mailbox_poll_loop(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Infinite loop polling for commands
    loop {
        // Check doorbell
        if (mailbox_doorbell != 0u) {
            // Process command
            mailbox_handler(global_id);
        }

        // Small delay to avoid consuming all GPU resources
        // (in real implementation, would use GPU interrupts)
        for (var i = 0u; i < 100u; i = i + 1u) {
            // Busy wait
        }
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

@compute @workgroup_size(1, 1, 1)
fn mailbox_init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Initialize mailbox state
    mailbox_cmd = 0u;
    mailbox_status = STATUS_READY;
    mailbox_resp = 0u;
    mailbox_doorbell = 0u;

    // Clear UART buffer
    atomicStore(&uart_write_idx, 0u);

    // Clear performance counters
    for (var i = 0u; i < 16u; i = i + 1u) {
        perf_counters[i] = 0u;
    }
}

// =============================================================================
// DEBUG AND MONITORING
// =============================================================================

@compute @workgroup_size(1, 1, 1)
fn get_stats(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Write stats to response register
    let received = perf_counters[PERF_COMMANDS_RECEIVED];
    let completed = perf_counters[PERF_COMMANDS_COMPLETED];
    let failed = perf_counters[PERF_COMMANDS_FAILED];

    // Pack stats into response (simplified)
    mailbox_resp = received | (completed << 8u) | (failed << 16u);
}
