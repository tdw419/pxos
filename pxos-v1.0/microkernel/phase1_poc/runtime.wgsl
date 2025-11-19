// pxOS GPU Runtime - WebGPU Compute Shader v0.2
// Executes os.pxi and sends privileged requests to the CPU via mailbox.

// Bindings
@group(0) @binding(0) var os_code: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> cpu_request_mailbox: array<atomic<u32>>;

// Opcodes (must match create_os_pxi.py and microkernel.asm)
const OP_PRINT_CHAR: u32 = 0x01u;
const OP_HALT: u32 = 0xFFu;

// Mailbox Request Opcodes
const REQ_MMIO_WRITE_UART: u32 = 0x80u;
const REQ_CPU_HALT: u32 = 0xFFu;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var pc: u32 = 0u;

    loop {
        let texture_size = textureDimensions(os_code);
        if (pc >= texture_size.x) {
            break;
        }

        // Fetch and decode instruction from os.pxi
        let pixel = textureLoad(os_code, vec2<i32>(i32(pc), 0), 0);
        let opcode = u32(pixel.r * 255.0);
        let arg1 = u32(pixel.g * 255.0);

        pc = pc + 1u;

        // Execute instruction
        switch (opcode) {
            case OP_PRINT_CHAR: {
                // Request CPU to print a character
                let request = (REQ_MMIO_WRITE_UART << 24) | arg1;
                atomicStore(&cpu_request_mailbox[id.x], request);
                break;
            }
            case OP_HALT: {
                // Request CPU to halt
                let request = (REQ_CPU_HALT << 24);
                atomicStore(&cpu_request_mailbox[id.x], request);
                break;
            }
            default: {
                // NOP
            }
        }

        if (opcode == OP_HALT) {
            break;
        }
    }
}
