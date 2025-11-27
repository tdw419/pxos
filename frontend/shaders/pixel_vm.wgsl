// frontend/shaders/pixel_vm.wgsl

// The PixelVM Firmware
// Reads commands from Row 0, State from Row 1, Data/Frame from Rows 2-31.

@group(0) @binding(0) var<storage, read_write> vram: array<u32>;

const WIDTH: u32 = 32u;
const COMMAND_ROW: u32 = 0u;
const REG_ROW: u32 = 1u;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // One thread per command slot (32 parallel commands max)
    let cmd_idx = id.x;
    if (cmd_idx >= WIDTH) { return; }

    // Fetch Command Pixel
    // Format: 0xAABBGGRR (Little Endian)
    // R = Opcode
    // G = Op1
    // B = Op2
    // A = Op3 / Imm

    let raw_cmd = vram[cmd_idx];

    let opcode = raw_cmd & 0xFFu;
    let op1    = (raw_cmd >> 8u) & 0xFFu;
    let op2    = (raw_cmd >> 16u) & 0xFFu;
    let op3    = (raw_cmd >> 24u) & 0xFFu;

    if (opcode == 0u) { return; } // NOP

    // Helper to get/set registers
    // Registers are at index 32 to 63

    // 0x01: SET_REG_IMM (Dest=Op1, Val=Op2|Op3)
    if (opcode == 0x01u) {
        let val = (op3 << 8u) | op2;
        if (op1 < 32u) {
            vram[WIDTH + op1] = val;
        }
    }
    // 0x02: ADD (Dest=Op1, Src1=Op2, Src2=Op3)
    else if (opcode == 0x02u) {
        if (op1 < 32u && op2 < 32u && op3 < 32u) {
            let val1 = vram[WIDTH + op2];
            let val2 = vram[WIDTH + op3];
            vram[WIDTH + op1] = val1 + val2;
        }
    }
    // 0x04: STORE_MEM (AddrReg=Op1, DataReg=Op2)
    else if (opcode == 0x04u) {
        if (op1 < 32u && op2 < 32u) {
            let addr = vram[WIDTH + op1];
            let data = vram[WIDTH + op2];
            // Bound check: only write to Data region (>= 64)
            if (addr >= 64u && addr < 1024u) {
                vram[addr] = data;
            }
        }
    }
    // 0x06: DRAW (X=Op1, Y=Op2, Color=Op3[Reg])
    else if (opcode == 0x06u) {
        // This instruction uses Op1/Op2 as immediate small coords for simplicity in this demo
        // Or registers? Let's use registers for flexibility.
        if (op1 < 32u && op2 < 32u && op3 < 32u) {
            let x = vram[WIDTH + op1];
            let y = vram[WIDTH + op2];
            let color = vram[WIDTH + op3];
            let addr = y * WIDTH + x;
            if (addr >= 64u && addr < 1024u) {
                vram[addr] = color;
            }
        }
    }

    // Clear command after execution (Consumption)
    // This turns the command ring into a stream consumer
    // vram[cmd_idx] = 0u;
    // Keep it for debug visibility for now, or clear it?
    // Clearing it allows the "Driver" to know it's done if it reads back.
    vram[cmd_idx] = 0u;
}
