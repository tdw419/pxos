// Define the structure for the VM's state.
// This will be stored in a buffer.
struct State {
  ip : u32,        // Instruction pointer (x-coordinate in the code texture)
  zero_flag: u32, // 1 if last CMP result was zero, 0 otherwise
  halted : u32,    // 1 if the VM is halted, 0 otherwise
  regs : array<u32, 16>, // 16 general-purpose 32-bit registers
};

// Define the bind group layout.
// group(0) is the standard for this simple setup.
@group(0) @binding(0) var code_tex : texture_2d<u32>;        // Program instructions (read-only)
@group(0) @binding(1) var<storage, read_write> data : array<u32>; // General purpose data/RAM
@group(0) @binding(2) var<storage, read_write> state_buf : array<State, 1>; // VM state (registers, ip, etc.)

// Helper function to unpack a u32 into four u8 values (RGBA).
// The texture loads a single u32, and we decode the instruction from its bytes.
fn unpack_rgba(pixel: u32) -> vec4<u32> {
  let r = (pixel >> 0u)  & 0xFFu;
  let g = (pixel >> 8u)  & 0xFFu;
  let b = (pixel >> 16u) & 0xFFu;
  let a = (pixel >> 24u) & 0xFFu;
  return vec4<u32>(r, g, b, a);
}

// The main compute shader entry point.
// We're using a workgroup size of 1, meaning a single thread will execute this.
// This simplifies the logic to a single VM instance.
@compute @workgroup_size(1)
fn step_pixel_vm() {
  // Get a mutable copy of the current state.
  var st = state_buf[0];

  // If the VM is already halted, do nothing.
  if (st.halted == 1u) {
    return;
  }

  // Fetch the instruction from the code texture.
  // We assume the program is a single horizontal line of pixels at y=0.
  let ip_x = i32(st.ip);
  let ip_y = 0;
  let instr_raw = textureLoad(code_tex, vec2<i32>(ip_x, ip_y), 0).r;

  // Unpack the raw u32 pixel data into RGBA components.
  let rgba = unpack_rgba(instr_raw);
  let opcode = rgba.x; // R channel
  let arg0   = rgba.y; // G channel
  let arg1   = rgba.z; // B channel
  let arg2   = rgba.w; // A channel (unused in most v0.1 opcodes)

  var next_ip = st.ip + 1u;

  // Decode and execute the opcode.
  switch (opcode) {
    case 1u: { // LOAD: st.regs[arg1] = data[arg0]
      st.regs[arg1] = data[arg0];
    }
    case 2u: { // STORE: data[arg1] = st.regs[arg0]
      data[arg1] = st.regs[arg0];
    }
    case 3u: { // ADD: st.regs[arg1] = st.regs[arg1] + st.regs[arg0]
      st.regs[arg1] = st.regs[arg1] + st.regs[arg0];
    }
    case 4u: { // JUMP: ip += arg0 (signed offset)
      var offset: i32;
      if (arg0 >= 128u) {
        offset = i32(arg0) - 256;
      } else {
        offset = i32(arg0);
      }
      next_ip = u32(i32(st.ip) + offset);
    }
    case 5u: { // CMP: sets zero_flag if st.regs[arg0] == st.regs[arg1]
      if (st.regs[arg0] == st.regs[arg1]) {
        st.zero_flag = 1u;
      } else {
        st.zero_flag = 0u;
      }
    }
    case 6u: { // JNE: Jumps if zero_flag is not set
      if (st.zero_flag == 0u) {
        var offset: i32;
        if (arg0 >= 128u) {
          offset = i32(arg0) - 256;
        } else {
          offset = i32(arg0);
        }
        next_ip = u32(i32(st.ip) + offset);
      }
    }
    case 255u: { // HALT
      st.halted = 1u;
    }
    default: {
      // Unknown opcode, halt for safety.
      st.halted = 1u;
    }
  }

  // Update the instruction pointer for the next cycle, unless halted.
  if (st.halted == 0u) {
    st.ip = next_ip;
  }

  // Write the modified state back to the buffer.
  state_buf[0] = st;
}
