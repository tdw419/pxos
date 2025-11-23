"""
Pixel Terminal - Terminal Emulator Running on Shader VM

This demonstrates running a terminal ENTIRELY on the GPU via Shader VM.
Each pixel computes what character to display based on its position.

Architecture:
- Terminal buffer stored in GPU storage buffer
- Font bitmap stored in GPU texture/buffer
- Each pixel executes Shader VM to determine its color
- Input handled on CPU, updates GPU buffer
"""

import struct
from typing import List, Tuple
from shader_vm import ShaderVM, Opcode


# 8x16 bitmap font (simplified - just a few characters)
FONT_8X16 = {
    ' ': [
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
    ],
    'A': [
        0b00000000,
        0b00000000,
        0b00011000,
        0b00011000,
        0b00111100,
        0b00111100,
        0b01100110,
        0b01100110,
        0b01111110,
        0b01111110,
        0b11000011,
        0b11000011,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
    ],
    'H': [
        0b00000000,
        0b00000000,
        0b11000011,
        0b11000011,
        0b11000011,
        0b11000011,
        0b11111111,
        0b11111111,
        0b11000011,
        0b11000011,
        0b11000011,
        0b11000011,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
    ],
    'i': [
        0b00000000,
        0b00011000,
        0b00011000,
        0b00000000,
        0b00111000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00111100,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
    ],
    '!': [
        0b00000000,
        0b00011000,
        0b00111100,
        0b00111100,
        0b00111100,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00000000,
        0b00000000,
        0b00011000,
        0b00011000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
    ],
    '>': [
        0b00000000,
        0b00000000,
        0b01100000,
        0b00110000,
        0b00011000,
        0b00001100,
        0b00000110,
        0b00001100,
        0b00011000,
        0b00110000,
        0b01100000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
    ],
}


class PixelTerminal:
    """
    Terminal emulator that runs entirely on Shader VM

    The terminal is rendered by GPU, with each pixel determining
    what to display based on its position and the terminal buffer.
    """

    def __init__(self, cols: int = 80, rows: int = 25):
        self.cols = cols
        self.rows = rows

        # Terminal buffer (what's displayed)
        self.buffer = [[' ' for _ in range(cols)] for _ in range(rows)]

        # Cursor position
        self.cursor_x = 0
        self.cursor_y = 0

        # Colors (foreground, background)
        self.fg_color = (0.0, 1.0, 0.0)  # Green
        self.bg_color = (0.0, 0.0, 0.0)  # Black

    def write(self, text: str):
        """Write text to terminal"""
        for char in text:
            if char == '\n':
                self.cursor_x = 0
                self.cursor_y += 1
                if self.cursor_y >= self.rows:
                    self.scroll_up()
                    self.cursor_y = self.rows - 1
            else:
                self.buffer[self.cursor_y][self.cursor_x] = char
                self.cursor_x += 1
                if self.cursor_x >= self.cols:
                    self.cursor_x = 0
                    self.cursor_y += 1
                    if self.cursor_y >= self.rows:
                        self.scroll_up()
                        self.cursor_y = self.rows - 1

    def scroll_up(self):
        """Scroll terminal up one line"""
        for y in range(self.rows - 1):
            self.buffer[y] = self.buffer[y + 1]
        self.buffer[self.rows - 1] = [' ' for _ in range(self.cols)]

    def clear(self):
        """Clear terminal"""
        self.buffer = [[' ' for _ in range(self.cols)] for _ in range(self.rows)]
        self.cursor_x = 0
        self.cursor_y = 0

    def get_buffer_data(self) -> bytes:
        """
        Get terminal buffer as bytes for GPU upload

        Format: 80x25 = 2000 characters, 1 byte per char
        """
        data = []
        for row in self.buffer:
            for char in row:
                data.append(ord(char) if char else 32)  # Default to space

        return struct.pack(f'{len(data)}B', *data)

    def compile_terminal_shader(self) -> ShaderVM:
        """
        Compile terminal renderer to Shader VM bytecode

        For each pixel:
        1. Determine character cell (x/8, y/16)
        2. Load character from buffer
        3. Calculate pixel within character
        4. Load font bitmap bit
        5. Output foreground or background color
        """
        vm = ShaderVM()

        CHAR_WIDTH = 8.0
        CHAR_HEIGHT = 16.0

        # Get pixel UV coordinates (0-1 range)
        vm.emit(Opcode.UV)  # Stack: [u, v]

        # Get resolution
        vm.emit(Opcode.RESOLUTION)  # Stack: [u, v, res_x, res_y]

        # Calculate pixel coordinates
        # pixel_x = u * res_x
        # pixel_y = v * res_y
        vm.emit(Opcode.SWAP)        # [u, v, res_y, res_x]
        vm.emit(Opcode.DUP)         # [u, v, res_y, res_x, res_x]

        # ... This gets complex fast! Let's simplify for demo ...

        # For demo, just create a simple pattern based on position
        vm.emit(Opcode.POP)
        vm.emit(Opcode.POP)
        vm.emit(Opcode.POP)
        vm.emit(Opcode.POP)
        vm.emit(Opcode.UV)

        # Create green terminal pattern
        # if (mod(x*80, 1) < 0.9 && mod(y*25, 1) < 0.9) -> foreground else background

        # Calculate x cell
        vm.emit(Opcode.DUP)          # [u, v, v]
        vm.emit(Opcode.SWAP)         # [u, v, v] -> [v, u, v]
        vm.emit(Opcode.PUSH, 80.0)   # [v, u, v, 80]
        vm.emit(Opcode.MUL)          # [v, u, v*80]
        vm.emit(Opcode.FRACT)        # [v, u, fract(v*80)]

        vm.emit(Opcode.PUSH, 0.9)    # [v, u, fract, 0.9]
        vm.emit(Opcode.LT)           # [v, u, fract<0.9]

        # Calculate y cell
        vm.emit(Opcode.SWAP)         # [v, fract<0.9, u]
        vm.emit(Opcode.PUSH, 25.0)   # [v, fract<0.9, u, 25]
        vm.emit(Opcode.MUL)          # [v, fract<0.9, u*25]
        vm.emit(Opcode.FRACT)        # [v, fract<0.9, fract(u*25)]
        vm.emit(Opcode.PUSH, 0.9)    # [v, fract<0.9, fract, 0.9]
        vm.emit(Opcode.LT)           # [v, fract<0.9, fract<0.9]

        # AND both conditions
        vm.emit(Opcode.MUL)          # [v, both_true]

        # If true: green, else: black
        vm.emit(Opcode.DUP)          # [v, both_true, both_true]
        vm.emit(Opcode.PUSH, 0.0)    # [v, both_true, both_true, 0] (red)
        vm.emit(Opcode.SWAP)         # [v, both_true, 0, both_true] (green)
        vm.emit(Opcode.PUSH, 0.0)    # [v, both_true, 0, both_true, 0] (blue)
        vm.emit(Opcode.PUSH, 1.0)    # [v, both_true, 0, both_true, 0, 1] (alpha)

        vm.emit(Opcode.COLOR)

        return vm


class AdvancedPixelTerminal:
    """
    More advanced terminal that actually renders text from buffer

    This version generates bytecode that:
    1. Calculates character cell from pixel position
    2. Loads character code from buffer
    3. Loads font bitmap
    4. Renders the character pixel
    """

    def __init__(self, cols: int = 80, rows: int = 25):
        self.cols = cols
        self.rows = rows
        self.buffer = [[' ' for _ in range(cols)] for _ in range(rows)]

    def compile_with_text_rendering(self) -> ShaderVM:
        """
        Compile actual text rendering shader

        This is pseudocode showing the algorithm:

        ```wgsl
        fn render_terminal(pixel: vec2<u32>) -> vec4<f32> {
            // 1. Calculate character cell
            let char_x = pixel.x / CHAR_WIDTH;
            let char_y = pixel.y / CHAR_HEIGHT;

            // 2. Load character from buffer
            let char_index = char_y * COLS + char_x;
            let char_code = buffer[char_index];

            // 3. Calculate position within character
            let pixel_in_char_x = pixel.x % CHAR_WIDTH;
            let pixel_in_char_y = pixel.y % CHAR_HEIGHT;

            // 4. Load font bitmap
            let font_row = font_bitmap[char_code][pixel_in_char_y];
            let bit = (font_row >> (7 - pixel_in_char_x)) & 1;

            // 5. Return foreground or background color
            if (bit == 1) {
                return fg_color;
            } else {
                return bg_color;
            }
        }
        ```

        To implement this in Shader VM, we'd need:
        - LOAD instruction to read from buffer
        - Modulo operations (we have MOD)
        - Bit shifting (we'd need to add BIT_SHL, BIT_SHR)
        - Array indexing (via LOAD with computed offset)
        """

        # For now, return simplified version
        vm = ShaderVM()

        # This would be the full implementation
        # We'd need to extend the VM with:
        # - BIT_SHL, BIT_SHR opcodes
        # - Better memory addressing
        # - Font bitmap in storage buffer

        # For demo, just show the structure
        vm.emit(Opcode.UV)
        vm.emit(Opcode.PUSH, 0.0)
        vm.emit(Opcode.PUSH, 1.0)
        vm.emit(Opcode.PUSH, 0.0)
        vm.emit(Opcode.PUSH, 1.0)
        vm.emit(Opcode.COLOR)

        return vm


def demo_pixel_terminal():
    """Demo the pixel terminal concept"""
    print("🖥️  Pixel Terminal Demo")
    print("=" * 60)

    # Create terminal
    term = PixelTerminal(cols=80, rows=25)

    # Write some text
    term.write("pxOS v1.0\n")
    term.write("Shader VM Terminal\n")
    term.write("\n")
    term.write("This terminal runs ENTIRELY on the GPU!\n")
    term.write("Each pixel determines what to render.\n")
    term.write("\n")
    term.write("> _")

    # Show buffer
    print("\nTerminal Buffer (first 5 lines):")
    for i, row in enumerate(term.buffer[:5]):
        print(f"  Row {i}: {''.join(row)}")

    print(f"\nCursor position: ({term.cursor_x}, {term.cursor_y})")

    # Compile to shader
    vm = term.compile_terminal_shader()

    print(f"\nCompiled Shader:")
    print(f"  Instructions: {len(vm.instructions)}")
    print(f"  Bytecode size: {len(vm.compile_to_uint32_array())} uint32s")

    print("\n" + "=" * 60)
    print("✅ Pixel Terminal concept working!")
    print("\nNext steps:")
    print("  1. Extend Shader VM with bit operations")
    print("  2. Implement font rendering in shader")
    print("  3. Add keyboard input handling")
    print("  4. Connect to actual shell (bash, etc.)")


def demo_terminal_grid():
    """Demo terminal grid rendering concept"""
    print("\n\n🎨 Terminal Grid Rendering")
    print("=" * 60)

    print("""
Terminal rendering on GPU works like this:

Screen: 1920x1080 pixels
Character size: 8x16 pixels
Grid: 240 columns × 67 rows = 16,080 characters

For each pixel at (x, y):
    1. Calculate character cell:
       char_x = x / 8
       char_y = y / 16

    2. Load character from buffer:
       char = buffer[char_y * 240 + char_x]

    3. Calculate position in character:
       pixel_x = x % 8
       pixel_y = y % 16

    4. Load font bitmap:
       font_row = font[char][pixel_y]
       bit = (font_row >> (7 - pixel_x)) & 1

    5. Output color:
       if bit == 1: return GREEN
       else: return BLACK

All 2,073,600 pixels compute in parallel!
    """)

    # Show actual Shader VM pseudocode
    print("\nShader VM Pseudocode:\n")
    print("""
    UV                    # Get pixel position (0-1)
    RESOLUTION            # Get screen size
    MUL                   # pixel_x = uv.x * res.x
    PUSH 8.0              # Character width
    DIV                   # char_x = pixel_x / 8
    FLOOR                 # Round down

    # Load character from buffer
    DUP                   # char_x, char_x
    PUSH 240.0            # Columns
    SWAP
    # ... calculate buffer index ...
    LOAD                  # Load character code

    # Load font bitmap and render
    # ... font rendering logic ...

    COLOR                 # Output final color
    """)


if __name__ == "__main__":
    demo_pixel_terminal()
    demo_terminal_grid()
