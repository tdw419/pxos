"""
pxOS Python API (Cross-Compiler Stubs)

This module provides a Python API for writing pxOS programs. These functions
do NOT run on pxOS itself - they are compiled into low-level primitives by
the pxpyc.py compiler.

The Pixel LLM can write natural Python code using this API, which then gets
translated into WRITE/DEFINE commands that pxOS can execute.

Usage:
    from pxos import clear_screen, print_text, print_char, move_cursor

    def main():
        clear_screen()
        print_text("Hello pxOS!", row=10, col=5)

    if __name__ == "__main__":
        main()
"""

from typing import Optional

# ============================================================================
# Display Functions
# ============================================================================

def clear_screen(color: int = 0x07) -> None:
    """
    Clear the screen with the specified color attribute.

    Args:
        color: VGA text attribute byte (default 0x07 = light gray on black)
               Format: BBBBFFFF (B=background, F=foreground)

    Compiles to:
        - Set ES to 0xB800 (VGA text buffer)
        - Fill 2000 words with 0x0720 (space + attribute)

    Example:
        clear_screen()           # Default colors
        clear_screen(0x1F)       # White on blue
    """
    pass


def print_char(char: str, row: Optional[int] = None, col: Optional[int] = None,
               attr: int = 0x07) -> None:
    """
    Print a single character at the specified position.

    Args:
        char: Single character to print
        row: Row position (0-24), None = current cursor
        col: Column position (0-79), None = current cursor
        attr: Color attribute (default 0x07)

    Compiles to:
        - BIOS INT 0x10, AH=0x0E (teletype) if position is None
        - Direct VGA memory write if position is specified

    Example:
        print_char('A')              # At cursor
        print_char('X', 10, 20)      # At row 10, col 20
        print_char('!', attr=0x4F)   # Red background
    """
    pass


def print_text(text: str, row: Optional[int] = None, col: Optional[int] = None,
               attr: int = 0x07) -> None:
    """
    Print a string at the specified position.

    Args:
        text: String to print
        row: Starting row (0-24), None = current cursor
        col: Starting column (0-79), None = current cursor
        attr: Color attribute

    Compiles to:
        - Series of print_char calls or optimized string routine

    Example:
        print_text("Hello World!")
        print_text("Status: OK", row=0, col=60, attr=0x2F)
    """
    pass


def move_cursor(row: int, col: int) -> None:
    """
    Move the cursor to the specified position.

    Args:
        row: Row position (0-24)
        col: Column position (0-79)

    Compiles to:
        - BIOS INT 0x10, AH=0x02 (set cursor position)

    Example:
        move_cursor(12, 40)  # Center of screen
    """
    pass


# ============================================================================
# Keyboard Functions
# ============================================================================

def read_key() -> str:
    """
    Wait for and read a single keypress.

    Returns:
        Single character pressed

    Compiles to:
        - BIOS INT 0x16, AH=0x00 (read key)
        - Returns AL (ASCII code)

    Example:
        key = read_key()
        if key == '\r':
            print_text("Enter pressed!")
    """
    pass


def check_key() -> Optional[str]:
    """
    Check if a key is available without blocking.

    Returns:
        Character if key is available, None otherwise

    Compiles to:
        - BIOS INT 0x16, AH=0x01 (check keyboard status)

    Example:
        key = check_key()
        if key:
            print_char(key)
    """
    pass


# ============================================================================
# String Functions
# ============================================================================

def write_string(text: str, null_terminate: bool = True) -> int:
    """
    Write a null-terminated string to a memory location.

    This is a compile-time function that emits data bytes.

    Args:
        text: String to write
        null_terminate: Add null terminator (default True)

    Returns:
        Memory address where string was written (assigned by compiler)

    Compiles to:
        - Series of WRITE commands with ASCII values
        - DEFINE command for the string label

    Example:
        msg_addr = write_string("Hello!")
        # Compiler generates:
        # DEFINE msg_0 0x7E00
        # WRITE 0x7E00 0x48  # 'H'
        # WRITE 0x7E01 0x65  # 'e'
        # ...
    """
    pass


# ============================================================================
# Memory Functions
# ============================================================================

def peek(addr: int) -> int:
    """
    Read a byte from memory.

    Args:
        addr: Memory address to read

    Returns:
        Byte value (0x00-0xFF)

    Compiles to:
        - MOV instruction to read from address

    Example:
        value = peek(0xB8000)  # Read from VGA memory
    """
    pass


def poke(addr: int, value: int) -> None:
    """
    Write a byte to memory.

    Args:
        addr: Memory address
        value: Byte value (0x00-0xFF)

    Compiles to:
        - MOV instruction to write to address

    Example:
        poke(0xB8000, 0x41)  # Write 'A' to VGA memory
    """
    pass


# ============================================================================
# Control Flow
# ============================================================================

def loop_forever() -> None:
    """
    Infinite loop (halt execution gracefully).

    Compiles to:
        - JMP $ (jump to self)

    Example:
        print_text("System halted")
        loop_forever()
    """
    pass


def delay(ms: int) -> None:
    """
    Delay for approximately the specified milliseconds.

    Args:
        ms: Milliseconds to delay

    Compiles to:
        - Busy loop calibrated for typical CPU

    Note: Timing is approximate and CPU-dependent

    Example:
        print_char('.')
        delay(500)  # Half second pause
    """
    pass


# ============================================================================
# Constants
# ============================================================================

# VGA Colors (4-bit foreground)
BLACK = 0x0
BLUE = 0x1
GREEN = 0x2
CYAN = 0x3
RED = 0x4
MAGENTA = 0x5
BROWN = 0x6
LIGHT_GRAY = 0x7
DARK_GRAY = 0x8
LIGHT_BLUE = 0x9
LIGHT_GREEN = 0xA
LIGHT_CYAN = 0xB
LIGHT_RED = 0xC
LIGHT_MAGENTA = 0xD
YELLOW = 0xE
WHITE = 0xF

# Helper function to create color attributes
def make_color(fg: int, bg: int = 0) -> int:
    """
    Create a VGA color attribute byte.

    Args:
        fg: Foreground color (0-15)
        bg: Background color (0-7)

    Returns:
        Color attribute byte

    Example:
        attr = make_color(WHITE, BLUE)  # White text on blue background
        print_text("Title", attr=attr)
    """
    return (bg << 4) | (fg & 0x0F)


# Memory regions
VGA_TEXT_BUFFER = 0xB8000
VGA_WIDTH = 80
VGA_HEIGHT = 25
VGA_SIZE = VGA_WIDTH * VGA_HEIGHT

# Special characters
CHAR_ENTER = '\r'
CHAR_NEWLINE = '\n'
CHAR_BACKSPACE = '\b'
CHAR_TAB = '\t'
CHAR_BELL = '\a'
