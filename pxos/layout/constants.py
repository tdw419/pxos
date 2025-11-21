"""
pxos/layout/constants.py

Layout constants for VRAM OS regions.

These define where different OS components live in the VRAM texture.
"""

# Default VRAM dimensions
DEFAULT_VRAM_WIDTH = 512
DEFAULT_VRAM_HEIGHT = 512

# Region definitions (x, y, width, height)
REGION_METADATA = {
    "x": 0,
    "y": 0,
    "w": 512,
    "h": 16,
    "color": (64, 128, 192, 255),  # Steel blue band
    "description": "OS metadata (version, boot flags, syscall table pointer, etc.)"
}

REGION_OPCODE_PALETTE = {
    "x": 0,
    "y": 16,
    "w": 512,
    "h": 16,
    "description": "Opcode palette - each color maps to an instruction"
}

REGION_KERNEL = {
    "x": 0,
    "y": 32,
    "w": 512,
    "h": 96,
    "color": (40, 40, 60, 255),  # Dark blue-grey
    "description": "Kernel code and core OS routines"
}

REGION_SYSCALL_TABLE = {
    "x": 0,
    "y": 128,
    "w": 512,
    "h": 32,
    "color": (60, 40, 80, 255),  # Dark purple
    "description": "System call jump table"
}

REGION_PROCESS_TABLE = {
    "x": 0,
    "y": 160,
    "w": 512,
    "h": 64,
    "color": (80, 60, 40, 255),  # Dark brown
    "description": "Process control blocks (PCBs)"
}

REGION_PROGRAM_AREA = {
    "x": 0,
    "y": 224,
    "w": 512,
    "h": 288,
    "color": (20, 20, 20, 255),  # Almost black (program background)
    "description": "User program space"
}

# Sample opcode palette (will be populated by roadmap steps)
OPCODE_PALETTE = {
    "NOP": (0, 0, 0, 255),         # Black
    "HALT": (255, 0, 0, 255),      # Red
    "LOAD": (0, 255, 0, 255),      # Green
    "STORE": (0, 0, 255, 255),     # Blue
    "ADD": (255, 255, 0, 255),     # Yellow
    "SUB": (255, 0, 255, 255),     # Magenta
    "JMP": (0, 255, 255, 255),     # Cyan
    "JZ": (128, 128, 128, 255),    # Grey
}
