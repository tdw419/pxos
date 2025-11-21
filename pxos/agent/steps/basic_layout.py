"""
pxos/agent/steps/basic_layout.py

Basic layout steps for initializing VRAM regions.

These steps paint the foundational structure of the OS into VRAM.
"""

from typing import Dict, Any
from pxos.vram_sim import SimulatedVRAM
from pxos.layout import constants as L


def step_init_background(vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Clear VRAM to a base background color."""
    # Use almost-black for program area background
    bg_color = ctx.get("background_color", (10, 10, 10, 255))
    vram.fill_rect(0, 0, vram.width, vram.height, bg_color)
    ctx["background_initialized"] = True
    return ctx


def step_layout_regions(vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Paint colored bands to mark each OS region."""

    regions = [
        ("metadata", L.REGION_METADATA),
        ("kernel", L.REGION_KERNEL),
        ("syscall_table", L.REGION_SYSCALL_TABLE),
        ("process_table", L.REGION_PROCESS_TABLE),
        ("program_area", L.REGION_PROGRAM_AREA),
    ]

    for name, region in regions:
        if "color" in region:
            vram.fill_rect(
                region["x"],
                region["y"],
                region["w"],
                region["h"],
                region["color"]
            )
            print(f"  ✓ Painted {name} region at y={region['y']}, h={region['h']}")

    ctx["regions_painted"] = True
    return ctx


def step_write_boot_banner(vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a boot status indicator in the metadata band.

    For now, just a colored stripe to signal "boot OK".
    Later this will be actual boot flags.
    """
    # Draw a bright green stripe at top-left to signal boot ready
    banner_color = ctx.get("boot_banner_color", (0, 255, 0, 255))
    vram.fill_rect(0, 0, vram.width, 4, banner_color)

    ctx["boot_banner_written"] = True
    ctx["boot_banner_color"] = banner_color
    return ctx


def step_write_opcode_palette(vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write the opcode palette into the designated region.

    Each opcode gets a unique color. The interpreter reads this palette
    to decode instruction pixels.
    """
    palette_y = L.REGION_OPCODE_PALETTE["y"]
    palette_height = L.REGION_OPCODE_PALETTE["h"]

    # Use the middle row of the palette region
    y = palette_y + (palette_height // 2)

    # Write each opcode color
    opcodes = list(L.OPCODE_PALETTE.items())
    x = 0
    stripe_width = vram.width // len(opcodes)

    for opcode_name, opcode_color in opcodes:
        vram.fill_rect(x, palette_y, stripe_width, palette_height, opcode_color)
        x += stripe_width

    ctx["opcode_palette_written"] = True
    ctx["opcode_count"] = len(opcodes)
    return ctx


def step_write_hello_program(vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a minimal 'hello world' program into the program area.

    This is just a placeholder sequence of colored pixels that represent:
        NOP, NOP, LOAD, HALT

    Later, this will be real PXI instructions.
    """
    program_x = ctx.get("hello_program_x", 32)
    program_y = L.REGION_PROGRAM_AREA["y"] + 8

    # Simple instruction sequence using opcode colors
    instructions = [
        ("NOP", L.OPCODE_PALETTE["NOP"]),
        ("NOP", L.OPCODE_PALETTE["NOP"]),
        ("LOAD", L.OPCODE_PALETTE["LOAD"]),
        ("HALT", L.OPCODE_PALETTE["HALT"]),
    ]

    x = program_x
    for opcode_name, opcode_color in instructions:
        # Write a 4x4 block for each instruction (easier to see)
        vram.fill_rect(x, program_y, 4, 4, opcode_color)
        x += 5  # 4px instruction + 1px gap

    ctx["hello_program_written"] = True
    ctx["hello_program_location"] = (program_x, program_y)
    return ctx
