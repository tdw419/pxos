// frontend/pixel_isa.ts

/**
 * Pixel Execution ABI (PixelISA) Definition
 *
 * Memory Map (32x32 VRAM = 1024 Pixels):
 * ---------------------------------------
 * Row 0 (0-31):    COMMAND RING (Instructions)
 * Row 1 (32-63):   REGISTERS (R0-R31)
 * Rows 2-31 (64+): DATA / FRAMEBUFFER
 *
 * Pixel Encoding (32-bit RGBA/UINT):
 * ----------------------------------
 * Format: 0xAABBGGRR (Little Endian in Memory) -> R is low byte.
 *
 * Opcode Layout (in Command Pixel):
 * - Byte 0 (R): Opcode
 * - Byte 1 (G): Operand 1 (Register Index or High Byte of Imm)
 * - Byte 2 (B): Operand 2 (Register Index or Low Byte of Imm)
 * - Byte 3 (A): Immediate/Flags (or Extended Opcode)
 *
 * Opcodes:
 * 0x00: NOP
 * 0x01: SET_REG_IMM (R = Dest Reg, G = Val High, B = Val Low) -> Dest = Imm16
 * 0x02: ADD_REG_REG (R = Dest, G = Src1, B = Src2) -> Dest = Src1 + Src2
 * 0x03: SUB_REG_REG (R = Dest, G = Src1, B = Src2) -> Dest = Src1 - Src2
 * 0x04: WRITE_MEM   (R = Addr Reg, G = Data Reg)   -> Mem[Addr] = Data
 * 0x05: READ_MEM    (R = Dest Reg, G = Addr Reg)   -> Dest = Mem[Addr]
 * 0x06: DRAW_PIXEL  (R = X Reg, G = Y Reg, B = Color Reg) -> Plot pixel
 * 0xFF: HALT
 */

export const MEMORY_MAP = {
    COMMAND_START: 0,
    COMMAND_END: 31,
    REGISTER_START: 32,
    REGISTER_END: 63,
    DATA_START: 64,
    WIDTH: 32,
    HEIGHT: 32
};

export enum Opcode {
    NOP = 0x00,
    SET = 0x01,
    ADD = 0x02,
    SUB = 0x03,
    STORE = 0x04,
    LOAD = 0x05,
    DRAW = 0x06,
    HALT = 0xFF
}
