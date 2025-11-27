import { Opcode, MEMORY_MAP } from './pixel_isa';

/**
 * Helper to assemble human-readable instructions into 32-bit pixel values.
 * Format: 0xAABBGGRR (Little Endian) -> R=Opcode
 */
export class PixelAssembler {
    static assemble(lines: string[]): Uint32Array {
        const buffer = new Uint32Array(32); // Max 32 commands in Row 0

        lines.forEach((line, idx) => {
            if (idx >= 32) return;
            const parts = line.trim().split(/\s+/);
            const opName = parts[0].toUpperCase();

            let op = 0;
            let a = 0, b = 0, c = 0; // Bytes for G, B, A channels

            if (opName === 'SET') {
                op = Opcode.SET;
                const reg = parseInt(parts[1].replace('R', ''));
                const val = parseInt(parts[2]);
                a = reg;
                b = val & 0xFF;
                c = (val >> 8) & 0xFF;
            } else if (opName === 'ADD') {
                op = Opcode.ADD;
                const dest = parseInt(parts[1].replace('R', ''));
                const src1 = parseInt(parts[2].replace('R', ''));
                const src2 = parseInt(parts[3].replace('R', ''));
                a = dest; b = src1; c = src2;
            } else if (opName === 'DRAW') {
                op = Opcode.DRAW;
                const xReg = parseInt(parts[1].replace('R', ''));
                const yReg = parseInt(parts[2].replace('R', ''));
                const colReg = parseInt(parts[3].replace('R', ''));
                a = xReg; b = yReg; c = colReg;
            }

            // Pack into 0xAABBGGRR
            // R = op, G = a, B = b, A = c
            buffer[idx] = (c << 24) | (b << 16) | (a << 8) | op;
        });

        return buffer;
    }
}
