
// TITAN-2 Parallel Core Simulation for Node.js Backend

const VRAM_WIDTH = 32;
const VRAM_HEIGHT = 32;
const BYTES_PER_CORE = 12; // R0, R1, FLAGS, PC, N, S, E, W, NE, NW, SE, SW

class ParallelCoreVM {
  constructor() {
    this.coreState = new Uint8Array(VRAM_WIDTH * VRAM_HEIGHT * BYTES_PER_CORE);
    this.vram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
  }

  seedGlider() {
    const setAlive = (x, y) => {
        const off = ((y * VRAM_WIDTH) + x) * BYTES_PER_CORE;
        this.coreState[off + 0] = 255; // R0 = 255 (Alive)
    };
    // Glider pattern at 1,0 (top-leftish)
    setAlive(1, 0);
    setAlive(2, 1);
    setAlive(0, 2);
    setAlive(1, 2);
    setAlive(2, 2);
    console.log("[TITAN-2] Glider seeded at (1,0)");
  }

  stepPixelCore(state, x, y, program) {
    const off = (y * VRAM_WIDTH + x) * BYTES_PER_CORE;

    if (state[off + 2] & 0x80) return; // HALT check

    const pc = state[off + 3];
    if (pc >= program.length) return;

    const instr = program[pc];
    let jumped = false;

    switch (instr.op) {
        case 'PLR0':
            state[off + 0] = instr.imm;
            break;
        case 'PLR1':
            state[off + 1] = instr.imm;
            break;
        case 'COUNT_NEIGHBORS': {
            let count = 0;
            for(let i=4; i<12; i++) {
                if (state[off + i] > 0) count++;
            }
            state[off + 1] = count;
            break;
        }
        case 'CMP_IMM_R1': {
            const r1 = state[off + 1];
            const imm = instr.imm;
            if (r1 === imm) state[off + 2] |= 0x01;
            else state[off + 2] &= ~0x01;
            break;
        }
        case 'JE': {
            if (state[off + 2] & 0x01) {
                state[off + 3] = instr.target;
                jumped = true;
            }
            break;
        }
        case 'JMP':
            state[off + 3] = instr.target;
            jumped = true;
            break;
        case 'BARRIER':
            break;
        case 'HALT':
            state[off + 2] |= 0x80;
            break;
    }

    if (!jumped) {
        state[off + 3]++;
    }
  }

  runFrame(program) {
    const next = new Uint8Array(this.coreState);

    // Phase 1: Neighbor Cache
    for (let y = 0; y < VRAM_HEIGHT; y++) {
        for (let x = 0; x < VRAM_WIDTH; x++) {
            const off = (y * VRAM_WIDTH + x) * BYTES_PER_CORE;
            const n = (y - 1 + VRAM_HEIGHT) % VRAM_HEIGHT;
            const s = (y + 1) % VRAM_HEIGHT;
            const w = (x - 1 + VRAM_WIDTH) % VRAM_WIDTH;
            const e = (x + 1) % VRAM_WIDTH;
            const getN = (ax, ay) => this.coreState[(ay * VRAM_WIDTH + ax) * BYTES_PER_CORE + 0];

            next[off + 4] = getN(x, n);
            next[off + 5] = getN(x, s);
            next[off + 6] = getN(e, y);
            next[off + 7] = getN(w, y);
            next[off + 8] = getN(e, n);
            next[off + 9] = getN(w, n);
            next[off + 10] = getN(e, s);
            next[off + 11] = getN(w, s);
        }
    }

    // Phase 2: Execute
    let neighborOps = 0;
    for (let y = 0; y < VRAM_HEIGHT; y++) {
        for (let x = 0; x < VRAM_WIDTH; x++) {
            const off = (y * VRAM_WIDTH + x) * BYTES_PER_CORE;
            let instrCount = 0;
            while (instrCount < 20) {
                const pc = next[off + 3];
                if (pc >= program.length) break;
                const op = program[pc].op;
                if (op === 'BARRIER') {
                    next[off + 3]++;
                    break;
                }
                if (op === 'COUNT_NEIGHBORS') neighborOps += 8;
                this.stepPixelCore(next, x, y, program);
                instrCount++;
            }
        }
    }

    this.coreState = next;

    // Phase 3: Render
    let aliveCount = 0;
    for (let i = 0; i < VRAM_WIDTH * VRAM_HEIGHT; i++) {
        const r0 = next[i * BYTES_PER_CORE + 0];
        this.vram[i] = r0 === 255 ? 0xFFFFFFFF : 0xFF000000;
        if (r0 === 255) aliveCount++;
    }

    console.log(`[TITAN-2] Frame Complete. Active Cores: 1024. Alive Cells: ${aliveCount}. Neighbor Ops: ${neighborOps}`);
  }
}

// Game of Life Program
const LIFE_PROGRAM = [
    { op: 'COUNT_NEIGHBORS' },
    { op: 'CMP_IMM_R1', imm: 3 },
    { op: 'JE', target: 7 },
    { op: 'CMP_IMM_R1', imm: 2 },
    { op: 'JE', target: 9 },
    { op: 'PLR0', imm: 0 },
    { op: 'JMP', target: 9 },
    { op: 'PLR0', imm: 255 },
    { op: 'JMP', target: 9 },
    { op: 'BARRIER' },
    { op: 'JMP', target: 0 }
];

// Run Demo
const vm = new ParallelCoreVM();
vm.seedGlider();
console.log("Starting simulation...");
for(let i=0; i<5; i++) {
    vm.runFrame(LIFE_PROGRAM);
}
