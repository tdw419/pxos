const VRAMMemoryManager = require('./vram');
const HeapAllocator = require('./heap');

class PixelISAVM {
    constructor(vram) {
        this.vram = vram;
        this.heap = new HeapAllocator(vram);
        this.registers = {};
        for (let i = 0; i < 16; i++) this.registers[`R${i}`] = 0;
        this.registers["PTR"] = 0;
        this.stack = [];
        this.ip = 0;
        this.program = [];
        this.labels = {};
        this.running = false;
        this.callStack = [];
        this.flags = { EQ: false, LT: false, GT: false };
    }

    loadProgram(program, labels) {
        this.program = program;
        this.labels = labels;
        this.ip = 0;
        this.running = true;
    }

    run(steps = null) {
        let count = 0;
        while (this.running && this.ip < this.program.length) {
            if (steps && count >= steps) break;

            const instruction = this.program[this.ip];
            this.execute(instruction);
            count++;

            if (!this.running) break;
        }
    }

    execute(instruction) {
        const op = instruction[0];
        const args = instruction.slice(1);

        switch (op) {
            case "MOV": {
                const [dest, src] = args;
                const val = this.getValue(src);
                this.setValue(dest, val);
                this.ip++;
                break;
            }
            case "ADD": {
                const [dest, src] = args;
                const val = this.getValue(src);
                const current = this.getValue(dest);
                this.setValue(dest, current + val);
                this.ip++;
                break;
            }
            case "SUB": {
                const [dest, src] = args;
                const val = this.getValue(src);
                const current = this.getValue(dest);
                this.setValue(dest, current - val);
                this.ip++;
                break;
            }
            case "DIV": {
                const [dest, src] = args;
                const val = this.getValue(src);
                const current = this.getValue(dest);
                if (val === 0) throw new Error("Division by zero");
                this.setValue(dest, Math.floor(current / val));
                this.ip++;
                break;
            }
            case "PUSH": {
                const val = this.getValue(args[0]);
                this.stack.push(val);
                this.ip++;
                break;
            }
            case "POP": {
                if (this.stack.length === 0) throw new Error("Stack Underflow");
                const val = this.stack.pop();
                this.setValue(args[0], val);
                this.ip++;
                break;
            }
            case "CALL": {
                const target = args[0];
                if (this.labels[target] !== undefined) {
                    this.callStack.push(this.ip + 1);
                    this.ip = this.labels[target];
                } else {
                    throw new Error(`Unknown label: ${target}`);
                }
                break;
            }
            case "RET": {
                if (this.callStack.length === 0) {
                    this.running = false;
                } else {
                    this.ip = this.callStack.pop();
                }
                break;
            }
            case "CMP": {
                const op1 = this.getValue(args[0]);
                const op2 = this.getValue(args[1]);
                this.flags = { EQ: op1 === op2, LT: op1 < op2, GT: op1 > op2 };
                this.ip++;
                break;
            }
            case "JE": {
                if (this.flags.EQ) {
                    const target = args[0];
                    if (this.labels[target] !== undefined) {
                        this.ip = this.labels[target];
                    } else {
                        throw new Error(`Unknown label: ${target}`);
                    }
                } else {
                    this.ip++;
                }
                break;
            }
            case "JNE": {
                if (!this.flags.EQ) {
                    const target = args[0];
                    if (this.labels[target] !== undefined) {
                        this.ip = this.labels[target];
                    } else {
                        throw new Error(`Unknown label: ${target}`);
                    }
                } else {
                    this.ip++;
                }
                break;
            }
            case "JMP": {
                const target = args[0];
                if (this.labels[target] !== undefined) {
                    this.ip = this.labels[target];
                } else {
                    throw new Error(`Unknown label: ${target}`);
                }
                break;
            }
            case "DRAW": {
                const x = this.getValue(args[0]);
                const y = this.getValue(args[1]);
                const c = this.getValue(args[2]);
                this.vram.drawPixel(x, y, c);
                this.ip++;
                break;
            }
            case "FILL": {
                if (args.length === 5) {
                    const x = this.getValue(args[0]);
                    const y = this.getValue(args[1]);
                    const w = this.getValue(args[2]);
                    const h = this.getValue(args[3]);
                    const c = this.getValue(args[4]);
                    this.vram.fillRect(x, y, w, h, c);
                }
                this.ip++;
                break;
            }
            // NEW HEAP OPCODES
            case "ALLOC": {
                const dest = args[0];
                const size = this.getValue(args[1]);
                const addr = this.heap.alloc(size);
                this.setValue(dest, addr);
                console.log(`ALLOC: ${dest} assigned address ${addr} (Size: ${size})`);
                this.ip++;
                break;
            }
            case "FREE": {
                const addr = this.getValue(args[0]);
                this.heap.free(addr);
                this.ip++;
                break;
            }
            case "PTR_STORE": {
                // PTR_STORE Base, Offset, Value
                const base = this.getValue(args[0]);
                const offset = this.getValue(args[1]);
                const val = this.getValue(args[2]);

                const addr = base + offset;
                this.heap.writeByte(addr, val);

                const src = this.heap.getXY(addr);
                const target = this.heap.getXY(val);

                console.log(`PTR_STORE: Linking (${src.x}, ${src.y}) -> (${target.x}, ${target.y})`);
                this.vram.drawLine(src.x, src.y, target.x, target.y, 1);

                this.ip++;
                break;
            }
            case "PTR_LOAD": {
                // PTR_LOAD Dest, Base, Offset
                const dest = args[0];
                const base = this.getValue(args[1]);
                const offset = this.getValue(args[2]);

                const addr = base + offset;
                const val = this.heap.readByte(addr);
                this.setValue(dest, val);

                this.ip++;
                break;
            }
            case "EXIT": {
                this.running = false;
                break;
            }
            default: {
                console.log(`Unknown Opcode: ${op}`);
                this.ip++;
            }
        }
    }

    getValue(arg) {
        if (typeof arg === 'number') return arg;
        if (typeof arg === 'string') {
            if (this.registers.hasOwnProperty(arg)) {
                return this.registers[arg];
            }
            const num = parseInt(arg);
            if (!isNaN(num)) return num;
        }
        throw new Error(`Invalid operand: ${arg}`);
    }

    setValue(arg, value) {
        if (this.registers.hasOwnProperty(arg)) {
            this.registers[arg] = value;
        } else {
            throw new Error(`Invalid destination: ${arg}`);
        }
    }
}

module.exports = PixelISAVM;
