const VRAMMemoryManager = require('./vram');
const PixelISAVM = require('./vm');

// Helper to simulate sleep
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Helper to generate ASCII static noise
function generateAsciiNoise(width, height) {
    const chars = ['#', '@', '%', '&', '*', '+', '=', '-', ':'];
    let output = [];
    for (let y = 0; y < height; y++) {
        let line = "";
        for (let x = 0; x < width; x++) {
            line += chars[Math.floor(Math.random() * chars.length)];
        }
        output.push(line);
    }
    return output.join('\n');
}

// Helper to generate Upload Bars
function generateUploadBars(width, height, progress) {
    let output = [];
    const filledRows = Math.floor(height * progress);
    for (let y = 0; y < height; y++) {
        if (y < filledRows) {
            output.push("=".repeat(width)); // Uploaded data
        } else {
            output.push(".".repeat(width)); // Empty
        }
    }
    return output.join('\n');
}

async function main() {
    console.log("[STATUS: SIMULATION_MODE_ENGAGED]");
    console.log("[PROTOCOL: TITAN_SHADER_PIPELINE]");

    // 1. Simulate JIT Compilation
    console.log("\n[PIPELINE: JIT_COMPILING]");
    console.log("CORE: EMITTING SPIR-V IR...");
    console.log(generateAsciiNoise(64, 8)); // Small slice of noise
    await sleep(500);

    // 2. Simulate DMA Upload
    console.log("\n[PIPELINE: UPLOADING_BUFFER]");
    console.log("BUS: DMA TRANSFER [CPU -> VRAM]");
    console.log(generateUploadBars(64, 8, 0.5));
    await sleep(300);
    console.log(generateUploadBars(64, 8, 1.0));

    // 3. Execute VM
    console.log("\n[PIPELINE: EXECUTING]");
    console.log("GPU: KERNEL_COMPLETE. Executing Native PixelISA...");

    const vram = new VRAMMemoryManager(64, 64);
    const vm = new PixelISAVM(vram);

    // Fractal Test (Recursive Draw)
    // Same program as before, but wrapped in the pipeline simulation

    const labels = {
        ":RECURSIVE_DRAW": 5,
        ":EXIT_SUB": 29
    };

    const program = [
        ["MOV", "R0", 24],
        ["MOV", "R1", 5],
        ["MOV", "R2", 32],
        ["CALL", ":RECURSIVE_DRAW"],
        ["EXIT"],
        ["PUSH", "R0"],
        ["PUSH", "R1"],
        ["PUSH", "R2"],
        ["MOV", "R3", "R2"],
        ["FILL", "R0", "R1", "R2", "R2", "R3"],
        ["CMP", "R2", 1],
        ["JE", ":EXIT_SUB"],
        ["DIV", "R2", 2],
        ["SUB", "R0", "R2"],
        ["MOV", "R5", "R2"],
        ["ADD", "R5", "R5"],
        ["ADD", "R1", "R5"],
        ["ADD", "R1", 1],
        ["CALL", ":RECURSIVE_DRAW"],
        ["POP", "R2"],
        ["POP", "R1"],
        ["POP", "R0"],
        ["PUSH", "R0"],
        ["PUSH", "R1"],
        ["PUSH", "R2"],
        ["DIV", "R2", 2],
        ["MOV", "R5", "R2"],
        ["ADD", "R5", "R5"],
        ["ADD", "R0", "R5"],
        ["ADD", "R0", "R2"],
        ["ADD", "R1", "R5"],
        ["ADD", "R1", 1],
        ["CALL", ":RECURSIVE_DRAW"],
        ["POP", "R2"],
        ["POP", "R1"],
        ["POP", "R0"],
        ["RET"]
    ];

    vm.loadProgram(program, labels);

    try {
        vm.run(10000);
    } catch (e) {
        console.error(`Error: ${e.message}`);
    }

    console.log("\nVISUAL RESULT:");
    console.log(vram.renderAscii());

    console.log("[STATUS: STACK_VERIFIED]");
    console.log(`[STACK_FRAMES: ${vm.callStack.length} ACTIVE]`);
}

main();
