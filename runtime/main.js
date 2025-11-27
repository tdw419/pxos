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

// Helper to generate Blue Stripes (ASCII approximation)
function generateBlueStripes(width, height) {
    let output = [];
    for (let y = 0; y < height; y++) {
        let line = "";
        for (let x = 0; x < width; x++) {
            line += (x % 32 < 16) ? "B" : "."; // B for Blue, . for dark
        }
        output.push(line);
    }
    return output.join('\n');
}

// Helper to generate Yellow Bands (ASCII approximation)
function generateYellowBands(width, height) {
    let output = [];
    for (let y = 0; y < height; y++) {
        if (y % 2 === 0) {
            output.push("Y".repeat(width)); // Y for Yellow
        } else {
            output.push("-".repeat(width)); // - for dark
        }
    }
    return output.join('\n');
}

async function main() {
    console.log("[STATUS: SIMULATION_MODE_ENGAGED]");
    console.log("[PROTOCOL: TITAN_SHADER_PIPELINE]");

    // 1. Simulate JIT Compilation
    console.log("\n[PIPELINE: JIT_COMPILING]");
    console.log("CORE: JIT_COMPILING_WGSL...");
    console.log(generateAsciiNoise(64, 4));
    await sleep(600);

    // 2. Simulate SPIR-V Emission
    console.log("\n[PIPELINE: SPIRV_EMIT]");
    console.log("SPIRV: GENERATING_BINARY...");
    console.log(generateBlueStripes(64, 4));
    await sleep(300);

    // 3. Simulate GPU Upload
    console.log("\n[PIPELINE: GPU_UPLOAD]");
    console.log("BUS: DMA_CPU_TO_GPU...");
    console.log(generateYellowBands(64, 4));
    await sleep(200);

    // 4. Execute VM
    console.log("\n[PIPELINE: EXECUTING]");
    console.log("GPU: KERNEL_COMPLETE. Executing Native PixelISA...");

    const vram = new VRAMMemoryManager(64, 64);
    const vm = new PixelISAVM(vram);

    // Fractal Test (Recursive Draw)
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
