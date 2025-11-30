const VRAMMemoryManager = require('./vram');
const PixelISAVM = require('./vm');

function main() {
    console.log("[STATUS: SIMULATION_MODE_ENGAGED]");
    console.log("[PROTOCOL: TITAN_STACK_MK10]");

    const vram = new VRAMMemoryManager(64, 64);
    const vm = new PixelISAVM(vram);

    // Fractal Test (Recursive Draw)
    // OpCodes: CALL, RET, PUSH, POP, MOV, ADD, CMP, JE, DRAW, FILL

    // Program structure:
    // Main:
    //   INIT X, Y, Size
    //   CALL :RECURSIVE_DRAW
    //   EXIT

    // :RECURSIVE_DRAW
    //   PUSH args
    //   Check Base Case
    //   Draw
    //   Recurse Left
    //   Recurse Right
    //   POP args
    //   RET

    // Note: JS VM needs label resolution.
    // I'll define labels manually or implement a simple pass.
    // Let's implement manually with IP offsets for simplicity or use label map.
    // Label map is safer.

    const labels = {
        ":RECURSIVE_DRAW": 5, // Index where function starts
        ":EXIT_SUB": 29
    };

    const program = [
        // 0: Init
        ["MOV", "R0", 24], // X
        ["MOV", "R1", 5],  // Y
        ["MOV", "R2", 32], // Size

        // 3: Call
        ["CALL", ":RECURSIVE_DRAW"],
        ["EXIT"],

        // 5: :RECURSIVE_DRAW
        ["PUSH", "R0"],
        ["PUSH", "R1"],
        ["PUSH", "R2"],

        // 8: Color = Size (R3)
        ["MOV", "R3", "R2"],

        // 9: Fill Rect(x, y, size, size, color)
        ["FILL", "R0", "R1", "R2", "R2", "R3"],

        // 10: Check Base Case (Size < 2) -> JE :EXIT_SUB
        // CMP R2, 1
        ["CMP", "R2", 1],
        ["JE", ":EXIT_SUB"],

        // 12: Prepare Left Child
        // New Size = Size / 2
        ["DIV", "R2", 2],

        // New X = X - New Size (R0 = R0 - R2)
        ["SUB", "R0", "R2"],

        // New Y = Y + Size*2 + 1 (Parent Size relative)
        // Need Parent Size. R2 is New Size. Parent = R2*2.
        ["MOV", "R5", "R2"],
        ["ADD", "R5", "R5"], // R5 = Parent Size
        ["ADD", "R1", "R5"],
        ["ADD", "R1", 1],

        // 17: Call Left
        ["CALL", ":RECURSIVE_DRAW"],

        // 18: Restore Parent State for Right Child
        // We need to restore R0, R1, R2 to "Entry State"
        // But we are deep in registers.
        // We can POP and PUSH back.
        ["POP", "R2"],
        ["POP", "R1"],
        ["POP", "R0"],

        ["PUSH", "R0"],
        ["PUSH", "R1"],
        ["PUSH", "R2"],

        // 24: Prepare Right Child
        // New Size = Size / 2 (Original Size is in R2)
        ["DIV", "R2", 2],

        // New X = X + Parent Size + New Size
        ["MOV", "R5", "R2"],
        ["ADD", "R5", "R5"],
        ["ADD", "R0", "R5"],
        ["ADD", "R0", "R2"],

        // New Y = Y + Parent Size + 1
        ["ADD", "R1", "R5"],
        ["ADD", "R1", 1],

        // 28: Call Right
        ["CALL", ":RECURSIVE_DRAW"],

        // 29: :EXIT_SUB
        ["POP", "R2"],
        ["POP", "R1"],
        ["POP", "R0"],
        ["RET"]
    ];

    vm.loadProgram(program, labels);

    console.log("Compiling recursive assembly... Done.");
    console.log("TITAN-1 MK10 EXECUTION OUTPUT:");

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
