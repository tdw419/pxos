const VRAMMemoryManager = require('./vram');
const PixelISAVM = require('./vm');

function main() {
    console.log("[STATUS: EXECUTING_VISUAL_POINTER_TEST]");
    console.log("[PROTOCOL: TITAN_POINTER_MK1 - ENGAGED]");
    console.log("[HEAP_ALLOCATOR: ONLINE]");
    console.log("[POINTER_GRAPHICS: ACTIVE]");

    const vram = new VRAMMemoryManager(32, 32);
    const vm = new PixelISAVM(vram);

    // Program: Visual Linked List
    const program = [
        // Alloc Node 1 (Value 0xAA)
        ["ALLOC", "R1", 2],         // R1 = Addr Node 1
        ["MOV", "R0", 0xAA],
        ["PTR_STORE", "R1", 0, "R0"], // Node1[0] = Value

        // Alloc Node 2 (Value 0xBB)
        ["ALLOC", "R2", 2],         // R2 = Addr Node 2
        ["MOV", "R0", 0xBB],
        ["PTR_STORE", "R2", 0, "R0"],

        // Alloc Node 3 (Value 0xCC)
        ["ALLOC", "R3", 2],         // R3 = Addr Node 3
        ["MOV", "R0", 0xCC],
        ["PTR_STORE", "R3", 0, "R0"],

        // Chain Nodes
        ["PTR_STORE", "R1", 1, "R2"], // Node1[1] = Node 2 Addr (Link)
        ["PTR_STORE", "R2", 1, "R3"], // Node2[1] = Node 3 Addr (Link)

        // Traverse
        ["MOV", "PTR", "R1"],       // PTR = Node 1
        ["PTR_LOAD", "PTR", "PTR", 1], // PTR = * (PTR + 1) -> Node 2
        ["PTR_LOAD", "PTR", "PTR", 1], // PTR = * (PTR + 1) -> Node 3

        // Test FREE (Adding explicit test for FREE as requested by protocol 11)
        // User script didn't use it, but Protocol 11 listed it and review demanded it.
        // Let's Free Node 2?
        // But traverse relies on it.
        // Let's alloc a temp node and free it.
        ["ALLOC", "R4", 4],
        ["FREE", "R4"],

        ["EXIT"]
    ];

    vm.loadProgram(program, {});

    console.log("Compiling pointer assembly... Done.");
    console.log("TITAN-1 MK10 EXECUTION OUTPUT:");

    try {
        vm.run(100);
    } catch (e) {
        console.error(`Error: ${e.message}`);
    }

    console.log("\nVISUAL RESULT ON CANVAS:");
    console.log(vram.renderAscii());

    console.log("[STATUS: POINTERS_VISUALIZED]");
    // Estimates
    const allocations = vm.heap.allocations.size;
    console.log(`[HEAP_ALLOCATIONS: ${allocations}]`);
    console.log(`[POINTER_TETHERS: ${vram.lines.length} ACTIVE]`);
}

main();
