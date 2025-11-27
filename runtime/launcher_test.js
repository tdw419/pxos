
// Verification script for Tiny TS Launcher Logic

const VRAM_WIDTH = 32;
const VRAM_HEIGHT = 32;
const vram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);

const setStatus = (text, state) => {
    console.log(`STATUS: ${text} [${state}]`);
};

// The user code (as string)
const code = `
// Access 'vram', 'VRAM_WIDTH', 'VRAM_HEIGHT'
// Return true to trigger updates
for (let i = 0; i < vram.length; i++) {
  vram[i] = i % 2 === 0 ? 0xFFFFFFFF : 0xFF000000;
}
setStatus("Test Code Ran", "EXECUTING");
return true;
`;

try {
    // Safe-ish execution environment
    const func = new Function('vram', 'VRAM_WIDTH', 'VRAM_HEIGHT', 'setStatus', code);
    const shouldUpdate = func(vram, VRAM_WIDTH, VRAM_HEIGHT, setStatus);

    if (shouldUpdate) {
        console.log("Launcher returned true (update triggered)");
        // Check if vram was modified
        if (vram[0] === 0xFFFFFFFF && vram[1] === 0xFF000000) {
            console.log("VRAM modification verified.");
        } else {
            console.error("VRAM not modified correctly.");
        }
    } else {
        console.log("Launcher returned false");
    }
} catch (e) {
    console.error(`LAUNCHER ERROR: ${e.message}`);
}
