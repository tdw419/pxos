import React, { useState, useRef, useEffect } from 'react';
import { PipelineState, KernelStatus } from './types';
import { GpuCanvas } from './GpuCanvas';
import { getShaderFromPrompt, SHADER_LIBRARY } from './shaders/library';

// Mock GPUProgram enum
enum GPUProgram {
  DrawTestPattern = 'DrawTestPattern',
  StaticNoise = 'StaticNoise'
}

// Mock runGpu and CPU fallbacks
const VRAM_WIDTH = 32;
const VRAM_HEIGHT = 32;

const runGpu = async (program: GPUProgram): Promise<Uint32Array | null> => {
  // Simulation of GPU execution
  return null; // Fallback to CPU for this mock
};

const cpuDrawTestPattern = (): Uint32Array => {
    const cpuVram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
    for (let y = 0; y < VRAM_HEIGHT; y++) {
        for (let x = 0; x < VRAM_WIDTH; x++) {
            const quadrant =
              (x < VRAM_WIDTH / 2 && y < VRAM_HEIGHT / 2) ? 0xFF0000FF : // Red
              (x >= VRAM_WIDTH / 2 && y < VRAM_HEIGHT / 2) ? 0xFF00FF00 : // Green
              (x < VRAM_WIDTH / 2 && y >= VRAM_HEIGHT / 2) ? 0xFFFF0000 : // Blue
              0xFFFFFFFF; // White
            cpuVram[y * VRAM_WIDTH + x] = quadrant;
        }
    }
    return cpuVram;
};

const cpuStaticNoise = (): Uint32Array => {
    const cpuVram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
    for (let i = 0; i < cpuVram.length; i++) {
        cpuVram[i] = Math.floor(Math.random() * 0xFFFFFF);
    }
    return cpuVram;
};

// Visual Noise Generators
const generateGreenStatic = (): Uint32Array => {
  const buffer = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
  for (let i = 0; i < buffer.length; i++) {
    buffer[i] = Math.random() > 0.5 ? 0xFF00FF00 : 0xFF000000;
  }
  return buffer;
};

const generateBlueStripes = (): Uint32Array => {
    const buffer = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
    for (let i = 0; i < buffer.length; i++) {
        buffer[i] = (i % 32 < 16) ? 0xFF0000FF : 0xFF000088;
    }
    return buffer;
};

const generateYellowBands = (): Uint32Array => {
    const buffer = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
    for (let y = 0; y < VRAM_HEIGHT; y++) {
        for (let x = 0; x < VRAM_WIDTH; x++) {
            buffer[y * VRAM_WIDTH + x] = (y % 2 === 0) ? 0xFFFFFF00 : 0xFF888800;
        }
    }
    return buffer;
};

// TITAN-2 Parallel Core Types
interface Instruction {
    op: string;
    dst?: number;
    src?: number;
    imm?: number;
    target?: number;
    a?: number;
    b?: number;
}

// Per-Core State Buffer Layout: [PLR0, PLR1, FLAGS, PC, N, S, E, W, NE, NW, SE, SW] (12 bytes per pixel)
const BYTES_PER_CORE = 12;

export const App = () => {
    const [vram, setVram] = useState<Uint32Array>(new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT));
    const vramRef = useRef<Uint32Array>(new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT));
    const [kernelStatus, setKernelStatus] = useState<KernelStatus>({
        pipelineState: 'IDLE',
        statusText: 'IDLE',
        cycleCount: 0
    });

    // Titan-2 State
    const [coreState, setCoreState] = useState<Uint8Array>(new Uint8Array(VRAM_WIDTH * VRAM_HEIGHT * BYTES_PER_CORE));
    const animationRef = useRef<number | null>(null);
    const isAnimating = useRef<boolean>(false);
    const [useGpuCanvas, setUseGpuCanvas] = useState(false);

    const setStatus = (text: string, state: PipelineState, metrics?: Partial<KernelStatus>) => {
        setKernelStatus(prev => ({ ...prev, statusText: text, pipelineState: state, ...metrics }));
    }

    // Callback for GPUCanvas to update global VRAM state
    const handleGpuVramUpdate = (newVram: Uint32Array) => {
        setVram(newVram);
    };

    // PixelVM Mode State
    const [pixelVmMode, setPixelVmMode] = useState(false);

    // Shader Lab State
    const [showShaderLab, setShowShaderLab] = useState(false);
    const [shaderPrompt, setShaderPrompt] = useState("");
    const [wgslCode, setWgslCode] = useState(SHADER_LIBRARY["life"]);
    const [shaderError, setShaderError] = useState<string | null>(null);
    const gpuCanvasRef = useRef<any>(null); // To access updateShader

    const handleGenerateShader = () => {
        // Simulate AI generation by lookup
        setStatus("AI: GENERATING_SHADER...", 'JIT_COMPILING');
        setTimeout(() => {
            const generated = getShaderFromPrompt(shaderPrompt);
            setWgslCode(generated);
            setStatus("AI: SHADER_GENERATED", 'IDLE');
        }, 500);
    };

    const handleRunShader = async () => {
        if (!useGpuCanvas) {
            setUseGpuCanvas(true);
        }
        if (gpuCanvasRef.current) {
            // Try remote compilation first if connected (mocked here for now as pure local)
            // In full version, this would call /api/generate-shader logic
            const error = await gpuCanvasRef.current.updateShader(wgslCode);
            if (error) {
                // Attempt to parse line number from error
                // Example: "Shader validation error: \n   ┌─ :5:16"
                const match = error.match(/:(\d+):\d+/);
                const line = match ? match[1] : '?';
                setShaderError(`Line ${line}: ${error}`);
                setStatus("GPU: COMPILATION_FAILED", 'IDLE');
            } else {
                setShaderError(null);
                setStatus("GPU: SHADER_ACTIVE", 'EXECUTING');
            }
        }
    };

    // TITAN-2 Executor Logic
    const stepPixelCore = (state: Uint8Array, x: number, y: number, program: Instruction[]) => {
        const off = (y * VRAM_WIDTH + x) * BYTES_PER_CORE;

        // Check HALT flag (bit 7 of FLAGS at offset 2)
        if (state[off + 2] & 0x80) return;

        const pc = state[off + 3];
        if (pc >= program.length) return; // Stop if PC out of bounds

        const instr = program[pc];
        let jumped = false;

        switch (instr.op) {
            case 'PLR0':
                state[off + 0] = instr.imm!;
                break;
            case 'PLR1':
                state[off + 1] = instr.imm!;
                break;
            case 'COUNT_NEIGHBORS': {
                // Neighbors are cached at off+4 (N), +5 (S), +6 (E), +7 (W)
                // For Game of Life we usually count diagonal neighbors too.
                // But the cache currently only holds NSEW.
                // Let's assume we need to fetch diagonals from the main buffer or assume 8-neighbor read was done.
                // For simplicity in this demo, we will implement 8-way read in the NEIGHBOR PHASE or just do it here.
                // The prompt says "READ_N... Diagonals can be derived".
                // Let's just count N+S+E+W stored in cache for now, or cheat and read the global buffer since we have it.
                // Wait, the prompt for Game of Life says "READ_N... and diagonals".
                // Let's upgrade our NEIGHBOR PHASE to cache all 8 or just read adjacent indices here.
                // Actually, to stick to the "cache" concept: Let's assume COUNT_NEIGHBORS counts alive (>0) values in N,S,E,W + Diagonals derived from neighbor's N/E/S/W? No too complex.
                // Let's implement 8-neighbor read in the Neighbor Phase.
                // Update: expanding BYTES_PER_CORE to 12 to hold NE, NW, SE, SW would be cleaner, or just read from `state` but read previous frame's values?
                // The `titan2FrameStep` copies `cur` to `next`. `state` here is `next`. We shouldn't read `next`'s neighbors as they might have updated?
                // Actually, `titan2FrameStep` passes `next` to `stepPixelCore`.
                // So we should use the cached values which came from `cur`.
                // Let's implement 8-neighbor caching in `titan2FrameStep` for the `life` command.

                let count = 0;
                for(let i=4; i<12; i++) {
                    if (state[off + i] > 0) count++;
                }
                state[off + 1] = count; // Store in R1
                break;
            }
            case 'CMP_IMM_R1': {
                const r1 = state[off + 1];
                const imm = instr.imm!;
                // Set ZERO flag (bit 0)
                if (r1 === imm) state[off + 2] |= 0x01;
                else state[off + 2] &= ~0x01;
                break;
            }
            case 'JE': {
                // Check ZERO flag
                if (state[off + 2] & 0x01) {
                    state[off + 3] = instr.target!;
                    jumped = true;
                }
                break;
            }
            case 'JMP':
                state[off + 3] = instr.target!;
                jumped = true;
                break;
            case 'BARRIER':
                // Implicitly handled by frame step, just NOP here or wait?
                // Usually PC increments past it.
                break;
            case 'HALT':
                state[off + 2] |= 0x80; // Set HALT flag
                break;
            case 'MOV':
                if (instr.dst === 0) state[off + 0] = instr.imm!;
                else if (instr.dst === 1) state[off + 1] = instr.imm!;
                break;
        }

        if (!jumped) {
            state[off + 3]++;
        }
    };

    const titan2FrameStep = (program: Instruction[]) => {
        let nextCoreState: Uint8Array | null = null;
        let nextVram: Uint32Array | null = null;
        let frameNeighborOps = 0;

        setCoreState(currentCoreState => {
            const next = new Uint8Array(currentCoreState);
            const CORE_SIZE = BYTES_PER_CORE; // Use global constant (12)

            // Phase 1: Neighbor Cache (Toroidal)
            for (let y = 0; y < VRAM_HEIGHT; y++) {
                for (let x = 0; x < VRAM_WIDTH; x++) {
                    const off = (y * VRAM_WIDTH + x) * CORE_SIZE;

                    // Indices
                    const n = (y - 1 + VRAM_HEIGHT) % VRAM_HEIGHT;
                    const s = (y + 1) % VRAM_HEIGHT;
                    const w = (x - 1 + VRAM_WIDTH) % VRAM_WIDTH;
                    const e = (x + 1) % VRAM_WIDTH;

                    const getN = (ax: number, ay: number) => currentCoreState[(ay * VRAM_WIDTH + ax) * CORE_SIZE + 0]; // Read R0 from CURRENT state

                    next[off + 4] = getN(x, n); // N
                    next[off + 5] = getN(x, s); // S
                    next[off + 6] = getN(e, y); // E
                    next[off + 7] = getN(w, y); // W
                    next[off + 8] = getN(e, n); // NE
                    next[off + 9] = getN(w, n); // NW
                    next[off + 10] = getN(e, s); // SE
                    next[off + 11] = getN(w, s); // SW
                }
            }

            // Phase 2: Execution
            let neighborOps = 0;
            for (let y = 0; y < VRAM_HEIGHT; y++) {
                for (let x = 0; x < VRAM_WIDTH; x++) {
                    const off = (y * VRAM_WIDTH + x) * CORE_SIZE;
                    let instrCount = 0;
                    const MAX_INSTR = 20;
                    while (instrCount < MAX_INSTR) {
                        const pc = next[off + 3];
                        if (pc >= program.length) break;

                        const op = program[pc].op;
                        if (op === 'BARRIER') {
                            next[off + 3]++; // Pass barrier
                            break;
                        }
                        if (op === 'COUNT_NEIGHBORS') neighborOps += 8;

                        stepPixelCore(next, x, y, program);
                        instrCount++;

                        if (next[off + 2] & 0x80) break;
                    }
                }
            }
            frameNeighborOps = neighborOps;

            // Phase 3: Render to VRAM logic (calculated here but set outside)
            const calculatedVram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
            for (let i = 0; i < VRAM_WIDTH * VRAM_HEIGHT; i++) {
                const r0 = next[i * CORE_SIZE + 0];
                calculatedVram[i] = r0 === 255 ? 0xFFFFFFFF : 0xFF000000;
            }
            nextVram = calculatedVram;
            nextCoreState = next;

            return next;
        });

        // Side effects outside the reducer (in a real app this might need useEffect but this is cleaner for simulation)
        if (nextVram) {
            setVram(nextVram);
            vramRef.current = nextVram;
            setStatus("PARALLEL: 1024 CORES ACTIVE", 'EXECUTING', {
                activeCores: 1024,
                neighborOps: frameNeighborOps
            });
        }
    };

    // NEW: JIT Compilation Simulation
    const simulateCompilationPipeline = async (command: string) => {
        // Only simulate for "custom" shaders (marked with jit prefix), not ROM kernels
        if (command.startsWith('jit ') || command.includes('shader_')) {
            // STAGE 1: JIT_COMPILE (Green static = parsing/IR generation)
            setStatus("CORE: JIT_COMPILING_WGSL...", 'JIT_COMPILING');
            vramRef.current = generateGreenStatic();
            setVram(vramRef.current);
            await new Promise(r => setTimeout(r, 600));

            // STAGE 2: SPIRV_EMIT (Blue stripes = code generation)
            setStatus("SPIRV: GENERATING_BINARY...", 'SPIRV_EMIT');
            vramRef.current = generateBlueStripes();
            setVram(vramRef.current);
            await new Promise(r => setTimeout(r, 300));

            // STAGE 3: GPU_UPLOAD (Yellow bands = DMA transfer)
            setStatus("BUS: DMA_CPU_TO_GPU...", 'GPU_UPLOAD');
            vramRef.current = generateYellowBands();
            setVram(vramRef.current);
            await new Promise(r => setTimeout(r, 200));

            return true; // Pipeline completed
        }
        return false; // Skip for ROM kernels
    };

    const simulateVisuals = async (command: string) => {
        const lowerCmd = command.toLowerCase();
        let newVram: Uint32Array | null = null;

        if (lowerCmd.includes('gpu_engage') || lowerCmd.includes('gpu_life')) {
            setUseGpuCanvas(true);
            setStatus("SYSTEM: GPU_ACCELERATION_ENGAGED", 'EXECUTING');
            return;
        }

        if (lowerCmd.includes('gpu_disengage')) {
            setUseGpuCanvas(false);
            setStatus("SYSTEM: GPU_ACCELERATION_DISENGAGED", 'IDLE');
            return;
        }

        // Handle Conway's Game of Life
        if (lowerCmd.includes('life') || lowerCmd.includes('conway') || lowerCmd.includes('glider')) {
            if (isAnimating.current) {
                cancelAnimationFrame(animationRef.current!);
                isAnimating.current = false;
            }

            // Seed Glider
            const CORE_SIZE = 12;
            const initialState = new Uint8Array(VRAM_WIDTH * VRAM_HEIGHT * CORE_SIZE);
            // Glider at 10,10
            const setAlive = (x: number, y: number) => {
                initialState[((y * VRAM_WIDTH) + x) * CORE_SIZE + 0] = 255;
            };
            setAlive(1, 0);
            setAlive(2, 1);
            setAlive(0, 2);
            setAlive(1, 2);
            setAlive(2, 2);

            setCoreState(initialState);
            isAnimating.current = true;

            const LIFE_PROGRAM: Instruction[] = [
                { op: 'COUNT_NEIGHBORS' },         // 0
                { op: 'CMP_IMM_R1', imm: 3 },      // 1
                { op: 'JE', target: 7 },           // 2 -> BORN
                { op: 'CMP_IMM_R1', imm: 2 },      // 3
                { op: 'JE', target: 9 },           // 4 -> STAY
                { op: 'PLR0', imm: 0 },            // 5 -> DIE
                { op: 'JMP', target: 9 },          // 6 -> BARRIER
                { op: 'PLR0', imm: 255 },          // 7 -> BORN
                { op: 'JMP', target: 9 },          // 8 -> BARRIER
                { op: 'BARRIER' },                 // 9 -> STAY (no change to R0), wait
                { op: 'JMP', target: 0 }           // 10 -> Loop back to start
            ];

            const animate = () => {
                if (!isAnimating.current) return;
                titan2FrameStep(LIFE_PROGRAM);
                animationRef.current = requestAnimationFrame(animate);
            };
            animate();
            return;
        }

        // Run compilation simulation for custom shaders
        const compiled = await simulateCompilationPipeline(lowerCmd);

        // ✅ Keep these — they are your "GPU ROM" (Fast Path)
        if (lowerCmd.includes('drawtestpattern')) {
            newVram = await runGpu(GPUProgram.DrawTestPattern) || cpuDrawTestPattern();
            setStatus("EXEC: ROM_KERNEL_0x01 (TEST_PATTERN)", 'EXECUTING');
        }
        if (lowerCmd.includes('staticnoise')) {
            newVram = await runGpu(GPUProgram.StaticNoise) || cpuStaticNoise();
            setStatus("EXEC: ROM_KERNEL_0x02 (STATIC_NOISE)", 'EXECUTING');
        }

        // If JIT compiled custom shader, execute result (Simulated effect)
        if (compiled) {
            // Check what effect was requested
             const effect = lowerCmd.split(' ')[1];
             if (effect === 'noise') {
                 newVram = await runGpu(GPUProgram.StaticNoise) || cpuStaticNoise();
             }
             setStatus("GPU: KERNEL_EXECUTED", 'EXECUTING');
        }

        if (newVram) {
            vramRef.current = newVram;
            setVram(newVram);
        }

        // Reset to IDLE after a short delay if we executed something
        if (newVram || compiled) {
             setTimeout(() => setStatus("IDLE", 'IDLE'), 1000);
        }
    };

    const handleSend = async (input: string) => {
        if (input.startsWith("lab") || input === "shader_lab") {
            setShowShaderLab(true);
            return;
        }
        if (input.startsWith("vm:") || input === "pixel_vm") {
            setPixelVmMode(true);
            setStatus("SYSTEM: PIXEL_VM_MODE_ENGAGED", 'IDLE');
            return;
        }
        await simulateVisuals(input);
    };

    // Tiny TS Launcher
    const [code, setCode] = useState(`// Access 'vram', 'VRAM_WIDTH', 'VRAM_HEIGHT'
// Return true to trigger updates
for (let i = 0; i < vram.length; i++) {
  vram[i] = Math.random() > 0.5 ? 0xFFFFFFFF : 0xFF000000;
}
return true;`);
    const [showLauncher, setShowLauncher] = useState(false);

    const runLauncherCode = () => {
        try {
            // Safe-ish execution environment
            const func = new Function('vram', 'VRAM_WIDTH', 'VRAM_HEIGHT', 'setStatus', code);
            const shouldUpdate = func(vramRef.current, VRAM_WIDTH, VRAM_HEIGHT, setStatus);
            if (shouldUpdate) {
                setVram(new Uint32Array(vramRef.current)); // Trigger re-render
                setStatus("LAUNCHER: CODE_EXECUTED", 'EXECUTING');
                // Persist as boot script
                try {
                    localStorage.setItem('pixelos_boot_script', code);
                } catch (e) { /* Ignore storage errors */ }
            }
        } catch (e) {
            setStatus(`LAUNCHER ERROR: ${(e as Error).message}`, 'IDLE');
        }
    };

    // Cleanup & Boot Script
    useEffect(() => {
        // Check for boot script
        try {
            const bootScript = localStorage.getItem('pixelos_boot_script');
            if (bootScript) {
                setCode(bootScript);
                setStatus("BOOT: SCRIPT_LOADED", 'IDLE');
            }
        } catch (e) { /* Ignore */ }

        return () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
        };
    }, []);

    return (
        <div style={{ background: '#000', color: '#0f0', fontFamily: 'monospace', padding: '20px' }}>
            {/* Canvas Display */}
            <div style={{
                width: '256px',
                height: '256px',
                position: 'relative',
                marginBottom: '20px',
                border: '2px solid #333'
            }}>
                {useGpuCanvas ? (
                    <GpuCanvas
                        ref={gpuCanvasRef}
                        onVramUpdate={handleGpuVramUpdate}
                    />
                ) : (
                    Array.from(vram).map((pixel, i) => {
                        const r = (pixel >> 24) & 0xFF;
                        const g = (pixel >> 16) & 0xFF;
                        const b = (pixel >> 8) & 0xFF;
                        return (
                            <div key={i} style={{
                                position: 'absolute',
                                left: (i % VRAM_WIDTH) * 8,
                                top: Math.floor(i / VRAM_WIDTH) * 8,
                                width: 8,
                                height: 8,
                                backgroundColor: `rgb(${r},${g},${b})`
                            }} />
                        );
                    })
                )}
            </div>

            {/* Status Panel */}
            <div style={{ borderTop: '1px solid #333', paddingTop: '10px' }}>
                <div>STATUS: {kernelStatus.statusText}</div>
                <div>PIPELINE: {kernelStatus.pipelineState}</div>
                {kernelStatus.activeCores && <div>ACTIVE CORES: {kernelStatus.activeCores}</div>}
                {kernelStatus.neighborOps && <div>NEIGHBOR OPS: {kernelStatus.neighborOps}/frame</div>}
            </div>

            {/* Input (Mock) */}
            <input
                type="text"
                onKeyDown={(e) => e.key === 'Enter' && handleSend(e.currentTarget.value)}
                style={{ background: '#111', color: '#0f0', border: 'none', marginTop: '10px', width: '100%' }}
                placeholder="Enter command..."
            />

            {/* Shader Lab Toggle */}
            <div style={{ marginTop: '10px', borderTop: '1px solid #333', paddingTop: '5px', display: 'flex', gap: '10px' }}>
                <button
                    onClick={() => setShowLauncher(!showLauncher)}
                    style={{ background: '#333', color: '#fff', border: 'none', padding: '5px', cursor: 'pointer' }}
                >
                    {showLauncher ? 'Hide TS Launcher' : 'Show TS Launcher'}
                </button>
                <button
                    onClick={() => setShowShaderLab(!showShaderLab)}
                    style={{ background: '#004444', color: '#fff', border: 'none', padding: '5px', cursor: 'pointer' }}
                >
                    {showShaderLab ? 'Hide Shader Lab' : 'Show Shader Lab (New)'}
                </button>
            </div>

            {/* Shader Lab UI */}
            {showShaderLab && (
                <div style={{ marginTop: '10px', border: '1px solid #008888', padding: '5px', background: '#001111' }}>
                    <div style={{ marginBottom: '5px', color: '#0ff', fontSize: '12px' }}>
                        AI SHADER GENERATOR (WGSL)
                    </div>

                    {/* Prompt Input */}
                    <div style={{ display: 'flex', marginBottom: '5px' }}>
                        <input
                            type="text"
                            value={shaderPrompt}
                            onChange={(e) => setShaderPrompt(e.target.value)}
                            placeholder="Ask AI: 'draw plasma', 'red circle', 'game of life'..."
                            style={{ flex: 1, background: '#002222', color: '#0ff', border: '1px solid #004444', padding: '5px' }}
                            onKeyDown={(e) => e.key === 'Enter' && handleGenerateShader()}
                        />
                        <button onClick={handleGenerateShader} style={{ background: '#004444', color: '#0ff', border: 'none', padding: '0 10px' }}>
                            GENERATE
                        </button>
                    </div>

                    {/* Editor */}
                    <textarea
                        value={wgslCode}
                        onChange={(e) => setWgslCode(e.target.value)}
                        style={{
                            width: '100%',
                            height: '150px',
                            background: '#002222',
                            color: '#0f0',
                            fontFamily: 'monospace',
                            border: 'none',
                            fontSize: '11px'
                        }}
                    />

                    {/* Error Display */}
                    {shaderError && (
                        <div style={{ color: 'red', fontSize: '11px', marginTop: '5px', whiteSpace: 'pre-wrap' }}>
                            {shaderError}
                        </div>
                    )}

                    <button
                        onClick={handleRunShader}
                        style={{
                            marginTop: '5px',
                            background: '#008800',
                            color: '#fff',
                            border: '1px solid #0f0',
                            padding: '5px 10px',
                            cursor: 'pointer',
                            width: '100%'
                        }}
                    >
                        COMPILE & RUN ON GPU
                    </button>
                </div>
            )}

            {/* Launcher UI */}
            {showLauncher && (
                <div style={{ marginTop: '10px', border: '1px solid #444', padding: '5px' }}>
                    <div style={{ marginBottom: '5px', color: '#aaa', fontSize: '12px' }}>
                        Inject TypeScript/JS to run directly in the simulator context.
                    </div>
                    <textarea
                        value={code}
                        onChange={(e) => setCode(e.target.value)}
                        style={{
                            width: '100%',
                            height: '100px',
                            background: '#222',
                            color: '#0f0',
                            fontFamily: 'monospace',
                            border: 'none'
                        }}
                    />
                    <button
                        onClick={runLauncherCode}
                        style={{
                            marginTop: '5px',
                            background: '#004400',
                            color: '#0f0',
                            border: '1px solid #0f0',
                            padding: '5px 10px',
                            cursor: 'pointer'
                        }}
                    >
                        RUN INJECTED CODE
                    </button>
                </div>
            )}
        </div>
    );
};
