import React, { useState, useRef } from 'react';
import { PIXEL_OS_SYSTEM_PROMPT } from './constants';
import { PipelineState, KernelStatus } from './types';

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

export const App = () => {
    const [vram, setVram] = useState<Uint32Array>(new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT));
    const vramRef = useRef<Uint32Array>(new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT));
    const [kernelStatus, setKernelStatus] = useState<KernelStatus>({
        pipelineState: 'IDLE',
        statusText: 'IDLE',
        cycleCount: 0
    });

    const setStatus = (text: string, state: PipelineState) => {
        setKernelStatus(prev => ({ ...prev, statusText: text, pipelineState: state }));
    }

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
        await simulateVisuals(input);
        // ... sendMessageToKernel logic ...
    };

    return (
        <div>
            {/* Mock UI */}
            <div>PixelOS Terminal</div>
            <div>{kernelStatus.statusText}</div>
            <div>Pipeline: {kernelStatus.pipelineState}</div>
        </div>
    );
};
