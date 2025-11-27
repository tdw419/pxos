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

const generateStaticNoise = (): Uint32Array => {
  const buffer = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
  for (let i = 0; i < buffer.length; i++) {
    // Green/black static (looks like GPU memory activity)
    buffer[i] = Math.random() > 0.5 ? 0xFF00FF00 : 0xFF000000;
  }
  return buffer;
};

export const App = () => {
    const [vram, setVram] = useState<Uint32Array>(new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT));
    const vramRef = useRef<Uint32Array>(new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT));
    const [status, setStatusText] = useState<string>('');
    const [kernelStatus, setKernelStatus] = useState<KernelStatus>({
        pipelineState: 'IDLE',
        statusText: 'IDLE',
        cycleCount: 0
    });

    const setStatus = (text: string) => {
        setStatusText(text);
        setKernelStatus(prev => ({ ...prev, statusText: text }));
    }

    const simulateVisuals = async (command: string) => {
        const lowerCmd = command.toLowerCase();
        let newVram: Uint32Array | null = null;

        // ✅ Keep these — they are your "GPU ROM"
        if (lowerCmd.includes('drawtestpattern')) {
            newVram = await runGpu(GPUProgram.DrawTestPattern) || cpuDrawTestPattern();
            setStatus("EXEC: ROM_KERNEL_0x01 (TEST_PATTERN)");
        }
        if (lowerCmd.includes('staticnoise')) {
            newVram = await runGpu(GPUProgram.StaticNoise) || cpuStaticNoise();
            setStatus("EXEC: ROM_KERNEL_0x02 (STATIC_NOISE)");
        }

        // JIT Mode for Custom Code
        if (lowerCmd.startsWith('jit ')) {
            // Show "compiling" static
            vramRef.current = generateStaticNoise();
            setVram(vramRef.current);
            setStatus("CORE: JIT_COMPILING...");

            await new Promise(r => setTimeout(r, 600));

            // Then run the intended effect
            const effect = lowerCmd.split(' ')[1];
            if (effect === 'noise') {
                newVram = await runGpu(GPUProgram.StaticNoise) || cpuStaticNoise();
            }
            setStatus("GPU: KERNEL_EXECUTED");
        }

        if (newVram) {
            vramRef.current = newVram;
            setVram(newVram);
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
            <div>{status}</div>
        </div>
    );
};
