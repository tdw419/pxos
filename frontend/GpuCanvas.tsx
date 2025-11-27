import React, { useEffect, useRef, useState } from 'react';
import { GpuRuntime } from './GpuRuntime';

interface GpuCanvasProps {
    onVramUpdate: (vram: Uint32Array) => void;
}

export const GpuCanvas: React.FC<GpuCanvasProps> = ({ onVramUpdate }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const runtimeRef = useRef<GpuRuntime>(new GpuRuntime());
    const animationRef = useRef<number>(0);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const init = async () => {
            try {
                await runtimeRef.current.init();
                animate();
            } catch (e) {
                console.error(e);
                setError((e as Error).message);
            }
        };
        init();

        return () => {
            cancelAnimationFrame(animationRef.current);
        };
    }, []);

    const animate = async () => {
        try {
            const data = await runtimeRef.current.step();
            if (data) {
                onVramUpdate(data);

                // Also draw to canvas for debug/visual verification if needed,
                // but we rely on the main App to render the VRAM array.
                // However, we can also render here directly for "Pure GPU" mode proof.
                if (canvasRef.current) {
                    const ctx = canvasRef.current.getContext('2d');
                    if (ctx) {
                        const imgData = ctx.createImageData(32, 32);
                        // Convert Uint32 ARGB to Uint8 RGBA
                        const buf32 = new Uint32Array(imgData.data.buffer);
                        // Direct copy works if endianness matches, usually yes.
                        // Our shader outputs 0xFF00FF00 (A=FF, R=00, G=FF, B=00)
                        // Canvas expects RGBA.
                        // Little Endian machine: 0xAABBGGRR in memory.
                        // 0xFF00FF00 -> R=00, G=FF, B=00, A=FF.
                        // Wait, 0xFF00FF00 = 11111111 00000000 11111111 00000000
                        // In Little Endian memory: 00 FF 00 FF (BB GG RR AA)
                        // Canvas RGBA: R G B A
                        // So 0xAA BB GG RR is what we want for Uint32.
                        // If we want Green: R=0, G=255, B=0, A=255.
                        // Memory: 00 FF 00 FF.
                        // Uint32 Value: 0xFF00FF00.
                        // So it matches!
                        buf32.set(data);
                        ctx.putImageData(imgData, 0, 0);
                    }
                }
            }
            // Limit to ~30-60fps
            // animationRef.current = requestAnimationFrame(animate);
            // Async recursion
            setTimeout(() => {
                animationRef.current = requestAnimationFrame(animate);
            }, 50); // Slow down slightly for visibility
        } catch (e) {
            console.error(e);
        }
    };

    if (error) {
        return <div style={{ color: 'red', border: '1px solid red', padding: '10px' }}>GPU Error: {error}</div>;
    }

    return (
        <div style={{ position: 'relative' }}>
            <div style={{ position: 'absolute', top: -20, color: '#0f0', fontSize: '10px' }}>GPU_ACCELERATED_VIEWPORT</div>
            <canvas
                ref={canvasRef}
                width={32}
                height={32}
                style={{
                    width: '256px',
                    height: '256px',
                    imageRendering: 'pixelated',
                    border: '2px solid #0f0',
                    display: 'block' // Explicitly show this canvas
                }}
            />
        </div>
    );
};
