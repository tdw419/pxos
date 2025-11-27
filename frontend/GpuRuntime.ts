// frontend/GpuRuntime.ts

// Default Shader: Game of Life
export const DEFAULT_SHADER_CODE = `
@group(0) @binding(0) var<storage, read> inputGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid: array<u32>;

const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;

fn get_idx(x: u32, y: u32) -> u32 {
    return y * WIDTH + x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    let idx = get_idx(x, y);
    let alive = inputGrid[idx] > 0u;
    var neighbors = 0u;

    // Iterate 3x3 grid
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            // Toroidal wrap
            let nx = u32((i32(x) + dx + i32(WIDTH)) % i32(WIDTH));
            let ny = u32((i32(y) + dy + i32(HEIGHT)) % i32(HEIGHT));

            if (inputGrid[get_idx(nx, ny)] > 0u) {
                neighbors++;
            }
        }
    }

    var next_state = 0u; // Dead (Black)
    let ALIVE_COLOR = 0xFF00FF00u; // Green

    if (alive) {
        if (neighbors == 2u || neighbors == 3u) {
            next_state = ALIVE_COLOR;
        }
    } else {
        if (neighbors == 3u) {
            next_state = ALIVE_COLOR;
        }
    }

    outputGrid[idx] = next_state;
}
`;

export class GpuRuntime {
    device: GPUDevice | null = null;
    pipeline: GPUComputePipeline | null = null;
    bindGroupA: GPUBindGroup | null = null; // Input: Buffer A, Output: Buffer B
    bindGroupB: GPUBindGroup | null = null; // Input: Buffer B, Output: Buffer A
    bufferA: GPUBuffer | null = null;
    bufferB: GPUBuffer | null = null;
    stagingBuffer: GPUBuffer | null = null;
    frame: number = 0;
    bindGroupLayout: GPUBindGroupLayout | null = null;

    async init(initialShaderCode: string = DEFAULT_SHADER_CODE) {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No WebGPU adapter found");
        }

        this.device = await adapter.requestDevice();

        // Create Buffers (32x32 * 4 bytes)
        const bufferSize = 32 * 32 * 4;
        this.bufferA = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.bufferB = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.stagingBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Bind Group Layout
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
            ]
        });

        // Bind Groups (Ping-Pong)
        this.bindGroupA = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferA } },
                { binding: 1, resource: { buffer: this.bufferB } }
            ]
        });

        this.bindGroupB = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferB } },
                { binding: 1, resource: { buffer: this.bufferA } }
            ]
        });

        // Initialize Buffer A with random noise
        const initialData = new Uint32Array(32 * 32);
        for (let i = 0; i < initialData.length; i++) {
            initialData[i] = Math.random() > 0.8 ? 0xFF00FF00 : 0;
        }
        this.device.queue.writeBuffer(this.bufferA, 0, initialData);

        // Initial compilation
        const error = await this.updateShader(initialShaderCode);
        if (error) {
            console.error("Initial shader failed:", error);
        }
    }

    async updateShader(code: string): Promise<string | null> {
        if (!this.device || !this.bindGroupLayout) return "Device not initialized";

        try {
            this.device.pushErrorScope('validation');

            const shaderModule = this.device.createShaderModule({
                code: code
            });

            // Check for shader compilation errors
            // Note: createShaderModule might not throw immediately, pipeline creation will

            const pipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
                compute: {
                    module: shaderModule,
                    entryPoint: "main"
                }
            });

            const error = await this.device.popErrorScope();
            if (error) {
                return error.message;
            }

            this.pipeline = pipeline;
            return null; // Success
        } catch (e) {
            return (e as Error).message;
        }
    }

    async step(): Promise<Uint32Array | null> {
        if (!this.device || !this.pipeline || !this.bindGroupA || !this.bindGroupB) return null;

        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        passEncoder.setPipeline(this.pipeline);
        // Ping-Pong: Even frame reads A writes B. Odd frame reads B writes A.
        const bindGroup = (this.frame % 2 === 0) ? this.bindGroupA : this.bindGroupB;
        passEncoder.setBindGroup(0, bindGroup);

        // Dispatch 32/8 = 4 workgroups in X and Y
        passEncoder.dispatchWorkgroups(4, 4);
        passEncoder.end();

        // Copy output to staging buffer for reading
        const sourceBuffer = (this.frame % 2 === 0) ? this.bufferB! : this.bufferA!;
        commandEncoder.copyBufferToBuffer(sourceBuffer, 0, this.stagingBuffer!, 0, 32 * 32 * 4);

        this.device.queue.submit([commandEncoder.finish()]);

        // Map and read
        await this.stagingBuffer!.mapAsync(GPUMapMode.READ);
        const arrayBuffer = this.stagingBuffer!.getMappedRange();
        const data = new Uint32Array(arrayBuffer.slice(0)); // Copy data
        this.stagingBuffer!.unmap();

        this.frame++;
        return data;
    }
}
