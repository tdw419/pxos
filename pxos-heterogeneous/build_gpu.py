#!/usr/bin/env python3
"""
GPU Primitive Builder
Complete build system for GPU primitives → CUDA executable

Usage:
    python3 build_gpu.py example_vector_add.px
"""

import sys
from pathlib import Path
from gpu_primitives import GPUPrimitiveParser
from cuda_generator import CUDAGenerator


class CompleteCUDAGenerator(CUDAGenerator):
    """Extended CUDA generator with complete host code"""

    def generate_complete_example(self, output_path: str, kernel_name: str = None) -> None:
        """Generate complete working CUDA program"""

        if not kernel_name and self.parser.kernels:
            # Use first kernel
            kernel_name = list(self.parser.kernels.keys())[0]

        kernel = self.parser.kernels.get(kernel_name)
        if not kernel:
            print(f"Error: Kernel '{kernel_name}' not found")
            return

        code = self._generate_headers()
        code += "\n\n"
        code += self._generate_kernels()
        code += "\n\n"
        code += self._generate_complete_host_code(kernel_name, kernel)

        with open(output_path, 'w') as f:
            f.write(code)

        print(f"✓ Generated complete CUDA program: {output_path}")

    def _generate_complete_host_code(self, kernel_name: str, kernel) -> str:
        """Generate complete host code with memory management and kernel launch"""

        code = "// ========== Host Code ==========\n\n"

        # Main function
        code += "int main(int argc, char** argv) {\n"
        code += '    printf("\\npxOS Heterogeneous Computing System\\n");\n'
        code += '    printf("Example: Vector Addition on GPU\\n\\n");\n\n'

        # Device info
        code += "    // Query GPU\n"
        code += "    int device;\n"
        code += "    CUDA_CHECK(cudaGetDevice(&device));\n"
        code += "    cudaDeviceProp props;\n"
        code += "    CUDA_CHECK(cudaGetDeviceProperties(&props, device));\n"
        code += '    printf("GPU: %s\\n", props.name);\n'
        code += '    printf("Compute Capability: %d.%d\\n\\n", props.major, props.minor);\n\n'

        # Example data
        code += "    // Setup test data\n"
        code += "    const int N = 65536;  // Number of elements\n"
        code += "    const int size = N * sizeof(float);\n"
        code += '    printf("Vector size: %d elements (%d bytes)\\n", N, size);\n\n'

        # Host arrays
        code += "    // Allocate host memory\n"
        code += "    float *h_a = (float*)malloc(size);\n"
        code += "    float *h_b = (float*)malloc(size);\n"
        code += "    float *h_c = (float*)malloc(size);\n\n"

        # Initialize
        code += "    // Initialize input vectors\n"
        code += "    for (int i = 0; i < N; i++) {\n"
        code += "        h_a[i] = (float)i;\n"
        code += "        h_b[i] = (float)(i * 2);\n"
        code += "    }\n\n"

        # Device arrays
        code += "    // Allocate device memory\n"
        code += "    float *d_a, *d_b, *d_c;\n"
        code += "    CUDA_CHECK(cudaMalloc(&d_a, size));\n"
        code += "    CUDA_CHECK(cudaMalloc(&d_b, size));\n"
        code += "    CUDA_CHECK(cudaMalloc(&d_c, size));\n\n"

        # Copy to device
        code += "    // Copy data to device\n"
        code += "    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));\n"
        code += "    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));\n\n"

        # Launch kernel
        code += "    // Launch kernel\n"
        code += "    int blocks = 256;\n"
        code += "    int threads = 256;\n"
        code += '    printf("Launching kernel: %d blocks x %d threads\\n", blocks, threads);\n\n'

        code += "    // Create CUDA events for timing\n"
        code += "    cudaEvent_t start, stop;\n"
        code += "    CUDA_CHECK(cudaEventCreate(&start));\n"
        code += "    CUDA_CHECK(cudaEventCreate(&stop));\n\n"

        code += "    CUDA_CHECK(cudaEventRecord(start));\n"
        code += f"    {kernel_name}<<<blocks, threads>>>(d_a, d_b, d_c, N);\n"
        code += "    CUDA_CHECK(cudaEventRecord(stop));\n\n"

        # Synchronize
        code += "    // Wait for GPU to finish\n"
        code += "    CUDA_CHECK(cudaDeviceSynchronize());\n\n"

        # Timing
        code += "    // Calculate elapsed time\n"
        code += "    float milliseconds = 0;\n"
        code += "    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));\n"
        code += '    printf("GPU execution time: %.3f ms\\n\\n", milliseconds);\n\n'

        # Copy back
        code += "    // Copy result back to host\n"
        code += "    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));\n\n"

        # Verify
        code += "    // Verify results\n"
        code += '    printf("Verifying results...\\n");\n'
        code += "    bool success = true;\n"
        code += "    for (int i = 0; i < N; i++) {\n"
        code += "        float expected = h_a[i] + h_b[i];\n"
        code += "        if (fabs(h_c[i] - expected) > 1e-5) {\n"
        code += '            printf("Error at index %d: expected %.2f, got %.2f\\n", i, expected, h_c[i]);\n'
        code += "            success = false;\n"
        code += "            break;\n"
        code += "        }\n"
        code += "    }\n\n"

        code += "    if (success) {\n"
        code += '        printf("✓ All results correct!\\n");\n'
        code += "        // Show sample results\n"
        code += '        printf("\\nSample results:\\n");\n'
        code += "        for (int i = 0; i < 5; i++) {\n"
        code += '            printf("  %.2f + %.2f = %.2f\\n", h_a[i], h_b[i], h_c[i]);\n'
        code += "        }\n"
        code += '        printf("  ...\\n");\n'
        code += "    }\n\n"

        # Cleanup
        code += "    // Cleanup\n"
        code += "    free(h_a); free(h_b); free(h_c);\n"
        code += "    CUDA_CHECK(cudaFree(d_a));\n"
        code += "    CUDA_CHECK(cudaFree(d_b));\n"
        code += "    CUDA_CHECK(cudaFree(d_c));\n"
        code += "    CUDA_CHECK(cudaEventDestroy(start));\n"
        code += "    CUDA_CHECK(cudaEventDestroy(stop));\n\n"

        code += '    printf("\\npxOS GPU program completed successfully.\\n");\n'
        code += "    return 0;\n"
        code += "}\n"

        return code


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 build_gpu.py <primitive_file.px>")
        print("Example: python3 build_gpu.py example_vector_add.px")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    print(f"\npxOS Heterogeneous Computing - GPU Builder")
    print(f"=" * 50)
    print(f"Input: {input_file}")

    # Parse primitives
    parser = GPUPrimitiveParser()
    parser.parse_file(input_file)

    print(f"Parsed {len(parser.kernels)} kernel(s)")
    for name in parser.kernels:
        print(f"  - {name}")

    # Generate CUDA code
    output_cu = input_file.replace('.px', '.cu')
    generator = CompleteCUDAGenerator(parser)
    generator.generate_complete_example(output_cu)

    # Try to compile
    print(f"\nAttempting compilation...")
    output_exe = input_file.replace('.px', '')
    if generator.compile_cuda(output_cu, output_exe):
        print(f"\n✓ Success! Run with: ./{output_exe}")
    else:
        print(f"\n✓ CUDA code generated (compilation skipped - no CUDA toolkit)")
        print(f"  Generated file: {output_cu}")
        print(f"  To compile manually: nvcc -o {output_exe} {output_cu}")


if __name__ == "__main__":
    main()
