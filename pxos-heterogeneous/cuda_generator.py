#!/usr/bin/env python3
"""
CUDA Code Generator
Generates complete CUDA C files from GPU primitives
"""

from gpu_primitives import GPUPrimitiveParser, GPUKernel, GPULaunch
from typing import List
import subprocess
from pathlib import Path


class CUDAGenerator:
    """Generates CUDA C code from GPU primitives"""

    def __init__(self, parser: GPUPrimitiveParser):
        self.parser = parser

    def generate_cuda_file(self, output_path: str) -> None:
        """Generate complete CUDA .cu file"""
        code = self._generate_headers()
        code += "\n\n"
        code += self._generate_kernels()
        code += "\n\n"
        code += self._generate_host_code()

        with open(output_path, 'w') as f:
            f.write(code)

        print(f"Generated CUDA code: {output_path}")

    def _generate_headers(self) -> str:
        """Generate CUDA headers"""
        return """// Generated from GPU primitives
// pxOS Heterogeneous Computing System

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)
"""

    def _generate_kernels(self) -> str:
        """Generate all kernel definitions"""
        code = "// ========== GPU Kernels ==========\n\n"

        for kernel_name, kernel in self.parser.kernels.items():
            code += kernel.to_cuda_code()
            code += "\n"

        return code

    def _generate_host_code(self) -> str:
        """Generate host code (main function, etc.)"""
        code = "// ========== Host Code ==========\n\n"
        code += "int main(int argc, char** argv) {\n"
        code += "    printf(\"pxOS Heterogeneous Computing - CUDA Runtime\\n\");\n\n"

        # Device info
        code += "    // Get device properties\n"
        code += "    int device;\n"
        code += "    CUDA_CHECK(cudaGetDevice(&device));\n"
        code += "    cudaDeviceProp props;\n"
        code += "    CUDA_CHECK(cudaGetDeviceProperties(&props, device));\n"
        code += '    printf("Using GPU: %s\\n", props.name);\n\n'

        code += "    // TODO: Allocate memory, transfer data, launch kernels\n"
        code += '    printf("GPU kernels defined. Ready for integration.\\n");\n\n'

        code += "    return 0;\n"
        code += "}\n"

        return code

    def compile_cuda(self, cu_file: str, output_exe: str = "gpu_program") -> bool:
        """Compile CUDA code using nvcc"""
        try:
            # Check if nvcc is available
            result = subprocess.run(['nvcc', '--version'],
                                   capture_output=True, text=True)
            print(f"Found CUDA compiler: {result.stdout.split()[4]}")

            # Compile
            print(f"Compiling {cu_file}...")
            compile_result = subprocess.run(
                ['nvcc', '-o', output_exe, cu_file],
                capture_output=True,
                text=True
            )

            if compile_result.returncode == 0:
                print(f"✓ Compilation successful: {output_exe}")
                return True
            else:
                print(f"✗ Compilation failed:")
                print(compile_result.stderr)
                return False

        except FileNotFoundError:
            print("✗ nvcc not found. Install CUDA Toolkit to compile.")
            print("  On Ubuntu: sudo apt install nvidia-cuda-toolkit")
            return False


def main():
    """Test CUDA generation"""
    # Sample primitives
    test_primitives = """
GPU_KERNEL vector_add
GPU_PARAM a float[]
GPU_PARAM b float[]
GPU_PARAM c float[]
GPU_PARAM n int

GPU_THREAD_CODE:
    THREAD_ID → tid
    COMPARE tid < n → in_bounds

    IF in_bounds:
        LOAD a[tid] → val_a
        LOAD b[tid] → val_b
        ADD val_a val_b → sum
        STORE sum → c[tid]
    ENDIF
GPU_END
"""

    # Parse primitives
    parser = GPUPrimitiveParser()
    for line_num, line in enumerate(test_primitives.split('\n'), 1):
        parser.parse_line(line, line_num)

    # Generate CUDA code
    generator = CUDAGenerator(parser)
    generator.generate_cuda_file("test_kernel.cu")

    # Try to compile
    generator.compile_cuda("test_kernel.cu", "test_kernel")


if __name__ == "__main__":
    main()
