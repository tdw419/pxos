"""
Pixel LLM Integration for pxOS

This package provides AI-assisted kernel development tools,
GPU-accelerated neural network primitives, and pixel-native
bytecode execution for the pxOS operating system.

Key modules:
  - core.hypervisor: Pixel-native code execution controller
  - core.pixel_vm: Stack-based bytecode virtual machine
  - core.pixel_asm: Assembler for pixel bytecode
  - tools.pxos_kernel_architect: AI-powered kernel development assistant
  - tools.god_pixel: Extreme compression system (16,384:1)

GPU kernels (WGSL):
  - gpu_kernels/attention.wgsl: Transformer attention mechanism
  - gpu_kernels/activations.wgsl: Neural network activations
  - gpu_kernels/mailbox_runtime.wgsl: Hardware mailbox protocol handler

Usage:
  >>> from pixel_llm.core.pixel_vm import PixelVM
  >>> vm = PixelVM(debug=True)
  >>> vm.load_from_pixel("program.pxi.png")
  >>> vm.run()

For AI-assisted development:
  $ python3 pixel_llm/tools/pxos_kernel_architect.py --mode interactive
"""

__version__ = "1.0.0-phase2"
__author__ = "pxOS Team + Claude"
