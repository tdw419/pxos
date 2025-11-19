"""
GPU Kernels (WGSL)

WGSL compute shaders for GPU-accelerated operations:
  - attention.wgsl: Transformer scaled dot-product attention
  - activations.wgsl: Neural network activation functions
  - mailbox_runtime.wgsl: Hardware mailbox protocol handler

These shaders execute on the GPU and provide the compute
primitives for the pxOS GPU runtime.
"""
