"""
pxOS GPU Terminal - Frozen Shader Architecture

A GPU-accelerated terminal for pxOS using the Frozen Shader Bus pattern.

Quick start:
    from pxos_gpu_terminal import PxOSTerminalGPU

    terminal = PxOSTerminalGPU()
    terminal.cmd_clear(0, 0, 0)
    terminal.cmd_rect(100, 100, 200, 150, 255, 0, 0)
    terminal.run()
"""

__version__ = "0.1.0"
__author__ = "pxOS Project"

from .pxos_gpu_terminal import PxOSTerminalGPU

__all__ = ["PxOSTerminalGPU"]
