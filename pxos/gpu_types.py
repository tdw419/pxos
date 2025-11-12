from typing import Generic, TypeVar

T = TypeVar('T')

class GPUBuffer(Generic[T]):
    """
    A placeholder type hint to represent a buffer of data that resides in GPU memory.
    The generic type `T` indicates the type of the elements in the buffer.

    Example:
        `data: GPUBuffer[float]` represents a buffer of floating-point numbers.
    """
    def __init__(self, size: int):
        # In a real implementation, this would allocate memory on the GPU.
        # For the compiler, it's just a type hint.
        self.size = size
        self.dtype = T
