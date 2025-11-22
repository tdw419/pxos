"""
Pixel Memory Manager (PMM)
Implements 2D memory allocation using Guillotine packing with Quadtree spatial indexing
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FreeRectangle:
    """Represents a free rectangular region in memory."""
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, width: int, height: int) -> bool:
        """Check if this rectangle can contain a block of given dimensions."""
        return self.width >= width and self.height >= height


class PixelMemoryManager:
    """
    Pixel Memory Manager - 2D Memory Allocator

    Uses the Guillotine algorithm with Best Area Fit (BAF) heuristic
    to allocate 16x16 pixel pages in a 2D grid.

    Key Features:
    - 16x16 page size (256 pixels = 768 bytes)
    - Guillotine packing with horizontal/vertical cuts
    - Best Area Fit heuristic for placement
    - Visual defragmentation support

    Memory Layout:
    - Memory is a (width, height, 3) NumPy array
    - Allocations are always aligned to 16x16 grid
    - Black pixels (0,0,0) represent free memory
    """

    def __init__(self, memory: np.ndarray, width: int, height: int):
        """
        Initialize the Pixel Memory Manager.

        Args:
            memory: Reference to the main memory array
            width: Screen width
            height: Screen height
        """
        self.memory = memory
        self.width = width
        self.height = height

        # Free rectangles list (Guillotine algorithm)
        # Start with one giant free rectangle: the entire screen
        self.free_rectangles: List[FreeRectangle] = [
            FreeRectangle(0, 0, width, height)
        ]

        # Track allocated regions: pid -> (x, y, w, h)
        self.allocated_regions: Dict[int, Tuple[int, int, int, int]] = {}

        # Statistics
        self.total_allocations = 0
        self.total_deallocations = 0

    def allocate_16x16_block(self, pid: int) -> Tuple[int, int]:
        """
        Allocate a 16x16 pixel block for a process.

        This implements the Guillotine algorithm with Best Area Fit:
        1. Find the smallest free rectangle that can fit a 16x16 block
        2. Place the block in the top-left corner of that rectangle
        3. Cut the remaining space into two new free rectangles

        Args:
            pid: Process ID requesting memory

        Returns:
            (x, y): Top-left coordinates of the allocated block

        Raises:
            MemoryError: If no space available
        """
        required_width = 16
        required_height = 16

        # Best Area Fit: Find the smallest rectangle that fits
        best_fit: Optional[FreeRectangle] = None
        best_index: int = -1

        for i, rect in enumerate(self.free_rectangles):
            if rect.contains(required_width, required_height):
                if best_fit is None or rect.area < best_fit.area:
                    best_fit = rect
                    best_index = i

        if best_fit is None:
            raise MemoryError(f"Out of pixel memory: Cannot allocate 16x16 block for PID {pid}")

        # Allocate at the top-left corner of the chosen rectangle
        x, y = best_fit.x, best_fit.y

        # Remove the chosen rectangle from the free list
        del self.free_rectangles[best_index]

        # Guillotine Cut: Split the remaining space
        # We make a horizontal cut first (can also try vertical cut for optimization)

        # Right remainder
        if best_fit.width > required_width:
            right_rect = FreeRectangle(
                x + required_width,
                y,
                best_fit.width - required_width,
                best_fit.height
            )
            self.free_rectangles.append(right_rect)

        # Bottom remainder
        if best_fit.height > required_height:
            bottom_rect = FreeRectangle(
                x,
                y + required_height,
                required_width,
                best_fit.height - required_height
            )
            self.free_rectangles.append(bottom_rect)

        # Record the allocation
        self.allocated_regions[pid] = (x, y, required_width, required_height)
        self.total_allocations += 1

        # Initialize the allocated memory with a PID-specific pattern
        self._initialize_block(pid, x, y, required_width, required_height)

        return (x, y)

    def deallocate_block(self, pid: int):
        """
        Free a previously allocated block.

        Args:
            pid: Process ID to free

        Raises:
            ValueError: If PID not found
        """
        if pid not in self.allocated_regions:
            raise ValueError(f"PID {pid} not found in allocated regions")

        x, y, w, h = self.allocated_regions[pid]

        # Create a new free rectangle
        freed_rect = FreeRectangle(x, y, w, h)
        self.free_rectangles.append(freed_rect)

        # Erase the memory (set to black)
        self.memory[x:x+w, y:y+h] = 0

        # Remove from allocated list
        del self.allocated_regions[pid]
        self.total_deallocations += 1

        # Merge adjacent free rectangles to reduce fragmentation
        self._merge_free_rectangles()

    def _initialize_block(self, pid: int, x: int, y: int, width: int, height: int):
        """
        Initialize a newly allocated block with a visual pattern.

        This creates the process "header" and initial memory state.

        Args:
            pid: Process ID
            x, y: Top-left corner
            width, height: Block dimensions
        """
        # Generate a PID-based color pattern
        # This makes different processes visually distinguishable
        base_color = (
            (pid * 50) % 256,
            (pid * 100) % 256,
            (pid * 150) % 256
        )

        # Fill the block with a gradient pattern
        for i in range(width):
            for j in range(height):
                color = (
                    (base_color[0] + i * 2) % 256,
                    (base_color[1] + j * 2) % 256,
                    (base_color[2] + (i + j)) % 256
                )
                self.memory[x + i, y + j] = color

        # Row 0: Process header (will be set by PixelProcess)
        # For now, just mark it with the PID
        self.memory[x, y] = (pid % 256, (pid // 256) % 256, 0)

    def _merge_free_rectangles(self):
        """
        Merge adjacent free rectangles to reduce fragmentation.

        This is a simplified version that merges rectangles with the same
        x or y coordinates and matching dimensions.
        """
        # Simple merge: look for rectangles that can be combined
        # A full implementation would use more sophisticated merging logic
        merged = True
        while merged:
            merged = False
            for i in range(len(self.free_rectangles)):
                for j in range(i + 1, len(self.free_rectangles)):
                    rect_a = self.free_rectangles[i]
                    rect_b = self.free_rectangles[j]

                    # Check if they can be merged horizontally
                    if (rect_a.y == rect_b.y and
                        rect_a.height == rect_b.height):
                        if rect_a.x + rect_a.width == rect_b.x:
                            # Merge rect_b into rect_a
                            new_rect = FreeRectangle(
                                rect_a.x,
                                rect_a.y,
                                rect_a.width + rect_b.width,
                                rect_a.height
                            )
                            self.free_rectangles[i] = new_rect
                            del self.free_rectangles[j]
                            merged = True
                            break

                    # Check if they can be merged vertically
                    if (rect_a.x == rect_b.x and
                        rect_a.width == rect_b.width):
                        if rect_a.y + rect_a.height == rect_b.y:
                            # Merge rect_b into rect_a
                            new_rect = FreeRectangle(
                                rect_a.x,
                                rect_a.y,
                                rect_a.width,
                                rect_a.height + rect_b.height
                            )
                            self.free_rectangles[i] = new_rect
                            del self.free_rectangles[j]
                            merged = True
                            break

                if merged:
                    break

    def get_free_memory(self) -> int:
        """
        Get the total amount of free memory in pixels.

        Returns:
            Total free pixels
        """
        return sum(rect.area for rect in self.free_rectangles)

    def get_fragmentation_ratio(self) -> float:
        """
        Calculate memory fragmentation ratio.

        Fragmentation = 1 - (largest_free_block / total_free_memory)

        Returns:
            Fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
        """
        total_free = self.get_free_memory()
        if total_free == 0:
            return 0.0

        largest_block = max(self.free_rectangles, key=lambda r: r.area).area if self.free_rectangles else 0
        return 1.0 - (largest_block / total_free)

    def defragment(self):
        """
        Perform visual defragmentation.

        This moves allocated blocks toward the origin (0,0) to consolidate
        free space. This is a visual operation that the user can observe.

        Note: In a real implementation, this would need to pause processes
        and update their base addresses. For now, this is a stub.
        """
        # TODO: Implement visual compaction
        # 1. Sort allocated regions by y, then x
        # 2. Move each region toward (0,0)
        # 3. Update process base addresses
        # 4. Animate the movement
        pass

    def print_memory_map(self):
        """Print a visual representation of the memory map."""
        print("\nMemory Map:")
        print(f"Free Rectangles: {len(self.free_rectangles)}")
        for i, rect in enumerate(self.free_rectangles):
            print(f"  {i}: ({rect.x}, {rect.y}) {rect.width}x{rect.height} = {rect.area} pixels")

        print(f"Allocated Regions: {len(self.allocated_regions)}")
        for pid, (x, y, w, h) in self.allocated_regions.items():
            print(f"  PID {pid}: ({x}, {y}) {w}x{h}")

        print(f"Fragmentation: {self.get_fragmentation_ratio():.2%}")
