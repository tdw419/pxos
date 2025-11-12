from PIL import Image
import numpy as np
from typing import List, Tuple

def generate_pxi_png(program: List[List[Tuple[int, int, int, int]]], output_path: str):
    """
    Generates a PXI program PNG from a 2D list of RGBA tuples.

    Args:
        program: A 2D list where each element is an RGBA tuple representing a PXI instruction.
        output_path: The path to save the generated PNG file.
    """
    height = len(program)
    width = len(program[0])

    # Create a numpy array from the program data
    arr = np.array(program, dtype=np.uint8)

    # Create an image from the array
    img = Image.fromarray(arr, 'RGBA')

    # Save the image
    img.save(output_path)
    print(f"Generated PXI program at '{output_path}'")

if __name__ == '__main__':
    # Example usage: create a 16x16 program of all NOPs
    NOP = (0, 0, 0, 255)
    program_data = [[NOP for _ in range(16)] for _ in range(16)]

    # Add a DRAW instruction at (8, 8)
    # The color of the DRAW instruction will be (0x60, 128, 128, 255)
    # R=0x60 (96), G=128, B=128, A=255
    DRAW_COLOR = (96, 128, 128, 255)
    program_data[8][8] = DRAW_COLOR

    generate_pxi_png(program_data, "example_program.png")
