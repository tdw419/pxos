
from PIL import Image

# Define the instructions for the jump test program.
# Program:
# 1. (ip=0) JUMP to ip=2 (offset +2)
# 2. (ip=1) HALT (should be skipped initially)
# 3. (ip=2) JUMP to ip=1 (offset -1)
# This program should execute instruction 0, then 2, then 1, and then halt.
instructions = [
    (4, 2, 0, 255),    # JUMP +2
    (255, 0, 0, 255),  # HALT
    (4, 255, 0, 255),  # JUMP -1 (255 is -1 as a signed 8-bit integer)
]

# Create a new image with the size of our program.
img = Image.new('RGBA', (len(instructions), 1), "black")
pixels = img.load()

# Set the pixel values for each instruction.
for i, instruction in enumerate(instructions):
    pixels[i, 0] = instruction

# Save the image to a file.
img.save("pxos-v2.0/programs/jump_test.png")

print("Program 'jump_test.png' created successfully.")
