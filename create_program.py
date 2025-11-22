from PIL import Image

# Define the instructions for our simple program.
# Each instruction is a tuple of (R, G, B, A) values.
# Program:
# 1. LOAD data[0] -> reg[0]
# 2. LOAD data[1] -> reg[1]
# 3. ADD reg[0], reg[1] -> reg[1]
# 4. STORE reg[1] -> data[2]
# 5. HALT
instructions = [
    (1, 0, 0, 255),   # LOAD 0, 0
    (1, 1, 1, 255),   # LOAD 1, 1
    (3, 0, 1, 255),   # ADD 0, 1
    (2, 1, 2, 255),   # STORE 1, 2
    (255, 0, 0, 255), # HALT
]

# Create a new image with the size of our program.
# The program is a single horizontal line of pixels.
img = Image.new('RGBA', (len(instructions), 1), "black")
pixels = img.load()

# Set the pixel values for each instruction.
for i, instruction in enumerate(instructions):
    pixels[i, 0] = instruction

# Save the image to a file.
img.save("pxos-v2.0/programs/simple.png")

print("Program 'simple.png' created successfully.")
