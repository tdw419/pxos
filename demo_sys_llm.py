#!/usr/bin/env python3
from pxi_cpu import PXICPU
from PIL import Image

# Create a program that asks the LLM "Who are you?"
img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
def write(pc, r, g, b, a=0):
    y, x = divmod(pc, 64)
    img.putpixel((x, y), (r, g, b, a))

# Write prompt: "Who are you?" as ASCII in G channel
prompt = "Who are you?"
for i, c in enumerate(prompt + "\0"):
    write(i, ord(c), ord(c), ord(c))  # R/B = junk, G = ASCII

# Program:
# 0: LOAD R0, 0          → prompt address
# 1: LOAD R1, 100        → output buffer
# 2: LOAD R2, 500        → max length
# 3: SYS_LLM
# 4: HALT
# The program starts at pc=64, after the prompt
prog_start = len(prompt) + 1
write(prog_start + 0, 0x10, 0, 0)        # LOAD R0, 0
write(prog_start + 1, 0x10, 1, 100)      # LOAD R1, 100
write(prog_start + 2, 0x10, 2, 500)      # LOAD R2, 500
write(prog_start + 3, 0xC8, 0, 0)        # SYS_LLM
write(prog_start + 4, 0xFF, 0, 0)        # HALT

img.save("pxi_llm_test.png")

print("Booting pxOS → asking local LLM...")
cpu = PXICPU(img)
cpu.pc = prog_start
cpu.run()

# The response is written to the PXI image, not the frame.
# We will read it back from the image to verify.
response = cpu.read_string(100)
print(f"Response received: '{response}'")

# Save the frame for visual inspection
cpu.frame.save("pxi_llm_response.png")
print("Check pxi_llm_response.png")
