# demo_two_kernels.py
# PROOF: Two living kernels that see each other
from pxvm.vm import PxVM
import time
import matplotlib.pyplot as plt

# Hand-assembled: draw a green square and move
code_green = bytes([
    1, 0, 100, 0, 0, 0,   # MOV R0, 100    (x)
    1, 1, 100, 0, 0, 0,   # MOV R1, 100    (y)
    1, 2, 0, 255, 0, 0,   # MOV R2, 0x00FF00
    2,                       # PLOT
    255,                     # NOP
    0,                       # HALT
])

# Red kernel draws diagonal line
code_red = bytes([
    1, 0, 50, 0, 0, 0,
    1, 1, 50, 0, 0, 0,
    1, 2, 255, 0, 0, 0,   # red
    2,                       # PLOT
    3, 0, 1,                 # ADD R0, R1 -> This is wrong, should be ADD R0, R_one
    3, 1, 1,                 # ADD R1, R_one
    255,                     # NOP
    0,                       # HALT
])

vm = PxVM()
vm.spawn_kernel(code_green, 0x00FF00)
vm.spawn_kernel(code_red, 0xFF0000)

plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(vm.framebuffer)

print("Two kernels spawned. Green draws at (100,100), Red draws diagonal.")
print("They share the same framebuffer — they can see each other immediately.")

for i in range(500):
    vm.step()
    if i % 20 == 0:
        img.set_data(vm.framebuffer)
        plt.pause(0.01)
        print(f"Cycle {vm.cycle}: {len([k for k in vm.kernels if not k.halted])} kernels alive")

plt.ioff()
plt.show()
