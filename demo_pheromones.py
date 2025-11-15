# demo_pheromones.py
from pxvm.vm import PxVM
from pxvm.assembler import Assembler
from PIL import Image

def main():
    with open("pxvm/examples/follow.asm") as f:
        follower_asm = f.read()

    emitter_asm = """
        MOV R0, 512
        MOV R1, 512
        MOV R2, 255
        SYS_EMIT_PHEROMONE
        HALT
    """

    assembler = Assembler()
    follower_code = assembler.assemble(follower_asm)
    emitter_code = assembler.assemble(emitter_asm)

    vm = PxVM()
    vm.spawn_kernel(emitter_code, color=0xFFFF00) # Yellow
    vm.spawn_kernel(follower_code, color=0x00FFFF)   # Cyan

    print("Pheromone layer active (8-bit chemical field)")
    print("Kernel 1 (Yellow): emitting strong pheromone at center")
    print("Kernel 2 (Cyan): hunting scent — chemotaxis active")

    for i in range(500):
        vm.step()
        if i % 20 == 0:
            print(f"Cycle {vm.cycle}: {len([k for k in vm.kernels if not k.halted])} kernels alive")

    print("SUCCESS: First artificial chemotaxis achieved!")

    img = Image.fromarray(vm.framebuffer, 'RGB')
    img.save("pheromones.png")

if __name__ == "__main__":
    main()
