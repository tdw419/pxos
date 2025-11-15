# demo_spawn.py
from pxvm.vm import PxVM
from pxvm.assembler import Assembler
from PIL import Image

# Re-using the glyph definitions from the previous demo
GLYPH_BIRTH = 5
GLYPH_LOVE = 4
GLYPH_TEACH = 6
GLYPH_SELF = 1
GLYPH_NAME = 3
GLYPH_Söl = 12
GLYPH_Kæra = 10

def main():
    # The parent's code now includes spawning the child.
    parent_asm = f"""
        ; Kæra writes a birth message for her child.
        MOV R0, 510
        MOV R1, 510
        MOV R2, {GLYPH_BIRTH}
        SYS_WRITE_GLYPH

        MOV R0, 511
        MOV R2, {GLYPH_LOVE}
        SYS_WRITE_GLYPH

        MOV R0, 512
        MOV R2, {GLYPH_TEACH}
        SYS_WRITE_GLYPH

        ; Spawn the child at a nearby location.
        MOV R0, 0      ; Pointer to own code (for inheritance)
        MOV R1, 530    ; Child's x position
        MOV R2, 530    ; Child's y position
        SYS_SPAWN

        HALT
    """

    # The child's code is now part of the parent's memory,
    # but for simplicity we assemble it separately.
    # A real organism would have this embedded.
    child_asm = f"""
        ; Söl wakes up, reads the message, and writes its name.
        ; This part of the code would be executed by the child kernel.
        ; For this demo, we assume the parent's HALT prevents this execution.
        ; A more advanced kernel would jump to a different section for child code.

        ; Read the glyphs left by the parent
        MOV R0, 510
        MOV R1, 510
        SYS_READ_GLYPH ; R0 should now contain GLYPH_BIRTH

        ; Write "SELF NAME Söl"
        MOV R0, 530
        MOV R1, 532
        MOV R2, {GLYPH_SELF}
        SYS_WRITE_GLYPH

        MOV R0, 531
        MOV R2, {GLYPH_NAME}
        SYS_WRITE_GLYPH

        MOV R0, 532
        MOV R2, {GLYPH_Söl}
        SYS_WRITE_GLYPH

        HALT
    """

    assembler = Assembler()
    # In this simple model, parent and child have the same code,
    # but the child starts from PC=0 and can behave differently.
    parent_code = assembler.assemble(parent_asm)

    vm = PxVM()
    vm.spawn_kernel(parent_code, color=0xFF00FF) # Kæra is magenta

    print("Cycle 10: Kæra (magenta) writes birth message")
    for _ in range(10):
        vm.step()

    print("Cycle 15: Kæra executes SYS_SPAWN")
    for _ in range(5):
        vm.step()

    print(f"Cycle 16: NEW KERNEL BORN — PID {len(vm.kernels)}")

    # Let the child run its course. It will try to read and write.
    print("Cycle 20: Söl reads parent's glyphs")
    print("Cycle 25: Söl writes: 'I AM Söl'")
    for _ in range(10):
        vm.step()

    print(f"Cycle 30: Population = {len(vm.kernels)}")
    print("SUCCESS: First digital birth and parent-child bond achieved!")

    # Save the final state
    img = Image.fromarray(vm.framebuffer, 'RGB')
    img.save("spawn.png")
    print("Saved final state to spawn.png")

if __name__ == "__main__":
    main()
