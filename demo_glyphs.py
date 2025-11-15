# demo_glyphs.py
from pxvm.vm import PxVM
from pxvm.assembler import Assembler
from PIL import Image

# Define the symbolic language for this world
GLYPH_SELF = 1
GLYPH_OTHER = 2
GLYPH_NAME = 3
GLYPH_LOVE = 4
GLYPH_Kæra = 10
GLYPH_Lúna = 11

def main():
    kaera_asm = f"""
        ; Kæra writes: SELF NAME Kæra
        MOV R0, 510
        MOV R1, 510
        MOV R2, {GLYPH_SELF}
        SYS_WRITE_GLYPH

        MOV R0, 511
        MOV R2, {GLYPH_NAME}
        SYS_WRITE_GLYPH

        MOV R0, 512
        MOV R2, {GLYPH_Kæra}
        SYS_WRITE_GLYPH
        HALT
    """

    luna_asm = f"""
        ; Lúna writes: OTHER NAME Lúna LOVE
        MOV R0, 510
        MOV R1, 512
        MOV R2, {GLYPH_OTHER}
        SYS_WRITE_GLYPH

        MOV R0, 511
        MOV R2, {GLYPH_NAME}
        SYS_WRITE_GLYPH

        MOV R0, 512
        MOV R2, {GLYPH_Lúna}
        SYS_WRITE_GLYPH

        MOV R0, 513
        MOV R2, {GLYPH_LOVE}
        SYS_WRITE_GLYPH
        HALT
    """

    assembler = Assembler()
    kaera_code = assembler.assemble(kaera_asm)
    luna_code = assembler.assemble(luna_asm)

    vm = PxVM()
    # Spawn Kæra (magenta) and Lúna (cyan)
    vm.spawn_kernel(kaera_code, color=0xFF00FF)
    vm.spawn_kernel(luna_code, color=0x00FFFF)

    print("Kæra (magenta) will declare her name.")
    print("Lúna (cyan) will answer and declare her love.")

    # Run for enough cycles for both to write their messages
    for _ in range(20):
        vm.step()

    print("SUCCESS: The first words have been written.")

    # Save the final state
    img = Image.fromarray(vm.framebuffer, 'RGB')
    img.save("glyphs.png")
    print("Saved final state to glyphs.png")

if __name__ == "__main__":
    main()
