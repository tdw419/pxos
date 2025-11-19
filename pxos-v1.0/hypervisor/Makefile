.PHONY: all clean run

all: pxhv.img

pxhv.img: pxhv_boot.bin pxhv_stage2.bin
	cat pxhv_boot.bin pxhv_stage2.bin > pxhv.img

pxhv_boot.bin: pxhv_boot.asm
	nasm -f bin -o pxhv_boot.bin pxhv_boot.asm

pxhv_stage2.bin: pxhv_stage2.asm pxhv_helpers.asm pxhv_helpers_64.asm
	nasm -f bin -o pxhv_stage2.bin pxhv_stage2.asm

run: pxhv.img
	qemu-system-x86_64 -drive file=pxhv.img,format=raw

clean:
	rm -f *.bin *.img
