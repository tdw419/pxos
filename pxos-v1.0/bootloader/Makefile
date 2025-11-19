# pxOS Custom Bootloader Makefile

ASM = nasm
QEMU = qemu-system-x86_64

# Output files
BOOT_BIN = stage1_boot.bin
STAGE2_BIN = stage2.bin
DISK_IMG = pxos_boot.img

# Build flags
ASM_FLAGS = -f bin

.PHONY: all clean run test

all: $(DISK_IMG)

# Build boot sector (exactly 512 bytes)
$(BOOT_BIN): stage1_boot.asm
	$(ASM) $(ASM_FLAGS) -o $@ $<
	@echo "Boot sector size: `stat -c %s $(BOOT_BIN)` bytes (should be 512)"

# Build stage 2 (4KB = 8 sectors)
$(STAGE2_BIN): stage2.asm utils.asm gpu_detect.asm paging.asm
	$(ASM) $(ASM_FLAGS) -o $@ stage2.asm
	@echo "Stage 2 size: `stat -c %s $(STAGE2_BIN)` bytes (should be 4096)"

# Create bootable disk image
$(DISK_IMG): $(BOOT_BIN) $(STAGE2_BIN)
	@echo "Creating bootable disk image..."
	# Create 1.44MB floppy image
	dd if=/dev/zero of=$@ bs=512 count=2880 2>/dev/null
	# Write boot sector
	dd if=$(BOOT_BIN) of=$@ conv=notrunc 2>/dev/null
	# Write stage 2 (starting at sector 2)
	dd if=$(STAGE2_BIN) of=$@ bs=512 seek=1 conv=notrunc 2>/dev/null
	@echo "Disk image created: $(DISK_IMG)"

# Test in QEMU
test: $(DISK_IMG)
	$(QEMU) -drive format=raw,file=$(DISK_IMG) -m 512M

# Run with serial output
run: $(DISK_IMG)
	$(QEMU) -drive format=raw,file=$(DISK_IMG) -m 512M -serial stdio

# Run with debugging
debug: $(DISK_IMG)
	$(QEMU) -drive format=raw,file=$(DISK_IMG) -m 512M -serial stdio -d int,cpu_reset -D qemu_debug.log

# Run with VGA device
run-vga: $(DISK_IMG)
	$(QEMU) -drive format=raw,file=$(DISK_IMG) -m 512M -device VGA,vgamem_mb=64

clean:
	rm -f $(BOOT_BIN) $(STAGE2_BIN) $(DISK_IMG) qemu_debug.log

# Help
help:
	@echo "pxOS Bootloader Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build bootloader disk image"
	@echo "  test     - Build and run in QEMU"
	@echo "  run      - Run with serial output"
	@echo "  debug    - Run with QEMU debugging"
	@echo "  run-vga  - Run with VGA device"
	@echo "  clean    - Remove build artifacts"
	@echo "  help     - Show this message"
