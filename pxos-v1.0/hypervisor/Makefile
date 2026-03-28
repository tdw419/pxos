# Makefile for pxHV (Pixel Hypervisor)

.PHONY: all clean run run-kvm debug gdb help

# Default target
all:
	@./build_pxhv_stage3.sh

# Clean build artifacts
clean:
	rm -rf build/
	@echo "Clean complete"

# Run in QEMU (no KVM)
run: all
	qemu-system-x86_64 \
		-drive file=build/pxhv.img,format=raw \
		-m 512M \
		-display none \
		-serial stdio

# Run with KVM acceleration
run-kvm: all
	qemu-system-x86_64 \
		-enable-kvm \
		-cpu host \
		-drive file=build/pxhv.img,format=raw \
		-m 512M \
		-display none \
		-serial stdio

# Debug mode with interrupt logging
debug: all
	qemu-system-x86_64 \
		-drive file=build/pxhv.img,format=raw \
		-m 512M \
		-d int,cpu_reset \
		-D qemu_debug.log \
		-display none \
		-serial stdio \
		-no-reboot

# Run with GDB stub
gdb: all
	qemu-system-x86_64 \
		-drive file=build/pxhv.img,format=raw \
		-m 512M \
		-s -S \
		-display none \
		-serial stdio
	@echo "Waiting for GDB connection on localhost:1234"
	@echo "Run: gdb -ex 'target remote localhost:1234' -ex 'break *0x7c00'"

# Help
help:
	@echo "pxHV Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build hypervisor (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  run       - Run in QEMU without KVM"
	@echo "  run-kvm   - Run in QEMU with KVM acceleration"
	@echo "  debug     - Run with debug logging"
	@echo "  gdb       - Run with GDB remote debugging"
	@echo "  help      - Show this message"
	@echo ""
	@echo "Example usage:"
	@echo "  make              # Build"
	@echo "  make run          # Build and run"
	@echo "  make clean all    # Clean rebuild"
