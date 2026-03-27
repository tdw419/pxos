class HeapAllocator {
    constructor(vram, startRow = 16, endRow = 31) {
        this.vram = vram;
        this.startRow = startRow;
        this.endRow = endRow;
        this.width = vram.width;
        this.heapStartAddr = 0;
        this.heapSize = (endRow - startRow + 1) * this.width;
        this.allocations = new Map(); // Addr -> Size
        this.freeList = [{ addr: 0, size: this.heapSize }];
    }

    alloc(size) {
        // First Fit
        for (let i = 0; i < this.freeList.length; i++) {
            const block = this.freeList[i];
            if (block.size >= size) {
                const addr = block.addr;

                // Update Free List
                if (block.size > size) {
                    block.addr += size;
                    block.size -= size;
                } else {
                    this.freeList.splice(i, 1);
                }

                this.allocations.set(addr, size);
                return addr;
            }
        }
        throw new Error("Heap Overflow");
    }

    free(addr) {
        if (!this.allocations.has(addr)) {
            console.warn(`Double Free or Invalid Address: ${addr}`);
            return;
        }
        const size = this.allocations.get(addr);
        this.allocations.delete(addr);

        // Add back to free list
        this.freeList.push({ addr, size });

        // Coalesce (Simple Sort & Merge)
        this.freeList.sort((a, b) => a.addr - b.addr);

        for (let i = 0; i < this.freeList.length - 1; i++) {
            const current = this.freeList[i];
            const next = this.freeList[i + 1];
            if (current.addr + current.size === next.addr) {
                current.size += next.size;
                this.freeList.splice(i + 1, 1);
                i--; // Retry merge with next
            }
        }

        console.log(`FREE: Address ${addr} (Size: ${size}) released.`);
    }

    getXY(addr) {
        const rowOffset = Math.floor(addr / this.width);
        const col = addr % this.width;
        const row = this.startRow + rowOffset;
        return { x: col, y: row };
    }

    writeByte(addr, val) {
        const { x, y } = this.getXY(addr);
        this.vram.drawPixel(x, y, val);
    }

    readByte(addr) {
        const { x, y } = this.getXY(addr);
        if (x >= 0 && x < this.width && y >= 0 && y < this.vram.height) {
            return this.vram.buffer[y * this.width + x];
        }
        return 0;
    }
}

module.exports = HeapAllocator;
