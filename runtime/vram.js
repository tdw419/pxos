class VRAMMemoryManager {
    constructor(width = 32, height = 32) {
        this.width = width;
        this.height = height;
        this.buffer = new Int32Array(width * height).fill(0);
        this.lines = []; // Store lines as {x1, y1, x2, y2, color}
    }

    clear() {
        this.buffer.fill(0);
        this.lines = [];
    }

    drawPixel(x, y, color) {
        if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
            this.buffer[y * this.width + x] = color;
        }
    }

    fillRect(x, y, w, h, color) {
        let x1 = Math.max(0, x);
        let y1 = Math.max(0, y);
        let x2 = Math.min(this.width, x + w);
        let y2 = Math.min(this.height, y + h);

        for (let py = y1; py < y2; py++) {
            for (let px = x1; px < x2; px++) {
                this.buffer[py * this.width + px] = color;
            }
        }
    }

    drawLine(x1, y1, x2, y2, color) {
        this.lines.push({ x1, y1, x2, y2, color });
    }

    renderAscii() {
        // Render buffer to ASCII grid
        const chars = ['.', '#', '@', '%', '&', '*', '+', '=', '-', ':'];
        const grid = Array.from({ length: this.height }, () => Array(this.width).fill('.'));

        // Buffer
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const val = this.buffer[y * this.width + x];
                if (val !== 0) {
                    grid[y][x] = chars[val % 10] || '?';
                }
            }
        }

        // Overlay Lines
        for (const line of this.lines) {
            this._plotLine(grid, line.x1, line.y1, line.x2, line.y2, '|');
        }

        return grid.map(row => row.join('')).join('\n');
    }

    _plotLine(grid, x0, y0, x1, y1, char) {
        let dx = Math.abs(x1 - x0);
        let dy = Math.abs(y1 - y0);
        let sx = (x0 < x1) ? 1 : -1;
        let sy = (y0 < y1) ? 1 : -1;
        let err = dx - dy;

        while (true) {
            if (x0 >= 0 && x0 < this.width && y0 >= 0 && y0 < this.height) {
                // Visualize line only if empty or just overwrite
                if (grid[y0][x0] === '.') {
                    grid[y0][x0] = '*';
                }
            }

            if (x0 === x1 && y0 === y1) break;
            let e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y0 += sy;
            }
        }
    }
}

module.exports = VRAMMemoryManager;
