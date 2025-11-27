import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Terminal, Power } from 'lucide-react';
import StatusPanel from './components/StatusPanel';
import VRAMVisualizer from './components/VRAMVisualizer';
import { sendMessageToKernel, initializeChat } from './services/geminiService';
import { Message, RegisterState, KernelStatus, PixelState, Blueprint } from './types';
import { INITIAL_REGISTERS } from './constants';
import { GpuBridge, GPUProgram, GPUDetails } from './services/gpuBridge';
import DBExplorer from './components/DBExplorer';

const VRAM_WIDTH = 32;
const VRAM_HEIGHT = 32;
const PIXEL_COUNT = VRAM_WIDTH * VRAM_HEIGHT;
const PIXEL_STATE_SIZE = 12; // registers + neighbors + blueprint refs
const MAX_BLUEPRINTS = 16;
const MAX_RULE_DATA = 8;
type InstructionOp = 'COUNT_NEIGHBORS' | 'CMP_EQ' | 'MOV' | 'JE' | 'JMP' | 'NOP' | 'HALT';
type Instruction = { op: InstructionOp; dst?: number; imm?: number; target?: number; a?: number; };

const HEAP_START = 0x2000;
const HEAP_SIZE = 0x1000; // 4KB heap
const BYTES_PER_HEAP_PIXEL = 16; // visual resolution for heap
const HEAP_ROW_BASE = 8; // start drawing heap at row 8 (pixel index offset = 8 * 32)

interface HeapBlock {
  address: number;
  size: number;
  allocated: boolean;
}

interface CPPMMetrics {
  executionPressure: number; // 0-1
  pixelsActive: number;
  heapFragmentation: number; // 0-1
  allocationBudget: number;  // allocations allowed this frame
}

interface CPPMMetrics {
  executionPressure: number; // 0-1
  pixelsActive: number;
  heapFragmentation: number; // 0-1
  allocationBudget: number;  // allocations allowed this frame
}

const createPixelStateBuffer = () => new Uint8Array(PIXEL_COUNT * PIXEL_STATE_SIZE);
const wrapCoordinate = (value: number, max: number) => {
  const mod = value % max;
  return mod < 0 ? mod + max : mod;
};
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * clamp01(t);
const colorFromRGB = (r: number, g: number, b: number) =>
  ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
const hashNoise = (x: number, y: number) => {
  let v = (x * 73856093) ^ (y * 19349663);
  v = (v ^ (v >> 13)) * 1274126177;
  return (v >>> 24) & 0xFF;
};
const colorFromRGBA = (r: number, g: number, b: number, a: number = 255) =>
  ((a & 0xFF) << 24) | ((b & 0xFF) << 16) | ((g & 0xFF) << 8) | (r & 0xFF);
const getPixelStateFromBuffer = (buffer: Uint8Array, x: number, y: number): PixelState => {
  const px = wrapCoordinate(x, VRAM_WIDTH);
  const py = wrapCoordinate(y, VRAM_HEIGHT);
  const offset = (py * VRAM_WIDTH + px) * PIXEL_STATE_SIZE;
  return {
    R0: buffer[offset + 0],
    R1: buffer[offset + 1],
    R2: buffer[offset + 2],
    R3: buffer[offset + 3],
    PC: buffer[offset + 4],
    FLAGS: buffer[offset + 5],
    NORTH: buffer[offset + 6],
    SOUTH: buffer[offset + 7],
    EAST: buffer[offset + 8],
    WEST: buffer[offset + 9],
    BLUEPRINT_ID: buffer[offset + 10],
    BLUEPRINT_PARAM: buffer[offset + 11],
  };
};
const renderPixelState = (state: PixelState) =>
  colorFromRGB(state.R0, state.R1, state.FLAGS & 0x01 ? 255 : 0);
const renderPixelStateBuffer = (buffer: Uint8Array) => {
  const view = new Uint32Array(PIXEL_COUNT);
  for (let y = 0; y < VRAM_HEIGHT; y++) {
    for (let x = 0; x < VRAM_WIDTH; x++) {
      view[y * VRAM_WIDTH + x] = renderPixelState(getPixelStateFromBuffer(buffer, x, y));
    }
  }
  return view;
};
const blueprintPatternFromToken = (token?: string) => {
  if (!token) return NaN;
  const normalized = token.toLowerCase();
  if (normalized === 'solid') return 0;
  if (normalized === 'gradient') return 1;
  if (normalized === 'checker') return 2;
  if (normalized === 'fractal') return 3;
  if (normalized === 'noise') return 4;
  if (normalized === 'animation') return 5;
  const numeric = parseInt(token, 10);
  return Number.isFinite(numeric) ? numeric : NaN;
};
const expandBlueprintColor = (x: number, y: number, bp: Blueprint, localParam: number = 0) => {
  const d = bp.ruleData;
  switch (bp.patternType) {
    case 0: {
      const r = d[0] ?? 0, g = d[1] ?? 0, b = d[2] ?? 0;
      return colorFromRGB(r, g, b);
    }
    case 1: {
      const r1 = d[0] ?? 0, g1 = d[1] ?? 0, b1 = d[2] ?? 0;
      const r2 = d[4] ?? 255, g2 = d[5] ?? 255, b2 = d[6] ?? 255;
      const direction = d[8] ?? 0;
      let t = 0;
      if (direction === 1) t = x / (VRAM_WIDTH - 1);
      else if (direction === 2) t = y / (VRAM_HEIGHT - 1);
      else t = (x + y) / (VRAM_WIDTH + VRAM_HEIGHT - 2);
      return colorFromRGB(
        Math.floor(lerp(r1, r2, t)),
        Math.floor(lerp(g1, g2, t)),
        Math.floor(lerp(b1, b2, t))
      );
    }
    case 2: {
      const cellSize = Math.max(1, d[0] ?? 4);
      const checker = ((Math.floor(x / cellSize) + Math.floor(y / cellSize)) % 2) === 0;
      const c1 = [d[1] ?? 0, d[2] ?? 0, d[3] ?? 0];
      const c2 = [d[4] ?? 255, d[5] ?? 255, d[6] ?? 255];
      const [r, g, b] = checker ? c1 : c2;
      return colorFromRGB(r, g, b);
    }
    case 3: {
      const maxIter = d[0] ?? 16;
      const zoom = (d[1] ?? 10) / 10;
      const cx = (d[2] ?? 0) / 10;
      const cy = (d[3] ?? 0) / 10;
      const xf = ((x / VRAM_WIDTH) - 0.5) * 3.5 / Math.max(zoom, 0.01) + cx;
      const yf = ((y / VRAM_HEIGHT) - 0.5) * 2.0 / Math.max(zoom, 0.01) + cy;
      let zr = 0, zi = 0, iter = 0;
      while (zr * zr + zi * zi <= 4 && iter < maxIter) {
        const tmp = zr * zr - zi * zi + xf;
        zi = 2 * zr * zi + yf;
        zr = tmp;
        iter++;
      }
      const shade = Math.floor(255 * (iter / maxIter));
      return colorFromRGB(shade, shade, shade);
    }
    case 4: {
      const scale = (d[0] ?? 10) / 10;
      const n = hashNoise(Math.floor(x * scale) + localParam, Math.floor(y * scale));
      return colorFromRGB(n, n, n);
    }
    case 5: {
      const frames = Math.max(1, d[0] ?? 8);
      const speed = Math.max(1, d[1] ?? 1);
      const phase = ((localParam + speed * performance.now()) / 1000) % frames;
      const t = (Math.sin((phase / frames) * Math.PI * 2) + 1) / 2;
      const r = Math.floor(lerp(d[2] ?? 0, d[3] ?? 255, t));
      const g = Math.floor(lerp(d[4] ?? 0, d[5] ?? 255, t));
      const b = Math.floor(lerp(d[6] ?? 0, d[7] ?? 255, t));
      return colorFromRGB(r, g, b);
    }
    default:
      return colorFromRGB(0, 0, 0);
  }
};

const initHeap = () => {
  heapBlocksRef.current = [{ address: HEAP_START, size: HEAP_SIZE, allocated: false }];
};

const heapAlloc = (size: number): number | null => {
  const blocks = heapBlocksRef.current;
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    if (!block.allocated && block.size >= size) {
      if (block.size > size) {
        blocks.splice(i + 1, 0, {
          address: block.address + size,
          size: block.size - size,
          allocated: false,
        });
      }
      block.size = size;
      block.allocated = true;
      return block.address;
    }
  }
  return null;
};

const addressToHeapPixel = (addr: number) => {
  const offset = addr - HEAP_START;
  if (offset < 0 || offset >= HEAP_SIZE) return -1;
  const pixelOffset = Math.floor(offset / BYTES_PER_HEAP_PIXEL);
  const idx = HEAP_ROW_BASE * VRAM_WIDTH + pixelOffset;
  return idx < PIXEL_COUNT ? idx : -1;
};

const setPixelSafe = (buf: Uint32Array, idx: number, color: number) => {
  if (idx >= 0 && idx < buf.length) buf[idx] = color;
};

const drawLine = (buf: Uint32Array, fromIdx: number, toIdx: number, color: number) => {
  const x0 = fromIdx % VRAM_WIDTH, y0 = Math.floor(fromIdx / VRAM_WIDTH);
  const x1 = toIdx % VRAM_WIDTH, y1 = Math.floor(toIdx / VRAM_WIDTH);
  let dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0);
  let sx = x0 < x1 ? 1 : -1;
  let sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  let x = x0, y = y0;
  while (true) {
    setPixelSafe(buf, y * VRAM_WIDTH + x, color);
    if (x === x1 && y === y1) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x += sx; }
    if (e2 <= dx) { err += dx; y += sy; }
  }
};

const calculateFragmentation = () => {
  const blocks = heapBlocksRef.current;
  const freeBlocks = blocks.filter(b => !b.allocated);
  if (!freeBlocks.length) return 0;
  const totalFree = freeBlocks.reduce((s, b) => s + b.size, 0);
  const largestFree = Math.max(...freeBlocks.map(b => b.size));
  return totalFree ? 1 - largestFree / totalFree : 0;
};

const initCPPM = () => {
  cppmRef.current = {
    executionPressure: 0,
    pixelsActive: 0,
    heapFragmentation: 0,
    allocationBudget: 100,
  };
};

const resetCPPMBudget = () => {
  const s = cppmRef.current;
  s.allocationBudget = Math.max(0, Math.floor(100 * (1 - s.executionPressure * 0.5)));
  s.executionPressure *= 0.9;
};

const heapAllocWithCPPM = (size: number): number | null => {
  const s = cppmRef.current;
  if (s.allocationBudget <= 0) return null;
  s.allocationBudget -= 1;
  s.executionPressure = Math.min(1, s.executionPressure + size / HEAP_SIZE);
  const addr = heapAlloc(size);
  if (addr !== null) {
    s.heapFragmentation = calculateFragmentation();
  }
  return addr;
};

const App: React.FC = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    { role: 'system', content: 'PixelOS Kernel v0.9.1 Initialized. Neural Link Established.', timestamp: Date.now() }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showTyping, setShowTyping] = useState(false);
  
  // Simulated Kernel State
  const [registers, setRegisters] = useState<RegisterState>(INITIAL_REGISTERS);
  const [status, setStatus] = useState<KernelStatus>({
    activeProgram: 'IDLE',
    cycleCount: 0,
    halted: false,
    vramDirty: false
  });
  const [gpuStatus, setGpuStatus] = useState<'checking' | 'ready' | 'fallback'>('checking');
  const [gpuMessage, setGpuMessage] = useState<string>('Detecting WebGPU...');
  const [gpuDetails, setGpuDetails] = useState<GPUDetails | null>(null);
  const [gpuMetrics, setGpuMetrics] = useState<{ lastRunMs: number | null; totalRuns: number }>({
    lastRunMs: null,
    totalRuns: 0
  });
  const [autoLoopEnabled, setAutoLoopEnabled] = useState(false);
  const [autoLoopCount, setAutoLoopCount] = useState(2);
  const [view, setView] = useState<'terminal' | 'db'>('terminal');
  const [pixelStates, setPixelStates] = useState<Uint8Array>(() => createPixelStateBuffer());

  // Simulated VRAM (Uint32Array for performance)
  const [vram, setVram] = useState<Uint32Array>(new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT).fill(0x000000));

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const vramRef = useRef<Uint32Array>(vram);
  const gpuBridgeRef = useRef<GpuBridge | null>(null);
  const pixelStatesRef = useRef<Uint8Array>(pixelStates);
  const blueprintRegistryRef = useRef<Map<number, Blueprint>>(new Map());
  const heapBlocksRef = useRef<HeapBlock[]>([{ address: HEAP_START, size: HEAP_SIZE, allocated: false }]);
  const cppmRef = useRef<CPPMetrics>({
    executionPressure: 0,
    pixelsActive: 0,
    heapFragmentation: 0,
    allocationBudget: 100,
  });
  const adaptiveNodesRef = useRef<number[]>([]);
  const targetNodesRef = useRef<number>(100);
  const rafRef = useRef<number | null>(null);
  const adaptiveRunningRef = useRef<boolean>(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    vramRef.current = vram;
  }, [vram]);

  useEffect(() => {
    pixelStatesRef.current = pixelStates;
  }, [pixelStates]);

  // Initial greeting
  useEffect(() => {
    initializeChat().catch(err => {
        setMessages(prev => [...prev, { 
            role: 'system', 
            content: `ERR: Neural Core Connection Failed. ${err.message}`, 
            timestamp: Date.now() 
        }]);
    });
  }, []);

  // Initialize WebGPU pipeline when available
  useEffect(() => {
    GpuBridge.init()
      .then(bridge => {
        gpuBridgeRef.current = bridge;
        const info = bridge.getInfo();
        setGpuDetails(info);
        setGpuStatus('ready');
        setGpuMessage(`WebGPU pipeline online${info?.adapterLabel ? ` (${info.adapterLabel})` : ''}`);
      })
      .catch(err => {
        setGpuStatus('fallback');
        setGpuMessage(err.message);
      });
  }, []);

  const renderHeap = (buf: Uint32Array) => {
    // Layer heap blocks
    for (const block of heapBlocksRef.current) {
      const startIdx = addressToHeapPixel(block.address);
      const pixels = Math.max(1, Math.floor(block.size / BYTES_PER_HEAP_PIXEL));
      for (let i = 0; i < pixels; i++) {
        const idx = startIdx + i;
        setPixelSafe(buf, idx, block.allocated ? colorFromRGBA(68, 136, 255) : colorFromRGBA(34, 34, 34));
      }
    }
  };

  const renderPointerTethers = (buf: Uint32Array, nodes: number[]) => {
    for (let i = 0; i < nodes.length - 1; i++) {
      const fromPixel = addressToHeapPixel(nodes[i]);
      const toPixel = addressToHeapPixel(nodes[i + 1]);
      if (fromPixel >= 0 && toPixel >= 0) {
        const brightness = Math.max(32, Math.floor(255 * (1 - cppmRef.current.executionPressure)));
        drawLine(buf, fromPixel, toPixel, colorFromRGBA(255, brightness, 0));
      }
    }
  };

  const renderCPPMOverlay = (buf: Uint32Array) => {
    const s = cppmRef.current;
    const pressureColor = Math.floor(s.executionPressure * 255);
    // Top 4 rows show pressure in red channel
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < VRAM_WIDTH; x++) {
        const idx = y * VRAM_WIDTH + x;
        const base = buf[idx] & 0x00FFFFFF;
        const r = pressureColor;
        buf[idx] = (0xFF << 24) | (base & 0x00FFFF00) | r;
      }
    }
    // Bottom row shows allocation budget (green bar)
    const budgetPixels = Math.min(VRAM_WIDTH, Math.floor((s.allocationBudget / 100) * VRAM_WIDTH));
    const y = VRAM_HEIGHT - 1;
    for (let x = 0; x < VRAM_WIDTH; x++) {
      const idx = y * VRAM_WIDTH + x;
      if (x < budgetPixels) {
        buf[idx] = colorFromRGBA(0, 255, 0);
      } else {
        buf[idx] = colorFromRGBA(68, 68, 68);
      }
    }
  };

  const assignBlueprintToPixels = (blueprintId: number, localParam: number = 0) => {
    const buffer = createPixelStateBuffer();
    for (let i = 0; i < PIXEL_COUNT; i++) {
      buffer[i * PIXEL_STATE_SIZE + 10] = blueprintId & 0xFF;
      buffer[i * PIXEL_STATE_SIZE + 11] = localParam & 0xFF;
    }
    pixelStatesRef.current = buffer;
    setPixelStates(buffer);
  };

  const createBlueprint = (patternToken: string, params: number[]) => {
    const patternType = blueprintPatternFromToken(patternToken);
    if (!Number.isFinite(patternType)) {
      setStatus(s => ({ ...s, activeProgram: 'BLUEPRINT_INVALID', cycleCount: s.cycleCount }));
      return null;
    }
    if (blueprintRegistryRef.current.size >= MAX_BLUEPRINTS) {
      setStatus(s => ({ ...s, activeProgram: 'BLUEPRINT_FULL', cycleCount: s.cycleCount }));
      return null;
    }
    const id = blueprintRegistryRef.current.size;
    const ruleData = params.slice(0, MAX_RULE_DATA);
    const bp: Blueprint = {
      id,
      patternType,
      ruleData,
      compressionRatio: (PIXEL_COUNT * 4) / Math.max(ruleData.length || 1, 1),
      lastExpanded: Date.now(),
    };
    blueprintRegistryRef.current.set(id, bp);
    return bp;
  };

  const expandBlueprintToVRAM = (bp: Blueprint) => {
    const frame = new Uint32Array(PIXEL_COUNT);
    for (let y = 0; y < VRAM_HEIGHT; y++) {
      for (let x = 0; x < VRAM_WIDTH; x++) {
        const idx = y * VRAM_WIDTH + x;
        const localParam = pixelStatesRef.current[idx * PIXEL_STATE_SIZE + 11] ?? 0;
        frame[idx] = expandBlueprintColor(x, y, bp, localParam);
      }
    }
    vramRef.current = frame;
    setVram(frame);
    bp.lastExpanded = Date.now();
    setStatus(s => ({ ...s, activeProgram: `BP_EXPAND_${bp.id}`, cycleCount: s.cycleCount + PIXEL_COUNT }));
  };

  const handleBlueprintCommand = (raw: string) => {
    const parts = raw.trim().split(/\s+/);
    const cmd = parts[0]?.toLowerCase();
    const timestamp = Date.now();

    const appendModel = (content: string) => setMessages(prev => [...prev, { role: 'model', content, timestamp: Date.now() }]);
    const appendUser = () => setMessages(prev => [...prev, { role: 'user', content: raw, timestamp }]);

    if (cmd === 'create_blueprint') {
      appendUser();
      const pattern = parts[1];
      const params = parts.slice(2).map(Number).filter(n => Number.isFinite(n));
      const bp = createBlueprint(pattern, params);
      if (bp) {
        assignBlueprintToPixels(bp.id);
        const preview = new Uint32Array(PIXEL_COUNT);
        for (let i = 0; i < PIXEL_COUNT; i++) {
          preview[i] = (i % 16 < 8) ? colorFromRGB(0, 68, 255) : colorFromRGB(68, 0, 68);
        }
        setVram(preview);
        vramRef.current = preview;
        appendModel(`[BP:${bp.id}] pattern=${bp.patternType} params=${bp.ruleData.join(',')} compression=${bp.compressionRatio?.toFixed(1)}x`);
        setStatus(s => ({ ...s, activeProgram: `BP_CREATE_${bp.id}`, cycleCount: s.cycleCount + 1 }));
      } else {
        appendModel('[BP_ERROR] creation_failed');
      }
      setIsProcessing(false);
      setShowTyping(false);
      return true;
    }

    if (cmd === 'expand_blueprint') {
      appendUser();
      const id = Number(parts[1]);
      if (!Number.isFinite(id)) {
        appendModel('[BP_ERROR] invalid_id');
        setIsProcessing(false);
        setShowTyping(false);
        return true;
      }
      const bp = blueprintRegistryRef.current.get(id);
      if (!bp) {
        appendModel(`[BP_ERROR] not_found_${id}`);
        setIsProcessing(false);
        setShowTyping(false);
        return true;
      }
      assignBlueprintToPixels(id);
      expandBlueprintToVRAM(bp);
      appendModel(`[BP_EXPANDED ${id}] compression=${bp.compressionRatio?.toFixed(1)}x`);
      setIsProcessing(false);
      setShowTyping(false);
      return true;
    }

    if (cmd === 'list_blueprints') {
      appendUser();
      const list = Array.from(blueprintRegistryRef.current.values())
        .map(bp => `#${bp.id}: type=${bp.patternType} params=${bp.ruleData.join(',')} ratio=${bp.compressionRatio?.toFixed(1)}x`);
      appendModel(list.length ? list.join('\n') : '[NO_BLUEPRINTS]');
      setStatus(s => ({ ...s, activeProgram: 'BP_LIST', cycleCount: s.cycleCount }));
      setIsProcessing(false);
      setShowTyping(false);
      return true;
    }

    return false;
  };

  // Simulate kernel visuals (prefers GPU, falls back to CPU)
  const simulateVisuals = useCallback(async (cmd: string) => {
    const lowerCmd = cmd.toLowerCase();
    const runGpu = async (program: GPUProgram) => {
      if (!gpuBridgeRef.current || gpuStatus !== 'ready') return null;
      try {
        const start = performance.now();
        const result = await gpuBridgeRef.current.run({
          program,
          input: vramRef.current,
          width: VRAM_WIDTH,
          height: VRAM_HEIGHT
        });
        const duration = performance.now() - start;
        setGpuMetrics(prev => ({
          lastRunMs: duration,
          totalRuns: prev.totalRuns + 1
        }));
        return result;
      } catch (err: any) {
        setGpuStatus('fallback');
        setGpuMessage(err?.message || 'WebGPU pipeline error, using CPU.');
        return null;
      }
    };
    
    let newVram: Uint32Array | null = null;

    if (lowerCmd.includes('pointer_test') || lowerCmd.includes('linkedlist')) {
      initHeap();
      const nodes: number[] = [];
      const NODE_SIZE = 8;
      const allocations = 5;
      for (let i = 0; i < allocations; i++) {
        const addr = heapAlloc(NODE_SIZE);
        if (addr === null) break;
        nodes.push(addr);
      }
      const vramBuf = new Uint32Array(PIXEL_COUNT).fill(0x000000);
      renderHeap(vramBuf);
      renderPointerTethers(vramBuf, nodes);
      newVram = vramBuf;
      setStatus(s => ({ ...s, activeProgram: `HEAP_LL_${nodes.length}`, cycleCount: s.cycleCount + nodes.length }));
      setRegisters({
        r0: nodes[0] ? `0x${nodes[0].toString(16)}` : '0x0000',
        r1: `0x${nodes.length.toString(16)}`,
        r2: '0x0000',
        r3: '0xFFFF',
        pc: '0xBEEF',
        flags: nodes.length ? '0001' : '0000'
      });
      setMessages(prev => [...prev, { role: 'model', content: `[POINTER_TEST] nodes=${nodes.length} heap_used=${nodes.length * NODE_SIZE}B`, timestamp: Date.now() }]);
      setVram(vramBuf);
      return;
    }

    if (lowerCmd.includes('cppm_linkedlist') || lowerCmd.includes('adaptive_heap')) {
      if (adaptiveRunningRef.current) {
        setMessages(prev => [...prev, { role: 'model', content: '[CPPM] demo already running' }]);
        return;
      }
      adaptiveRunningRef.current = true;
      initHeap();
      initCPPM();
      adaptiveNodesRef.current = [];
      targetNodesRef.current = 100;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);

      const NODE_SIZE = 8;

      const step = () => {
        resetCPPMBudget();
        const s = cppmRef.current;

        // Adaptive target based on pressure
        if (s.executionPressure > 0.8) {
          targetNodesRef.current = Math.max(10, targetNodesRef.current - 10);
        } else if (s.executionPressure < 0.3) {
          targetNodesRef.current = Math.min(200, targetNodesRef.current + 10);
        }

        // Allocate within budget
        while (s.allocationBudget > 0 && adaptiveNodesRef.current.length < targetNodesRef.current) {
          const addr = heapAllocWithCPPM(NODE_SIZE);
          if (addr === null) break;
          adaptiveNodesRef.current.push(addr);
        }

        // Render overlays
        const vramBuf = new Uint32Array(PIXEL_COUNT).fill(0x000000);
        renderHeap(vramBuf);
        renderPointerTethers(vramBuf, adaptiveNodesRef.current);
        renderCPPMOverlay(vramBuf);
        setVram(vramBuf);
        vramRef.current = vramBuf;

        // Update status
        setStatus(sPrev => ({
          ...sPrev,
          activeProgram: 'CPPM_HEAP',
          cycleCount: sPrev.cycleCount + adaptiveNodesRef.current.length,
        }));
        setRegisters({
          r0: adaptiveNodesRef.current[0] ? `0x${adaptiveNodesRef.current[0].toString(16)}` : '0x0000',
          r1: `0x${adaptiveNodesRef.current.length.toString(16)}`,
          r2: '0x0000',
          r3: '0xFFFF',
          pc: '0xCAFE',
          flags: adaptiveNodesRef.current.length ? '0001' : '0000'
        });

        if (adaptiveNodesRef.current.length < targetNodesRef.current) {
          rafRef.current = requestAnimationFrame(step);
        } else {
          adaptiveRunningRef.current = false;
          setMessages(prev => [...prev, { role: 'model', content: `[CPPM_COMPLETE] nodes=${adaptiveNodesRef.current.length} pressure=${(s.executionPressure * 100).toFixed(1)}% frag=${(s.heapFragmentation * 100).toFixed(1)}%`, timestamp: Date.now() }]);
        }
      };

      step();
      return;
    }

    if (lowerCmd.includes('cppm_metrics') || lowerCmd.includes('cppm show')) {
      const s = cppmRef.current;
      const metrics = [
        '=== CPPM METRICS ===',
        `Pressure: ${(s.executionPressure * 100).toFixed(1)}%`,
        `Budget: ${s.allocationBudget}/100`,
        `Fragmentation: ${(s.heapFragmentation * 100).toFixed(1)}%`,
        `Active Pixels: ${s.pixelsActive}`,
        `Heap Blocks: ${heapBlocksRef.current.length}`,
        `Adaptive Nodes: ${adaptiveNodesRef.current.length}`,
        adaptiveRunningRef.current ? 'Demo: RUNNING' : 'Demo: IDLE'
      ].join('\n');
      setMessages(prev => [...prev, { role: 'model', content: metrics, timestamp: Date.now() }]);
      setStatus('CPPM: METRICS');
      return;
    }

    if (lowerCmd.includes('cppm_reset') || lowerCmd.includes('reset_heap')) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      adaptiveRunningRef.current = false;
      adaptiveNodesRef.current = [];
      heapBlocksRef.current = [{ address: HEAP_START, size: HEAP_SIZE, allocated: false }];
      initCPPM();
      const cleared = new Uint32Array(PIXEL_COUNT).fill(0xFF000000);
      vramRef.current = cleared;
      setVram(cleared);
      setMessages(prev => [...prev, { role: 'model', content: '[CPPM] heap/reset complete', timestamp: Date.now() }]);
      setStatus('CPPM: RESET');
      return;
    }

    if (lowerCmd === 'stop' || lowerCmd.includes('cppm_cancel')) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      adaptiveRunningRef.current = false;
      setStatus('CPPM: DEMO_CANCELLED');
      setMessages(prev => [...prev, { role: 'model', content: '[CPPM] demo cancelled', timestamp: Date.now() }]);
      return;
    }

    if (lowerCmd.includes('drawtestpattern')) {
        newVram = await runGpu(GPUProgram.DrawTestPattern);
        if (!newVram) {
          const cpuVram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
          for (let y = 0; y < VRAM_HEIGHT; y++) {
              for (let x = 0; x < VRAM_WIDTH; x++) {
                  const idx = y * VRAM_WIDTH + x;
                  if (x < 16 && y < 16) cpuVram[idx] = 0xFF0000;
                  else if (x >= 16 && y < 16) cpuVram[idx] = 0x00FF00;
                  else if (x < 16 && y >= 16) cpuVram[idx] = 0x0000FF;
                  else cpuVram[idx] = 0xFFFFFF;
              }
          }
          newVram = cpuVram;
        }
        setStatus(s => ({ ...s, activeProgram: 'DrawTestPattern', cycleCount: s.cycleCount + 1024 }));
    } 
    else if (lowerCmd.includes('staticnoise')) {
        newVram = await runGpu(GPUProgram.StaticNoise);
        if (!newVram) {
          const cpuVram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT);
          for (let i = 0; i < cpuVram.length; i++) {
              cpuVram[i] = Math.floor(Math.random() * 0xFFFFFF);
          }
          newVram = cpuVram;
        }
        setStatus(s => ({ ...s, activeProgram: 'StaticNoise', cycleCount: s.cycleCount + 512 }));
    }
    else if (lowerCmd.includes('bufferswap')) {
        newVram = await runGpu(GPUProgram.BufferSwap);
        if (!newVram) {
          const cpuVram = new Uint32Array(vramRef.current.length);
          for (let i = 0; i < cpuVram.length; i++) {
              cpuVram[i] = vramRef.current[i] ^ 0xFFFFFF;
          }
          newVram = cpuVram;
        }
        setStatus(s => ({ ...s, activeProgram: 'BufferSwap', cycleCount: s.cycleCount + 15365 }));
    }
    else if (lowerCmd.includes('clear')) {
         newVram = await runGpu(GPUProgram.Clear);
         if (!newVram) {
          newVram = new Uint32Array(VRAM_WIDTH * VRAM_HEIGHT).fill(0x000000);
         }
         setStatus(s => ({ ...s, activeProgram: 'CLEAR', cycleCount: s.cycleCount + 100 }));
    }

    if (newVram) {
        setVram(newVram);
        // Randomize registers to simulate activity
        setRegisters({
            r0: `0x${Math.floor(Math.random()*0xFFFFFF).toString(16).toUpperCase().padStart(6, '0')}`,
            r1: `0x1000`,
            r2: `0x2000`,
            r3: `0xFFFF`,
            pc: `0x${Math.floor(Math.random()*0xFFFF).toString(16).toUpperCase().padStart(4, '0')}`,
            flags: '0001'
        });
    }
  }, [gpuStatus]);

  const runLoopSequence = useCallback(async (initialContent: string) => {
    let content = initialContent;
    const loopTarget = autoLoopEnabled ? Math.min(Math.max(autoLoopCount, 0), 5) : 0;
    let remainingLoops = loopTarget;
    let isFirst = true;

    while (true) {
      const displayContent = isFirst ? content : `[loop] ${content}`;
      const userMsg: Message = { role: 'user', content: displayContent, timestamp: Date.now() };
      setMessages(prev => [...prev, userMsg]);

      await simulateVisuals(content);

      try {
        setShowTyping(true);
        const responseText = await sendMessageToKernel(content);
        const aiMsg: Message = { 
            role: 'model', 
            content: responseText || 'NO_RESPONSE', 
            timestamp: Date.now() 
        };
        setMessages(prev => [...prev, aiMsg]);

        if (remainingLoops > 0 && responseText) {
          remainingLoops -= 1;
          content = responseText;
          isFirst = false;
          continue;
        }
      } catch (error) {
        setMessages(prev => [...prev, { role: 'system', content: 'ERR: EXECUTION_FAILURE', timestamp: Date.now() }]);
      } finally {
        setShowTyping(false);
      }
      break;
    }
  }, [autoLoopEnabled, autoLoopCount, simulateVisuals]);

  const handleSend = async () => {
    if (!input.trim() || isProcessing) return;

    const content = input;
    setInput('');
    setIsProcessing(true);

    if (handleBlueprintCommand(content)) {
      return;
    }

    try {
      await runLoopSequence(content);
    } finally {
      setIsProcessing(false);
      setShowTyping(false);
    }
  };

  const renderTerminal = () => (
    <div className="min-h-screen bg-black text-emerald-500 font-mono flex relative overflow-hidden">
      {/* Visual Effects */}
      <div className="scanline absolute inset-0 pointer-events-none"></div>
      
      {/* Sidebar */}
      <StatusPanel registers={registers} status={status} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen relative z-10">
        
        {/* Top Bar */}
        <div className="h-12 border-b border-emerald-900 bg-black/90 flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <Terminal size={18} />
            <span className="font-bold">NEURAL_LINK_ESTABLISHED</span>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-emerald-700">MEM_USAGE: 14%</span>
            <div className="flex items-center gap-1 text-emerald-400">
                <Power size={12} className={isProcessing ? "text-amber-500 animate-pulse" : "text-emerald-500"} />
                {isProcessing ? 'PROCESSING' : 'ONLINE'}
            </div>
            <button
              onClick={() => setView('db')}
              className="px-2 py-1 border border-emerald-900 text-[10px] hover:bg-emerald-900/40 rounded"
            >
              Open DB
            </button>
            <div className={`px-2 py-1 rounded text-[10px] ${gpuStatus === 'ready' ? 'bg-emerald-900/40 text-emerald-200' : 'bg-amber-900/40 text-amber-200'}`}>
                {gpuStatus === 'ready' ? 'GPU_ACCEL' : 'CPU_FALLBACK'}
            </div>
          </div>
        </div>

        {/* Workspace Grid */}
        <div className="flex-1 flex overflow-hidden">
            
            {/* Terminal Area */}
            <div className="flex-1 flex flex-col p-4 overflow-hidden relative">
                <div className="flex-1 overflow-y-auto space-y-4 pr-2 pb-4">
                    {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[85%] p-3 rounded border ${
                        msg.role === 'user' 
                            ? 'bg-emerald-950/30 border-emerald-800 text-emerald-100' 
                            : msg.role === 'system'
                            ? 'bg-red-950/10 border-red-900/50 text-red-400 italic'
                            : 'bg-black/50 border-emerald-900/50 text-emerald-400'
                        }`}>
                        <div className="text-[10px] opacity-50 mb-1 uppercase tracking-wider flex justify-between gap-4">
                            <span>{msg.role === 'model' ? 'KERNEL_CORE' : msg.role}</span>
                            <span>{new Date(msg.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed">
                            {msg.content}
                        </pre>
                        </div>
                    </div>
                    ))}
                    {showTyping && (
                      <div className="flex gap-3 justify-start">
                        <div className="max-w-[85%] p-3 rounded border bg-black/50 border-emerald-900/50 text-emerald-400">
                          <div className="text-[10px] opacity-50 mb-1 uppercase tracking-wider flex justify-between gap-4">
                              <span>KERNEL_CORE</span>
                              <span>{new Date().toLocaleTimeString()}</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm leading-relaxed">
                            <span className="inline-flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            <span>Processing...</span>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="mt-4 flex gap-2 items-center bg-black/80 border border-emerald-900 p-2 rounded">
                    <span className="text-emerald-600 font-bold px-2">{`>`}</span>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        className="flex-1 bg-transparent border-none outline-none text-emerald-100 placeholder-emerald-900"
                        placeholder="Enter kernel command..."
                        autoFocus
                        disabled={isProcessing}
                    />
                    <button 
                        onClick={handleSend}
                        disabled={isProcessing}
                        className="p-2 hover:bg-emerald-900/30 rounded text-emerald-500 disabled:opacity-50 flex items-center justify-center min-w-[32px]"
                    >
                        {isProcessing ? (
                          <span className="h-4 w-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></span>
                        ) : (
                          <Send size={16} />
                        )}
                    </button>
                </div>
            </div>

            {/* Visualizer Panel (Right side) */}
            <div className="w-80 border-l border-emerald-900 bg-black/90 p-4 flex flex-col gap-6 overflow-y-auto">
                <div>
                    <h3 className="text-xs font-bold text-emerald-700 mb-3 uppercase tracking-widest border-b border-emerald-900/50 pb-2">
                        VRAM_BRIDGE [0x1000]
                    </h3>
                    <VRAMVisualizer 
                        data={vram} 
                        width={VRAM_WIDTH} 
                        height={VRAM_HEIGHT} 
                        label="DISPLAY_BUFFER"
                    />
                </div>

                <div className="text-[10px] text-emerald-600 font-mono space-y-2">
                    <p className="border-l-2 border-emerald-800 pl-2">
                        VISUAL_BRIDGE_STATUS:<br/>
                        <span className="text-emerald-400">ACTIVE</span>
                    </p>
                    <p className="border-l-2 border-emerald-800 pl-2">
                        WGSL_COMPUTE_UNITS:<br/>
                        <span className="text-emerald-400">ONLINE</span>
                    </p>
                    <p className="border-l-2 border-emerald-800 pl-2">
                        PYTHON_ORCHESTRATOR:<br/>
                        <span className="text-emerald-400">READY</span>
                    </p>
                    <p className="border-l-2 border-emerald-800 pl-2">
                        GPU_ACCELERATION:<br/>
                        <span className={gpuStatus === 'ready' ? 'text-emerald-400' : 'text-amber-500'}>
                            {gpuStatus === 'ready' ? 'ACTIVE' : 'FALLBACK'}
                        </span>
                    </p>
                    {gpuMessage && (
                      <p className="border-l-2 border-emerald-800 pl-2 text-[9px] text-emerald-700">
                        {gpuMessage}
                      </p>
                    )}
                    {gpuDetails && (
                      <p className="border-l-2 border-emerald-800 pl-2 text-[10px] text-emerald-500">
                        Adapter: {gpuDetails.adapterLabel || 'Unknown'}
                        {gpuDetails.features.length > 0 && (
                          <span className="block text-[9px] text-emerald-700">
                            Features: {gpuDetails.features.join(', ')}
                          </span>
                        )}
                      </p>
                    )}
                    <p className="border-l-2 border-emerald-800 pl-2 text-[10px] text-emerald-500">
                        GPU Runs: {gpuMetrics.totalRuns} {gpuMetrics.lastRunMs !== null && `(last: ${gpuMetrics.lastRunMs.toFixed(2)} ms)`}
                    </p>
                </div>

                <div className="mt-auto p-2 border border-emerald-900/30 bg-emerald-950/10 rounded">
                    <div className="text-[10px] text-emerald-500 mb-1">QUICK_CMDS:</div>
                    <div className="flex flex-wrap gap-2">
                        <button onClick={() => setInput('DrawTestPattern')} className="text-[10px] border border-emerald-800 px-2 py-1 hover:bg-emerald-900">TestPatt</button>
                        <button onClick={() => setInput('StaticNoise')} className="text-[10px] border border-emerald-800 px-2 py-1 hover:bg-emerald-900">Noise</button>
                        <button onClick={() => setInput('Clear')} className="text-[10px] border border-emerald-800 px-2 py-1 hover:bg-emerald-900">Clear</button>
                    </div>
                </div>

                <div className="p-2 border border-emerald-900/30 bg-emerald-950/10 rounded">
                    <div className="text-[10px] text-emerald-500 mb-1">AUTO_LOOP:</div>
                    <div className="flex items-center justify-between text-[10px] text-emerald-400">
                        <span>Status:</span>
                        <span className={autoLoopEnabled ? 'text-emerald-300' : 'text-amber-500'}>
                            {autoLoopEnabled ? `ON (${Math.min(Math.max(autoLoopCount, 0), 5)} hops)` : 'OFF'}
                        </span>
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                        <button
                          onClick={() => setAutoLoopEnabled(v => !v)}
                          className="text-[10px] border border-emerald-800 px-2 py-1 hover:bg-emerald-900"
                        >
                          {autoLoopEnabled ? 'Disable' : 'Enable'}
                        </button>
                        <input
                          type="number"
                          min={0}
                          max={5}
                          value={autoLoopCount}
                          onChange={e => setAutoLoopCount(Number(e.target.value) || 0)}
                          className="w-12 bg-black border border-emerald-900 text-emerald-200 text-[10px] px-1 py-0.5"
                          title="Number of times to feed the model's response back into itself (max 5)"
                        />
                    </div>
                    <p className="text-[9px] text-emerald-700 mt-1">
                        When enabled, the model's reply is re-submitted up to N times for recursive exploration.
                    </p>
                </div>
            </div>

        </div>
      </div>
    </div>
  );

  if (view === 'db') {
    return (
      <div className="min-h-screen bg-black text-emerald-500 font-mono flex relative overflow-hidden">
        <DBExplorer onClose={() => setView('terminal')} />
      </div>
    );
  }

  return renderTerminal();
};

export default App;
