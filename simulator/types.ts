export interface Message {
  role: 'user' | 'model' | 'system';
  content: string;
  timestamp: number;
}

export interface RegisterState {
  r0: string;
  r1: string;
  r2: string;
  r3: string;
  pc: string; // Program Counter
  flags: string;
}

export interface KernelStatus {
  activeProgram: string;
  cycleCount: number;
  halted: boolean;
  vramDirty: boolean; // Signal to re-render canvas
}

export enum PixelOpCode {
  HALT = 'HALT',
  NOP = 'NOP',
  SET = 'SET',
  LOAD = 'LOAD',
  STORE = 'STORE',
  ADD = 'ADD',
  SUB = 'SUB',
  AND = 'AND',
  OR = 'OR',
  XOR = 'XOR',
  CMP = 'CMP',
  JZ = 'JZ',
  JNZ = 'JNZ',
  BLOCK_STORE = 'BLOCK_STORE',
  PIXEL_BLEND = 'PIXEL_BLEND'
}

export interface PixelState {
  R0: number;
  R1: number;
  R2: number;
  R3: number;
  PC: number;
  FLAGS: number;
  NORTH: number;
  SOUTH: number;
  EAST: number;
  WEST: number;
  BLUEPRINT_ID: number;
  BLUEPRINT_PARAM: number;
}

export interface Blueprint {
  id: number;
  patternType: number;   // 0=solid, 1=gradient, 2=checker, 3=fractal, 4=noise, 5=animation
  ruleData: number[];
  compressionRatio?: number;
  lastExpanded?: number;
}
