export type PipelineState = 'IDLE' | 'JIT_COMPILING' | 'SPIRV_EMIT' | 'GPU_UPLOAD' | 'EXECUTING';

export interface KernelStatus {
  pipelineState: PipelineState;
  statusText: string;
  cycleCount: number;
  activeCores?: number;
  neighborOps?: number;
}
