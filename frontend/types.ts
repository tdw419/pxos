export type PipelineState = 'IDLE' | 'JIT_COMPILING' | 'UPLOADING_BUFFER' | 'EXECUTING';

export interface KernelStatus {
  pipelineState: PipelineState;
  statusText: string;
  cycleCount: number;
}
