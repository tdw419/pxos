#!/usr/bin/env python3
"""
GPU Hypervisor Scheduler

Intelligent scheduling of GPU primitives across multiple VMs
Uses LLM-like analysis to make optimal resource allocation decisions
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import time

@dataclass
class VMPrimitiveRequest:
    """Represents a primitive submitted by a VM for execution"""
    vm_id: int
    primitive_code: str
    priority: int  # 0-100
    estimated_gpu_time_ms: float
    memory_required: int
    timestamp: float

@dataclass
class SchedulingDecision:
    """Hypervisor's scheduling decision for a VM"""
    vm_id: int
    granted_gpu_percent: float
    granted_memory_mb: int
    execution_order: int
    reason: str

class GPUHypervisorScheduler:
    """
    LLM-driven scheduler for GPU hypervisor
    Manages fair resource allocation across VMs
    """

    def __init__(self, total_gpu_memory_mb: int):
        self.total_gpu_memory = total_gpu_memory_mb
        self.vm_queues: Dict[int, List[VMPrimitiveRequest]] = {}
        self.vm_quotas: Dict[int, float] = {}  # VM ID → GPU % quota
        self.vm_usage: Dict[int, float] = {}   # VM ID → actual usage

    def submit_primitive(self, req: VMPrimitiveRequest):
        """Guest VM submits a primitive for execution"""
        if req.vm_id not in self.vm_queues:
            self.vm_queues[req.vm_id] = []
        self.vm_queues[req.vm_id].append(req)

    def schedule(self) -> List[SchedulingDecision]:
        """
        LLM-driven scheduling decision
        Analyzes all pending requests and decides execution order
        """
        decisions = []

        # 1. Analyze workload from each VM
        vm_workloads = self._analyze_workloads()

        if not vm_workloads:
            return []

        # 2. Fair sharing calculation
        fair_allocations = self._calculate_fair_shares(vm_workloads)

        # 3. Priority adjustments
        adjusted_allocations = self._apply_priority_rules(fair_allocations, vm_workloads)

        # 4. Create execution plan
        execution_plan = self._create_execution_plan(adjusted_allocations, vm_workloads)

        return execution_plan

    def _analyze_workloads(self) -> Dict[int, dict]:
        """LLM analyzes each VM's workload characteristics"""
        workloads = {}

        for vm_id, queue in self.vm_queues.items():
            if not queue:
                continue

            # Analyze primitives using LLM-like heuristics
            total_estimated_time = sum(req.estimated_gpu_time_ms for req in queue)
            total_memory = sum(req.memory_required for req in queue)
            avg_priority = sum(req.priority for req in queue) / len(queue)

            # Classify workload type
            workload_type = self._classify_workload(queue)

            workloads[vm_id] = {
                'pending_ops': len(queue),
                'estimated_time_ms': total_estimated_time,
                'memory_required_mb': total_memory / (1024 * 1024),
                'avg_priority': avg_priority,
                'workload_type': workload_type,
            }

        return workloads

    def _classify_workload(self, queue: List[VMPrimitiveRequest]) -> str:
        """LLM classifies workload type based on primitive analysis"""
        # Analyze primitive patterns
        has_large_parallel = any('PARALLEL' in req.primitive_code for req in queue)
        has_small_latency = any('LATENCY_CRITICAL' in req.primitive_code for req in queue)
        has_long_running = any(req.estimated_gpu_time_ms > 100 for req in queue)

        if has_small_latency:
            return 'REALTIME'  # Needs low latency (e.g., rendering)
        elif has_long_running:
            return 'BATCH'     # Long-running compute (e.g., AI training)
        elif has_large_parallel:
            return 'THROUGHPUT'  # High throughput (e.g., data processing)
        else:
            return 'INTERACTIVE'  # Interactive workload

    def _calculate_fair_shares(self, workloads: Dict[int, dict]) -> Dict[int, float]:
        """Calculate fair GPU allocation for each VM"""
        if not workloads:
            return {}

        # Start with equal shares
        num_active_vms = len(workloads)
        base_share = 100.0 / num_active_vms

        allocations = {}
        for vm_id in workloads:
            allocations[vm_id] = base_share

        return allocations

    def _apply_priority_rules(self, allocations: Dict[int, float],
                               workloads: Dict[int, dict]) -> Dict[int, float]:
        """Apply priority-based adjustments (LLM-driven rules)"""
        adjusted = allocations.copy()

        for vm_id, workload in workloads.items():
            workload_type = workload['workload_type']

            # LLM rules for priority adjustments
            if workload_type == 'REALTIME':
                # Real-time workloads get guaranteed allocation
                adjusted[vm_id] = max(adjusted[vm_id], 20.0)
            elif workload_type == 'BATCH':
                # Batch workloads can flex down if needed
                adjusted[vm_id] = min(adjusted[vm_id], 60.0)

        # Normalize to 100%
        total = sum(adjusted.values())
        if total > 0:
            for vm_id in adjusted:
                adjusted[vm_id] = (adjusted[vm_id] / total) * 100.0

        return adjusted

    def _create_execution_plan(self, allocations: Dict[int, float],
                                workloads: Dict[int, dict]) -> List[SchedulingDecision]:
        """Create final execution plan with specific resource grants"""
        plan = []

        # Sort by priority (REALTIME first, then by allocation %)
        def priority_key(item):
            vm_id, gpu_percent = item
            workload_type = workloads[vm_id]['workload_type']
            priority_order = {'REALTIME': 0, 'INTERACTIVE': 1, 'THROUGHPUT': 2, 'BATCH': 3}
            return (priority_order.get(workload_type, 4), -gpu_percent)

        sorted_allocations = sorted(allocations.items(), key=priority_key)

        for order, (vm_id, gpu_percent) in enumerate(sorted_allocations):
            workload = workloads[vm_id]
            memory_grant = min(
                workload['memory_required_mb'],
                self.total_gpu_memory * (gpu_percent / 100.0)
            )

            decision = SchedulingDecision(
                vm_id=vm_id,
                granted_gpu_percent=gpu_percent,
                granted_memory_mb=int(memory_grant),
                execution_order=order,
                reason=f"Workload: {workload['workload_type']}, "
                       f"Fair share among {len(allocations)} VMs"
            )
            plan.append(decision)

        return plan


# Example usage
def demo_scheduler():
    """Demonstrate the GPU hypervisor scheduler"""
    print("=" * 70)
    print("GPU Hypervisor Scheduler Demo")
    print("=" * 70)
    print()

    scheduler = GPUHypervisorScheduler(total_gpu_memory_mb=24000)  # 24GB GPU

    # VM 1: AI Training (batch, low priority, long-running)
    print("VM 1: Submitting AI training workload (batch, long-running)")
    scheduler.submit_primitive(VMPrimitiveRequest(
        vm_id=1,
        primitive_code="""
GPU_KERNEL train_layer
GPU_PARAM weights float[]
GPU_THREAD_CODE:
    PARALLEL_MATMUL weights input → output
GPU_END
        """,
        priority=30,
        estimated_gpu_time_ms=500.0,
        memory_required=8000 * 1024 * 1024,  # 8GB
        timestamp=time.time()
    ))

    # VM 2: Real-time rendering (interactive, high priority, low latency)
    print("VM 2: Submitting real-time rendering workload (latency-critical)")
    scheduler.submit_primitive(VMPrimitiveRequest(
        vm_id=2,
        primitive_code="""
GPU_KERNEL render_frame LATENCY_CRITICAL
GPU_PARAM framebuffer pixel[]
GPU_THREAD_CODE:
    THREAD_ID → tid
    CALCULATE_PIXEL tid → color
    STORE color → framebuffer[tid]
GPU_END
        """,
        priority=90,
        estimated_gpu_time_ms=16.0,  # 16ms for 60 FPS
        memory_required=200 * 1024 * 1024,  # 200MB
        timestamp=time.time()
    ))

    # VM 3: Data processing (throughput-oriented)
    print("VM 3: Submitting data processing workload (throughput)")
    scheduler.submit_primitive(VMPrimitiveRequest(
        vm_id=3,
        primitive_code="""
GPU_KERNEL process_data
GPU_PARAM data float[]
GPU_THREAD_CODE:
    PARALLEL FOR data:
        TRANSFORM data → result
GPU_END
        """,
        priority=50,
        estimated_gpu_time_ms=100.0,
        memory_required=2000 * 1024 * 1024,  # 2GB
        timestamp=time.time()
    ))

    print()

    # Schedule!
    print("Scheduler analyzing workloads...")
    print()
    decisions = scheduler.schedule()

    print("Scheduling Decisions:")
    print("=" * 70)
    for decision in decisions:
        print(f"\nVM {decision.vm_id}:")
        print(f"  GPU Allocation: {decision.granted_gpu_percent:.1f}%")
        print(f"  Memory: {decision.granted_memory_mb} MB")
        print(f"  Execution Order: {decision.execution_order}")
        print(f"  Reason: {decision.reason}")
    print()

if __name__ == "__main__":
    demo_scheduler()
