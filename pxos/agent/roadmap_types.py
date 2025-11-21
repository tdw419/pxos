"""
pxos/agent/roadmap_types.py

Type definitions for the roadmap-driven VRAM OS builder.

A roadmap is a sequence of steps that transform VRAM from blank to a bootable OS.
Each step is a pure function: VRAM + context → updated context
"""

from typing import Callable, Dict, Any, List
from dataclasses import dataclass, field
from pxos.vram_sim import SimulatedVRAM

# Step function signature: takes VRAM and context, returns updated context
StepFn = Callable[[SimulatedVRAM, Dict[str, Any]], Dict[str, Any]]


@dataclass
class RoadmapStep:
    """A single step in the VRAM OS build roadmap."""
    name: str
    fn: StepFn
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def execute(self, vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this step, passing params to the function via context."""
        # Merge params into context for this step
        step_ctx = {**ctx, **self.params}
        return self.fn(vram, step_ctx)


@dataclass
class RoadmapMetadata:
    """Metadata about the roadmap itself."""
    name: str
    version: str
    vram_width: int
    vram_height: int
    description: str = ""
    generation: int = 1


@dataclass
class Roadmap:
    """Complete roadmap specification."""
    metadata: RoadmapMetadata
    steps: List[RoadmapStep]
    output_config: Dict[str, Any] = field(default_factory=dict)

    def get_output_path(self, default: str = "artifacts/vram_os.png") -> str:
        """Get the output PNG path from config."""
        return self.output_config.get("png_path", default)
