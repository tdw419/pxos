"""
pxos/agent/roadmap_loader.py

Utilities for loading roadmaps from YAML files and converting them to Roadmap objects.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from pxos.agent.roadmap_types import Roadmap, RoadmapMetadata, RoadmapStep
from pxos.layout import constants as layout


def load_roadmap_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load roadmap YAML file into a dict."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def resolve_step_function(module_path: str, function_name: str):
    """
    Dynamically import a step function.

    Example:
        module_path = "pxos.agent.steps.basic_layout"
        function_name = "step_init_background"
    """
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def build_roadmap_from_dict(data: Dict[str, Any]) -> Roadmap:
    """Convert a loaded YAML dict into a Roadmap object."""

    # Parse metadata
    meta_data = data.get("metadata", {})
    metadata = RoadmapMetadata(
        name=meta_data.get("name", "Unnamed Roadmap"),
        version=meta_data.get("version", "1.0"),
        vram_width=meta_data.get("vram_width", layout.DEFAULT_VRAM_WIDTH),
        vram_height=meta_data.get("vram_height", layout.DEFAULT_VRAM_HEIGHT),
        description=meta_data.get("description", ""),
        generation=meta_data.get("generation", 1),
    )

    # Parse steps
    steps = []
    for step_data in data.get("steps", []):
        step_fn = resolve_step_function(
            step_data["module"],
            step_data["function"]
        )
        step = RoadmapStep(
            name=step_data.get("name", step_data["function"]),
            fn=step_fn,
            description=step_data.get("description", ""),
            params=step_data.get("params", {}),
        )
        steps.append(step)

    # Parse output config
    output_config = data.get("output", {})

    return Roadmap(
        metadata=metadata,
        steps=steps,
        output_config=output_config,
    )


def load_roadmap(yaml_path: str) -> Roadmap:
    """Load a complete roadmap from a YAML file."""
    data = load_roadmap_yaml(yaml_path)
    return build_roadmap_from_dict(data)
