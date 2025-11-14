#!/usr/bin/env python3
"""
PX Reflex Core Engine

The autonomous reflex system that operates between VM execution cycles.

Biological Layering (executed in order):
1. Nervous System - Immediate pixel sensation (change detection)
2. Immune System - Protection enforcement (revert unauthorized writes)
3. Cortex - Pattern recognition (emit events on matches)
4. Physics - Environmental forces (diffusion, entropy control)
"""
from typing import List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field

from .events import ReflexEvent, ReflexEventType, EventRingBuffer


@dataclass
class ProtectedRegion:
    """A region of pixels that is immune to modification"""
    x: int
    y: int
    width: int
    height: int
    whitelist_pids: Set[int] = field(default_factory=set)
    absolute: bool = False  # If True, even whitelisted PIDs can't write

    def contains(self, x: int, y: int) -> bool:
        """Check if pixel is within this region"""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)


@dataclass
class Pattern:
    """A visual pattern that the cortex can recognize"""
    pattern_id: int
    kernel: np.ndarray  # 2D array of expected pixel values (or -1 for wildcard)
    threshold: float = 0.9  # Match confidence threshold


class ReflexEngine:
    """
    The PX Reflex autonomous system

    Operates every VM tick, maintaining biological reflexes:
    - Nervous: Feel every pixel change
    - Immune: Protect sacred regions
    - Cortex: Recognize patterns
    - Physics: Apply environmental forces
    """

    # Memory-mapped regions in VM address space
    CHANGE_MAP_ADDR = 0xFFFE0000  # 8-bit change intensity per pixel
    EVENT_BUFFER_ADDR = 0xFFFF0000  # 256-slot event ring buffer

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height

        # Frame buffers (double-buffered for change detection)
        self.current_frame = np.zeros((height, width, 4), dtype=np.uint8)
        self.previous_frame = np.zeros((height, width, 4), dtype=np.uint8)

        # Layer 1: Nervous System
        self.change_map = np.zeros((height, width), dtype=np.uint8)

        # Layer 2: Immune System
        self.protected_regions: List[ProtectedRegion] = []
        self.violation_count = 0

        # Layer 3: Cortex (Pattern Recognition)
        self.patterns: List[Pattern] = []
        self.last_pattern_scan_tick = 0
        self.pattern_scan_interval = 10  # Scan every N ticks

        # Layer 4: Physics
        self.physics_enabled = False
        self.diffusion_rate = 0.1
        self.entropy_damping = 0.99

        # Event system
        self.event_buffer = EventRingBuffer()
        self.current_tick = 0

        # Statistics
        self.stats = {
            'pixels_changed': 0,
            'writes_blocked': 0,
            'patterns_matched': 0,
            'events_emitted': 0
        }

    def tick(self, framebuffer: np.ndarray, current_pid: Optional[int] = None) -> np.ndarray:
        """
        Execute one complete reflex cycle

        Args:
            framebuffer: Current frame from VM (height x width x 4)
            current_pid: Process ID that just executed (for immune checking)

        Returns:
            Modified framebuffer after all reflex layers
        """
        self.current_tick += 1

        # Update frame buffers
        self.previous_frame[:] = self.current_frame
        self.current_frame[:] = framebuffer

        # Layer 1: Nervous System (change detection)
        self._layer1_nervous()

        # Layer 2: Immune System (protection enforcement)
        modified_frame = self._layer2_immune(current_pid)

        # Layer 3: Cortex (pattern recognition)
        if self.current_tick % self.pattern_scan_interval == 0:
            self._layer3_cortex()

        # Layer 4: Physics (environmental forces)
        if self.physics_enabled:
            modified_frame = self._layer4_physics(modified_frame)

        return modified_frame

    def _layer1_nervous(self):
        """
        Layer 1: Nervous System

        Detects pixel-level changes and builds change map.
        This is pure sensation - no judgment, just awareness.
        """
        # Compute absolute difference in all channels
        diff = np.abs(self.current_frame.astype(np.int16) -
                     self.previous_frame.astype(np.int16))

        # Sum across RGB channels (ignore alpha)
        change_intensity = np.sum(diff[:, :, :3], axis=2).astype(np.uint8)

        # Clamp to 255
        self.change_map = np.minimum(change_intensity, 255)

        # Count changed pixels
        changed_pixels = np.count_nonzero(self.change_map > 0)
        self.stats['pixels_changed'] = changed_pixels

        # Emit hotspot events if large change detected
        if changed_pixels > 1000:  # Threshold for "hotspot"
            # Find centroid of change
            y_coords, x_coords = np.where(self.change_map > 0)
            if len(x_coords) > 0:
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))

                event = ReflexEvent(
                    event_type=ReflexEventType.HOTSPOT_DETECTED,
                    x=cx,
                    y=cy,
                    data=changed_pixels,
                    tick=self.current_tick
                )
                self.emit_event(event)

    def _layer2_immune(self, current_pid: Optional[int]) -> np.ndarray:
        """
        Layer 2: Immune System

        Enforces protected regions by reverting unauthorized writes.
        This is the guardian - automatic, merciless, instant.
        """
        modified_frame = self.current_frame.copy()

        for region in self.protected_regions:
            # Check if this region was modified
            region_mask = np.zeros((self.height, self.width), dtype=bool)
            y_start = max(0, region.y)
            y_end = min(self.height, region.y + region.height)
            x_start = max(0, region.x)
            x_end = min(self.width, region.x + region.width)

            region_mask[y_start:y_end, x_start:x_end] = True

            # Find pixels that changed in this region
            changed_in_region = (self.change_map > 0) & region_mask

            if np.any(changed_in_region):
                # Check if current PID is whitelisted
                allowed = (current_pid in region.whitelist_pids) if not region.absolute else False

                if not allowed:
                    # REVERT: Restore previous frame pixels
                    modified_frame[changed_in_region] = self.previous_frame[changed_in_region]

                    # Count violations
                    violation_count = np.count_nonzero(changed_in_region)
                    self.stats['writes_blocked'] += violation_count
                    self.violation_count += violation_count

                    # Emit intrusion event
                    y_coords, x_coords = np.where(changed_in_region)
                    if len(x_coords) > 0:
                        cx = int(np.mean(x_coords))
                        cy = int(np.mean(y_coords))

                        event = ReflexEvent(
                            event_type=ReflexEventType.INTRUSION_DETECTED,
                            x=cx,
                            y=cy,
                            data=violation_count,
                            tick=self.current_tick
                        )
                        self.emit_event(event)

        return modified_frame

    def _layer3_cortex(self):
        """
        Layer 3: Cortex (Pattern Recognition)

        Scans for registered patterns and emits events on matches.
        This is higher-level cognition - seeing shapes, not just pixels.
        """
        self.last_pattern_scan_tick = self.current_tick

        for pattern in self.patterns:
            matches = self._scan_for_pattern(pattern)

            for (x, y, confidence) in matches:
                event = ReflexEvent(
                    event_type=ReflexEventType.PATTERN_MATCHED,
                    x=x,
                    y=y,
                    data=pattern.pattern_id | (int(confidence * 255) << 16),
                    tick=self.current_tick
                )
                self.emit_event(event)
                self.stats['patterns_matched'] += 1

    def _scan_for_pattern(self, pattern: Pattern) -> List[Tuple[int, int, float]]:
        """
        Scan framebuffer for pattern matches

        Returns list of (x, y, confidence) tuples
        """
        matches = []
        kernel_h, kernel_w = pattern.kernel.shape[:2]

        # Simple sliding window (could be optimized with convolution)
        for y in range(0, self.height - kernel_h, 8):  # Step by 8 for performance
            for x in range(0, self.width - kernel_w, 8):
                region = self.current_frame[y:y+kernel_h, x:x+kernel_w, :3]  # RGB only

                # Check if region matches pattern
                confidence = self._match_confidence(region, pattern.kernel)

                if confidence >= pattern.threshold:
                    matches.append((x, y, confidence))

        return matches

    def _match_confidence(self, region: np.ndarray, kernel: np.ndarray) -> float:
        """
        Compute match confidence between region and pattern kernel

        kernel values: -1 = wildcard (any value matches)
        """
        if region.shape[:2] != kernel.shape[:2]:
            return 0.0

        # For simplicity, we'll just check if non-wildcard pixels are similar
        # In a real system, this would be more sophisticated (normalized cross-correlation, etc.)
        mask = kernel >= 0  # Non-wildcard pixels
        if not np.any(mask):
            return 1.0  # All wildcard = perfect match

        # Simple pixel-wise comparison
        diff = np.abs(region.astype(np.float32) - kernel.astype(np.float32))
        error = np.mean(diff[mask])

        # Convert error to confidence (lower error = higher confidence)
        confidence = max(0.0, 1.0 - (error / 255.0))
        return confidence

    def _layer4_physics(self, framebuffer: np.ndarray) -> np.ndarray:
        """
        Layer 4: Environmental Physics

        Applies subtle forces: diffusion, entropy damping, color gravity.
        This makes the world feel alive even when nothing is happening.
        """
        # Simple diffusion (blur effect)
        from scipy.ndimage import gaussian_filter

        if self.diffusion_rate > 0:
            for c in range(3):  # RGB channels only
                framebuffer[:, :, c] = gaussian_filter(
                    framebuffer[:, :, c],
                    sigma=self.diffusion_rate
                ).astype(np.uint8)

        # Entropy damping (slight fade to prevent heat death)
        if self.entropy_damping < 1.0:
            framebuffer[:, :, :3] = (framebuffer[:, :, :3].astype(np.float32) *
                                     self.entropy_damping).astype(np.uint8)

        return framebuffer

    def emit_event(self, event: ReflexEvent):
        """Add event to ring buffer"""
        self.event_buffer.push(event)
        self.stats['events_emitted'] += 1

    def add_protected_region(self, region: ProtectedRegion):
        """Register a new immune-protected region"""
        self.protected_regions.append(region)

    def register_pattern(self, pattern: Pattern):
        """Register a new pattern for cortex to recognize"""
        self.patterns.append(pattern)

    def write_to_vm_memory(self, memory: bytearray):
        """
        Write reflex state to VM memory-mapped regions

        - Change map at 0xFFFE0000
        - Event buffer at 0xFFFF0000
        """
        # Write change map (flattened 8-bit intensity map)
        # For now, we'll write the first 64KB of change map
        change_bytes = self.change_map.flatten().tobytes()
        max_change_bytes = min(len(change_bytes), 0x10000)  # 64KB max

        if self.CHANGE_MAP_ADDR < len(memory):
            end_addr = min(self.CHANGE_MAP_ADDR + max_change_bytes, len(memory))
            memory[self.CHANGE_MAP_ADDR:end_addr] = change_bytes[:end_addr - self.CHANGE_MAP_ADDR]

        # Write event buffer
        self.event_buffer.write_to_memory(memory, self.EVENT_BUFFER_ADDR)

    def get_stats(self) -> dict:
        """Get reflex engine statistics"""
        return {
            **self.stats,
            'protected_regions': len(self.protected_regions),
            'registered_patterns': len(self.patterns),
            'events_pending': self.event_buffer.count,
            'current_tick': self.current_tick
        }
