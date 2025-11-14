#!/usr/bin/env python3
"""
PX Reflex Event System

Events are emitted by the reflex layers and delivered to
subscribed processes via the event ring buffer.
"""
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class ReflexEventType(IntEnum):
    """Predefined reflex event types"""
    # Nervous system events (0-31)
    PIXEL_CHANGED = 0
    REGION_CHANGED = 1
    HOTSPOT_DETECTED = 2

    # Immune system events (32-63)
    WRITE_BLOCKED = 32
    INTRUSION_DETECTED = 33
    PROTECTION_VIOLATION = 34

    # Cortex events (64-127)
    PATTERN_MATCHED = 64
    EDGE_DETECTED = 65
    MOTION_DETECTED = 66
    COLOR_CLUSTER = 67

    # Physics events (128-191)
    ENTROPY_THRESHOLD = 128
    THERMAL_ANOMALY = 129

    # Custom events (192-255)
    CUSTOM_START = 192


@dataclass
class ReflexEvent:
    """A single reflex event in the ring buffer"""
    event_type: int  # 0-255
    x: int  # Location x coordinate
    y: int  # Location y coordinate
    data: int  # Event-specific data (pattern ID, pixel count, etc.)
    tick: int  # VM tick when event occurred

    def to_bytes(self) -> bytes:
        """Pack into 16-byte buffer entry"""
        return bytes([
            self.event_type & 0xFF,
            (self.x >> 8) & 0xFF, self.x & 0xFF,
            (self.y >> 8) & 0xFF, self.y & 0xFF,
            (self.data >> 24) & 0xFF, (self.data >> 16) & 0xFF,
            (self.data >> 8) & 0xFF, self.data & 0xFF,
            (self.tick >> 24) & 0xFF, (self.tick >> 16) & 0xFF,
            (self.tick >> 8) & 0xFF, self.tick & 0xFF,
            0, 0, 0  # Padding to 16 bytes
        ])

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ReflexEvent':
        """Unpack from 16-byte buffer entry"""
        event_type = data[0]
        x = (data[1] << 8) | data[2]
        y = (data[3] << 8) | data[4]
        event_data = (data[5] << 24) | (data[6] << 16) | (data[7] << 8) | data[8]
        tick = (data[9] << 24) | (data[10] << 16) | (data[11] << 8) | data[12]
        return cls(event_type, x, y, event_data, tick)


class EventRingBuffer:
    """256-slot ring buffer at 0xFFFF0000 in VM memory"""

    BUFFER_START = 0xFFFF0000
    BUFFER_SIZE = 256  # slots
    SLOT_SIZE = 16  # bytes per event

    def __init__(self):
        self.events: list[Optional[ReflexEvent]] = [None] * self.BUFFER_SIZE
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0

    def push(self, event: ReflexEvent) -> bool:
        """Add event to ring buffer. Returns False if full."""
        if self.count >= self.BUFFER_SIZE:
            # Overwrite oldest
            self.read_pos = (self.read_pos + 1) % self.BUFFER_SIZE
            self.count -= 1

        self.events[self.write_pos] = event
        self.write_pos = (self.write_pos + 1) % self.BUFFER_SIZE
        self.count += 1
        return True

    def pop(self) -> Optional[ReflexEvent]:
        """Remove and return oldest event"""
        if self.count == 0:
            return None

        event = self.events[self.read_pos]
        self.events[self.read_pos] = None
        self.read_pos = (self.read_pos + 1) % self.BUFFER_SIZE
        self.count -= 1
        return event

    def peek(self) -> Optional[ReflexEvent]:
        """View oldest event without removing"""
        if self.count == 0:
            return None
        return self.events[self.read_pos]

    def clear(self):
        """Empty the buffer"""
        self.events = [None] * self.BUFFER_SIZE
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0

    def write_to_memory(self, memory: bytearray, base_addr: int = BUFFER_START):
        """Write ring buffer to VM memory at specified address"""
        # Write header: write_pos (2), read_pos (2), count (2), reserved (10)
        offset = base_addr - self.BUFFER_START
        if offset < 0 or offset + 16 > len(memory):
            return  # Out of bounds

        memory[offset:offset+2] = self.write_pos.to_bytes(2, 'little')
        memory[offset+2:offset+4] = self.read_pos.to_bytes(2, 'little')
        memory[offset+4:offset+6] = self.count.to_bytes(2, 'little')

        # Write events (256 slots * 16 bytes = 4096 bytes)
        event_start = offset + 16
        for i, event in enumerate(self.events):
            slot_offset = event_start + (i * self.SLOT_SIZE)
            if event:
                event_bytes = event.to_bytes()
                memory[slot_offset:slot_offset+self.SLOT_SIZE] = event_bytes
            else:
                # Empty slot - all zeros
                memory[slot_offset:slot_offset+self.SLOT_SIZE] = bytes(self.SLOT_SIZE)
