"""
PX Reflex - Biological Reflex System for pxOS

The autonomous nervous/immune/cognitive layer that operates
between kernel execution cycles.

Architecture (in biological order):
1. Nervous System - Immediate pixel change detection
2. Immune System - Protected region enforcement
3. Cortex - Pattern recognition and event emission
4. Physics - Environmental forces and homeostasis
"""

__version__ = "0.1.0"

from .core import ReflexEngine
from .events import ReflexEvent, ReflexEventType

__all__ = ['ReflexEngine', 'ReflexEvent', 'ReflexEventType']
