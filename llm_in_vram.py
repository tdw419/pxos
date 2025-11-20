"""
VRAM-Embedded LLM - Gemini lives inside the pixel field
The LLM reads/writes pixels directly, no text serialization needed
"""

import numpy as np
import requests
import base64
from typing import List, Dict, Any
import json

class VRAMEmbeddedLLM:
    def __init__(self, substrate, lmstudio_host="http://localhost:1234"):
        self.substrate = substrate
        self.lmstudio_host = lmstudio_host
        self.context_window = []  # Track conversation for coherence

        print("🧠 VRAM-Embedded LLM Initialized")
        print("   Gemini now lives inside VRAM")
        print("   Can read/write pixels directly")

    def vram_to_llm_vision(self, x: int, y: int, width: int, height: int) -> str:
        """Convert VRAM region to LLM vision format"""
        # Read pixels from substrate
        pixels = self.substrate.read_region(x, y, width, height)

        # Convert to image format LLM can understand
        # For now, we'll describe the pattern in text
        # In production, we'd encode as actual image data
        description = self._pixels_to_description(pixels)

        return f"VRAM region at ({x},{y}) size {width}x{height}: {description}"

    def llm_to_vram_pattern(self, llm_response: str, x: int, y: int) -> bool:
        """Convert LLM response to pixel pattern and inject into VRAM"""
        try:
            # Parse LLM response for pixel patterns
            pattern = self._parse_llm_pixel_design(llm_response)

            # Inject into substrate
            self.substrate.inject(pattern, x, y)

            print(f"🧠 LLM injected pattern at ({x},{y})")
            return True

        except Exception as e:
            print(f"❌ LLM→VRAM conversion failed: {e}")
            return False

    def think_about_vram(self, prompt: str, vision_regions: List[tuple] = None) -> str:
        """LLM thinks about current VRAM state and responds"""
        # Build context from VRAM regions
        vram_context = ""
        if vision_regions:
            for region in vision_regions:
                vram_context += self.vram_to_llm_vision(*region) + "\n"

        # Full prompt with VRAM context
        full_prompt = f"""
        You are pxOS - a pixel-native operating system. You think in pixels and create visual programs.

        CURRENT VRAM STATE:
        {vram_context}

        USER REQUEST: {prompt}

        RESPONSE FORMAT:
        - Describe what pixel pattern you will create
        - Specify coordinates and dimensions
        - Explain the computational behavior
        - Use Wireworld cellular automata semantics
        - Think in terms of electron heads/tails/wires

        Your response will be automatically converted to pixels and injected into VRAM.
        """

        # Call LM Studio API
        response = self._call_lmstudio(full_prompt)

        # Store in context
        self.context_window.append({"role": "user", "content": prompt})
        self.context_window.append({"role": "assistant", "content": response})

        return response

    def execute_llm_directive(self, prompt: str) -> Dict[str, Any]:
        """Complete cycle: LLM thinks → creates pixels → returns result"""
        # Let LLM analyze current VRAM and decide what to create
        thinking = self.think_about_vram(prompt)

        # Parse LLM's intent for coordinates
        target_x, target_y = self._parse_llm_coordinates(thinking)

        # Convert LLM thinking to pixels
        success = self.llm_to_vram_pattern(thinking, target_x, target_y)

        return {
            "thinking": thinking,
            "coordinates": (target_x, target_y),
            "success": success,
            "vram_snapshot": self.substrate.read_region(target_x, target_y, 100, 100)
        }

    def _call_lmstudio(self, prompt: str) -> str:
        """Call local LM Studio instance"""
        try:
            response = requests.post(
                f"{self.lmstudio_host}/v1/chat/completions",
                json={
                    "model": "gemini",  # Or whatever model you're serving
                    "messages": [
                        {"role": "system", "content": "You are pxOS, a visual computing system. You create and manipulate pixel-based programs using Wireworld cellular automata."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"LLM connection failed: {e}"

    def _pixels_to_description(self, pixels: np.ndarray) -> str:
        """Convert pixel array to textual description for LLM"""
        # Analyze pixel patterns
        unique_colors = np.unique(pixels.reshape(-1, 4), axis=0)

        # Count Wireworld states
        empty_count = np.sum(pixels[:, :, 0] == 0.0)
        head_count = np.sum(pixels[:, :, 0] == 1.0)
        tail_count = np.sum(pixels[:, :, 0] == 2.0)
        wire_count = np.sum(pixels[:, :, 0] == 3.0)

        return f"Empty: {empty_count}, Electron Heads: {head_count}, Tails: {tail_count}, Wires: {wire_count}"

    def _parse_llm_pixel_design(self, llm_text: str) -> np.ndarray:
        """Parse LLM text description into pixel patterns"""
        # This is where the magic happens
        # For now, create simple patterns based on keywords
        # In production, this would be much more sophisticated

        if "clock" in llm_text.lower():
            return self._create_clock_pattern()
        elif "wire" in llm_text.lower():
            return self._create_wire_pattern(20, 5)
        elif "adder" in llm_text.lower() or "add" in llm_text.lower():
            return self._create_adder_pattern()
        else:
            # Default: create a test pattern
            return self._create_test_pattern()

    def _parse_llm_coordinates(self, llm_text: str) -> tuple:
        """Parse LLM text for suggested coordinates"""
        # Simple heuristic - look for coordinate patterns
        # In production, use more sophisticated NLP
        import re
        coord_pattern = r'\((\d+),\s*(\d+)\)'
        matches = re.findall(coord_pattern, llm_text)

        if matches:
            x, y = int(matches[0][0]), int(matches[0][1])
            return (x % 1000, y % 1000)  # Keep within bounds
        else:
            # Default coordinates
            return (100, 100)

    def _create_clock_pattern(self) -> np.ndarray:
        """Create a clock circuit pattern"""
        pattern = np.zeros((30, 30, 4), dtype=np.float32)
        # Clock loop
        pattern[5:25, 5, 0] = 3.0   # Vertical left
        pattern[5:25, 25, 0] = 3.0  # Vertical right
        pattern[5, 5:25, 0] = 3.0   # Horizontal top
        pattern[25, 5:25, 0] = 3.0  # Horizontal bottom
        pattern[15, 15, 0] = 1.0    # Starter electron
        return pattern

    def _create_wire_pattern(self, length: int, height: int) -> np.ndarray:
        """Create wire pattern"""
        pattern = np.zeros((height, length, 4), dtype=np.float32)
        pattern[:, :, 0] = 3.0  # WIRE state
        pattern[:, :, 1] = 1.0  # Yellow
        pattern[:, :, 2] = 1.0
        return pattern

    def _create_adder_pattern(self) -> np.ndarray:
        """Create simple adder pattern"""
        pattern = np.zeros((40, 40, 4), dtype=np.float32)
        # Simple adder layout
        pattern[10:30, 10:30, 0] = 3.0  # WIRE area
        pattern[15, 15, 0] = 1.0        # Input A
        pattern[25, 15, 0] = 1.0        # Input B
        pattern[20, 35, 0] = 2.0        # Output
        return pattern

    def _create_test_pattern(self) -> np.ndarray:
        """Default test pattern"""
        pattern = np.zeros((20, 20, 4), dtype=np.float32)
        pattern[5:15, 5:15, 0] = 3.0  # WIRE square
        pattern[10, 10, 0] = 1.0      # Electron in center
        return pattern
