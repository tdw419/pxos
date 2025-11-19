"""
Display system where kernels interpret VRAM content
This is where 'x' vs 'xyz' produces different outputs!
"""

import numpy as np

class KernelDisplaySystem:
    def __init__(self, vram_input):
        self.vram_input = vram_input
        self.active_mode = "pattern"
        self.display_modes = {}

        # Register different display kernels
        self._register_display_kernels()

    def _register_display_kernels(self):
        """Register kernels that interpret VRAM differently"""
        self.display_modes = {
            "direct": self._direct_display_kernel,
            "mirror": self._mirror_display_kernel,
            "invert": self._invert_display_kernel,
            "shift": self._shift_display_kernel,
            "pattern": self._pattern_display_kernel,
            "xyz_magic": self._xyz_magic_kernel,
        }

    def set_display_mode(self, mode: str):
        """Set how VRAM is interpreted for display"""
        if mode in self.display_modes:
            self.active_mode = mode

    def render_frame(self) -> np.ndarray:
        """Render current frame by processing VRAM through active kernel"""
        vram = self.vram_input.get_vram().copy()

        # Process VRAM through the active kernel
        kernel = self.display_modes[self.active_mode]
        processed_vram = kernel(vram)

        return processed_vram

    def _direct_display_kernel(self, vram: np.ndarray) -> np.ndarray:
        """Direct display - show VRAM as-is"""
        return vram

    def _mirror_display_kernel(self, vram: np.ndarray) -> np.ndarray:
        """Mirror the display"""
        return np.flip(vram, axis=1)

    def _invert_display_kernel(self, vram: np.ndarray) -> np.ndarray:
        """Invert colors"""
        return 255 - vram

    def _shift_display_kernel(self, vram: np.ndarray) -> np.ndarray:
        """Shift pixels"""
        return np.roll(vram, 10, axis=(0, 1))

    def _pattern_display_kernel(self, vram: np.ndarray) -> np.ndarray:
        """
        THE MAGIC HAPPENS HERE!
        Detect input patterns and transform display accordingly
        """
        input_seq = self.vram_input.get_input_sequence()

        # Check what was typed
        if 'xyz' in input_seq:
            # 'xyz' was typed - show special XYZ pattern!
            return self._create_xyz_pattern(vram)
        elif 'x' in input_seq:
            # Just 'x' was typed - show different pattern
            return self._create_x_pattern(vram)
        elif 'abc' in input_seq:
            # 'abc' sequence
            return self._create_abc_pattern(vram)

        # Default: show VRAM with slight effect
        return vram

    def _xyz_magic_kernel(self, vram: np.ndarray) -> np.ndarray:
        """Special kernel that only activates when 'xyz' is typed"""
        input_seq = self.vram_input.get_input_sequence()

        if input_seq.endswith('xyz'):
            return self._create_rainbow_explosion(vram)

        return vram

    def _create_x_pattern(self, base_vram: np.ndarray) -> np.ndarray:
        """
        Special display when 'x' is pressed
        Creates a diagonal gradient pattern
        """
        height, width = base_vram.shape[:2]
        pattern = base_vram.copy()

        # Add gradient overlay
        for y in range(height):
            for x in range(width):
                if np.any(base_vram[y, x] > 0):  # Only affect non-black pixels
                    # Diagonal gradient
                    pattern[y, x] = (
                        (x % 256),
                        (y % 256),
                        ((x + y) % 256)
                    )

        return pattern

    def _create_xyz_pattern(self, base_vram: np.ndarray) -> np.ndarray:
        """
        Special display when 'xyz' sequence is detected
        Creates a completely different visual effect!
        """
        height, width = base_vram.shape[:2]
        pattern = np.zeros((height, width, 3), dtype=np.uint8)

        # Create XYZ-specific pattern: concentric circles
        center_x, center_y = width // 2, height // 2

        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = x - center_x
                dy = y - center_y
                distance = int(np.sqrt(dx*dx + dy*dy))

                # Create ring pattern
                pattern[y, x] = (
                    (distance * 3) % 256,
                    (distance * 5) % 256,
                    (distance * 7) % 256
                )

        # Blend with original VRAM
        mask = np.any(base_vram > 0, axis=2, keepdims=True)
        return np.where(mask, pattern, base_vram)

    def _create_abc_pattern(self, base_vram: np.ndarray) -> np.ndarray:
        """Pattern for 'abc' sequence"""
        height, width = base_vram.shape[:2]
        pattern = base_vram.copy()

        # Create wave pattern
        for y in range(height):
            for x in range(width):
                if np.any(base_vram[y, x] > 0):
                    wave_x = int(20 * np.sin(y / 20.0))
                    wave_y = int(20 * np.cos(x / 20.0))

                    new_x = (x + wave_x) % width
                    new_y = (y + wave_y) % height

                    pattern[new_y, new_x] = base_vram[y, x]

        return pattern

    def _create_rainbow_explosion(self, base_vram: np.ndarray) -> np.ndarray:
        """Rainbow explosion for xyz magic mode"""
        height, width = base_vram.shape[:2]
        pattern = np.zeros((height, width, 3), dtype=np.uint8)

        # Create expanding rainbow
        center_x, center_y = width // 2, height // 2

        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = y - center_y
                angle = np.arctan2(dy, dx)
                distance = np.sqrt(dx*dx + dy*dy)

                # Rainbow based on angle
                hue = (angle + np.pi) / (2 * np.pi)
                pattern[y, x] = self._hue_to_rgb(hue, distance / max(width, height))

        return pattern

    def _hue_to_rgb(self, hue: float, saturation: float = 1.0) -> tuple:
        """Convert HSV to RGB (simplified)"""
        h = hue * 6.0
        i = int(h)
        f = h - i
        i = i % 6

        v = 255
        p = 0
        q = int(255 * (1 - f))
        t = int(255 * f)

        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

        return (0, 0, 0)
