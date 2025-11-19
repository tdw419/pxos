#!/usr/bin/env python3
"""
Main pxOS runtime - where your vision comes alive!

Type 'x' and see one pattern
Type 'x' then 'y' then 'z' and see a completely different pattern!

This demonstrates VRAM-to-VRAM programming where the same input
produces different outputs based on program interpretation.
"""

import sys
import os

# Add runtime directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'input'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'display'))

import pygame
import numpy as np
from vram_input import VRAMInputSystem
from kernel_display import KernelDisplaySystem

class PxOSRuntime:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1024, 768
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("pxOS - Pixel Primitives Operating System")

        self.clock = pygame.time.Clock()
        self.running = True

        # Our core systems
        self.vram_input = VRAMInputSystem(self.width, self.height)
        self.display_system = KernelDisplaySystem(self.vram_input)

        # Start with pattern mode (where magic happens)
        self.display_system.set_display_mode("pattern")

        # Font for instructions
        self.font = pygame.font.Font(None, 24)

        # Display instructions
        self._show_instructions()

    def _show_instructions(self):
        """Show what to type"""
        print("=" * 60)
        print("pxOS v1.0 - Pixel Primitives Operating System")
        print("=" * 60)
        print("\nKEYBOARD MAGIC:")
        print("  Type 'x' - see gradient pattern")
        print("  Type 'x', 'y', 'z' in sequence - see concentric circles!")
        print("  Type 'a', 'b', 'c' - see wave pattern")
        print("\nMODE SWITCHING:")
        print("  Press '1' - Direct display mode")
        print("  Press '2' - Pattern recognition mode (default)")
        print("  Press '3' - Mirror mode")
        print("  Press '4' - Invert colors mode")
        print("  Press 'c' - Clear screen")
        print("  Press ESC - Exit")
        print("\nREADY! Start typing...\n")

    def run(self):
        """Main runtime loop"""
        show_instructions = True
        instruction_alpha = 255

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key_event(event)
                    show_instructions = False  # Hide instructions after first key

            # Render current frame
            self._render_frame()

            # Show instructions overlay (fade out)
            if show_instructions and instruction_alpha > 0:
                self._render_instructions(instruction_alpha)
                instruction_alpha -= 2

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

    def _handle_key_event(self, event):
        """Handle keyboard input"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_c:
            # Clear screen
            self.vram_input.clear_vram()
            print("Screen cleared!")
        elif event.key == pygame.K_1:
            self.display_system.set_display_mode("direct")
            print("Mode: DIRECT - showing VRAM as-is")
        elif event.key == pygame.K_2:
            self.display_system.set_display_mode("pattern")
            print("Mode: PATTERN - pattern recognition active")
        elif event.key == pygame.K_3:
            self.display_system.set_display_mode("mirror")
            print("Mode: MIRROR - mirrored display")
        elif event.key == pygame.K_4:
            self.display_system.set_display_mode("invert")
            print("Mode: INVERT - inverted colors")
        else:
            # Convert key to character and send to VRAM system
            key_char = self._key_to_char(event.key)
            if key_char:
                self.vram_input.handle_keypress(key_char)
                sequence = self.vram_input.get_input_sequence()
                print(f"Input: '{key_char}' | Sequence: '{sequence}'")

    def _key_to_char(self, key) -> str:
        """Convert pygame key to character"""
        key_map = {
            pygame.K_x: 'x',
            pygame.K_y: 'y',
            pygame.K_z: 'z',
            pygame.K_a: 'a',
            pygame.K_b: 'b',
            pygame.K_c: 'c',
            pygame.K_SPACE: ' ',
        }
        return key_map.get(key, None)

    def _render_frame(self):
        """Render one frame to screen"""
        # Get processed VRAM from display system
        display_buffer = self.display_system.render_frame()

        # Convert numpy array to pygame surface
        # Need to transpose from (height, width, channels) to (width, height, channels)
        display_buffer = np.transpose(display_buffer, (1, 0, 2))
        pygame_surface = pygame.surfarray.make_surface(display_buffer)

        self.screen.blit(pygame_surface, (0, 0))

    def _render_instructions(self, alpha):
        """Render instruction overlay"""
        instructions = [
            "pxOS - Pixel Primitives OS",
            "",
            "Type 'x' or 'xyz' to see the magic!",
            "Press 'c' to clear, ESC to exit"
        ]

        y_offset = 20
        for line in instructions:
            text = self.font.render(line, True, (255, 255, 255))
            text.set_alpha(alpha)

            # Draw shadow
            shadow = self.font.render(line, True, (0, 0, 0))
            shadow.set_alpha(alpha)
            self.screen.blit(shadow, (22, y_offset + 2))

            # Draw text
            self.screen.blit(text, (20, y_offset))
            y_offset += 30

def main():
    """Entry point"""
    pxos = PxOSRuntime()
    pxos.run()

if __name__ == "__main__":
    main()
