#!/usr/bin/env python3
"""
Infinite Map Chat - Talk to LLMs on an infinite 2D plane

Navigate an infinite grid where each tile can have:
- Its own conversation history
- A selected PXDigest LLM model
- Visual representation of chat state

Controls:
    Arrow Keys / WASD - Navigate map
    Enter - Toggle chat for current tile
    1-9 - Select LLM model (from registry)
    Tab - Show model info
    Esc - Quit

Each tile is a separate conversational context.
The map is truly infinite (32-bit signed coordinates).
"""

import pygame
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pxi_cpu import query_local_llm_via_digest


@dataclass
class TileConversation:
    """Conversation state for one map tile"""
    x: int
    y: int
    model_id: Optional[int] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    last_prompt: str = ""
    last_response: str = ""

    def get_summary(self) -> str:
        """Get short summary for tile display"""
        if not self.history:
            return "Empty"
        last = self.history[-1]
        return last.get("content", "")[:20]


class InfiniteMapChat:
    """Infinite map with per-tile LLM conversations"""

    def __init__(self):
        pygame.init()

        # Display settings
        self.tile_size = 32  # pixels per tile
        self.view_w = 30  # tiles visible horizontally
        self.view_h = 20  # tiles visible vertically

        self.screen_w = self.view_w * self.tile_size
        self.screen_h = self.view_h * self.tile_size + 200  # +200 for chat UI

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("pxOS Infinite Map Chat")

        self.font_small = pygame.font.Font(None, 16)
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)

        # Map state
        self.cam_x = 0  # Camera position (tile coords)
        self.cam_y = 0
        self.cursor_x = self.view_w // 2  # Cursor (relative to view)
        self.cursor_y = self.view_h // 2

        # Conversation state
        self.tiles: Dict[str, TileConversation] = {}
        self.state_file = Path("infinite_map_state.json")
        self.load_state()

        # LLM models
        self.models = self.load_models()
        self.selected_model_idx = 0

        # Chat UI
        self.chatting = False
        self.chat_input = ""
        self.chat_history_display = []

        # Colors
        self.bg_color = (10, 10, 20)
        self.grid_color = (40, 40, 60)
        self.cursor_color = (255, 255, 100)
        self.tile_has_chat = (100, 150, 255)
        self.text_color = (220, 220, 220)
        self.chat_bg = (30, 30, 40)

        self.clock = pygame.time.Clock()

    def load_models(self) -> List[Dict]:
        """Load PXDigest LLM registry"""
        registry_path = Path("llm_pixel_registry.json")

        if not registry_path.exists():
            print("Warning: llm_pixel_registry.json not found")
            print("Create an LLM pixel with: px_digest_model.py create <name> ...")
            return []

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        models = []
        for pid, config in registry.items():
            models.append({
                "id": int(pid),
                "name": config.get("name", "Unnamed"),
                "pixel": config.get("pixel", [0, 0, 0, 0])
            })

        return models

    def load_state(self):
        """Load map state from disk"""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            for key, data in state.get("tiles", {}).items():
                x, y = map(int, key.split(","))
                self.tiles[key] = TileConversation(
                    x=x,
                    y=y,
                    model_id=data.get("model_id"),
                    history=data.get("history", [])
                )
        except Exception as e:
            print(f"Warning: Could not load state: {e}")

    def save_state(self):
        """Save map state to disk"""
        state = {"tiles": {}}

        for key, tile in self.tiles.items():
            state["tiles"][key] = {
                "model_id": tile.model_id,
                "history": tile.history
            }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_current_tile(self) -> TileConversation:
        """Get or create tile at current cursor position"""
        world_x = self.cam_x + self.cursor_x
        world_y = self.cam_y + self.cursor_y
        key = f"{world_x},{world_y}"

        if key not in self.tiles:
            self.tiles[key] = TileConversation(x=world_x, y=world_y)

        return self.tiles[key]

    def handle_chat_submit(self, text: str):
        """Submit chat message for current tile"""
        tile = self.get_current_tile()

        # Select default model if none chosen
        if tile.model_id is None and self.models:
            tile.model_id = self.models[self.selected_model_idx]["id"]

        if not tile.model_id:
            print("No LLM model available")
            return

        # Add user message
        tile.history.append({"role": "user", "content": text})
        tile.last_prompt = text

        # Build prompt from history (last 10 messages)
        recent = tile.history[-10:]
        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in recent
        ])

        print(f"\n[Tile ({tile.x},{tile.y})] Querying LLM...")
        print(f"Model ID: 0x{tile.model_id:08X}")

        # Query LLM
        response = query_local_llm_via_digest(prompt, tile.model_id, max_tokens=200)

        # Add assistant response
        tile.history.append({"role": "assistant", "content": response})
        tile.last_response = response

        print(f"Response: {response[:100]}...")

        # Update display
        self.chat_history_display = tile.history[-5:]  # Last 5 messages

    def run(self):
        """Main loop"""
        running = True

        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if self.chatting:
                        # Chat input mode
                        if event.key == pygame.K_RETURN:
                            if self.chat_input.strip():
                                self.handle_chat_submit(self.chat_input.strip())
                                self.chat_input = ""
                            self.chatting = False

                        elif event.key == pygame.K_ESCAPE:
                            self.chatting = False
                            self.chat_input = ""

                        elif event.key == pygame.K_BACKSPACE:
                            self.chat_input = self.chat_input[:-1]

                        else:
                            self.chat_input += event.unicode

                    else:
                        # Navigation mode
                        if event.key == pygame.K_ESCAPE:
                            running = False

                        elif event.key in (pygame.K_LEFT, pygame.K_a):
                            self.cam_x -= 1

                        elif event.key in (pygame.K_RIGHT, pygame.K_d):
                            self.cam_x += 1

                        elif event.key in (pygame.K_UP, pygame.K_w):
                            self.cam_y -= 1

                        elif event.key in (pygame.K_DOWN, pygame.K_s):
                            self.cam_y += 1

                        elif event.key == pygame.K_RETURN:
                            tile = self.get_current_tile()
                            self.chat_history_display = tile.history[-5:]
                            self.chatting = True

                        elif event.key == pygame.K_TAB:
                            self.show_model_info()

                        elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                           pygame.K_4, pygame.K_5, pygame.K_6,
                                           pygame.K_7, pygame.K_8, pygame.K_9):
                            idx = event.key - pygame.K_1
                            if 0 <= idx < len(self.models):
                                self.selected_model_idx = idx
                                tile = self.get_current_tile()
                                tile.model_id = self.models[idx]["id"]
                                print(f"Selected model: {self.models[idx]['name']}")

            # Render
            self.render()

            pygame.display.flip()
            self.clock.tick(30)

        # Cleanup
        self.save_state()
        pygame.quit()

    def render(self):
        """Render the map and UI"""
        self.screen.fill(self.bg_color)

        # Draw map tiles
        for ty in range(self.view_h):
            for tx in range(self.view_w):
                world_x = self.cam_x + tx
                world_y = self.cam_y + ty
                key = f"{world_x},{world_y}"

                x = tx * self.tile_size
                y = ty * self.tile_size

                # Grid lines
                pygame.draw.rect(self.screen, self.grid_color,
                                 (x, y, self.tile_size, self.tile_size), 1)

                # Tile has conversation?
                if key in self.tiles and self.tiles[key].history:
                    pygame.draw.rect(self.screen, self.tile_has_chat,
                                     (x+2, y+2, self.tile_size-4, self.tile_size-4))

                    # Show message count
                    count = len(self.tiles[key].history)
                    text = self.font_small.render(str(count), True, self.text_color)
                    self.screen.blit(text, (x+4, y+4))

        # Draw cursor
        cursor_x = self.cursor_x * self.tile_size
        cursor_y = self.cursor_y * self.tile_size
        pygame.draw.rect(self.screen, self.cursor_color,
                         (cursor_x, cursor_y, self.tile_size, self.tile_size), 3)

        # Draw coordinates
        world_x = self.cam_x + self.cursor_x
        world_y = self.cam_y + self.cursor_y
        coord_text = self.font.render(f"({world_x}, {world_y})", True, self.text_color)
        self.screen.blit(coord_text, (10, self.view_h * self.tile_size + 10))

        # Draw chat UI
        chat_y = self.view_h * self.tile_size
        pygame.draw.rect(self.screen, self.chat_bg,
                         (0, chat_y, self.screen_w, 200))

        if self.chatting:
            # Chat input box
            input_text = f"> {self.chat_input}_"
            text_surf = self.font.render(input_text, True, self.cursor_color)
            self.screen.blit(text_surf, (10, chat_y + 170))

        # Show recent history
        if self.chat_history_display:
            y_off = chat_y + 10
            for msg in self.chat_history_display[-3:]:
                role = msg.get("role", "?")
                content = msg.get("content", "")[:60]
                color = (100, 255, 100) if role == "user" else (255, 255, 100)

                text = self.font_small.render(f"{role}: {content}", True, color)
                self.screen.blit(text, (10, y_off))
                y_off += 20

        # Show selected model
        if self.models:
            model_name = self.models[self.selected_model_idx]["name"]
            model_text = self.font_small.render(f"Model: {model_name} (press 1-9 to change)",
                                                 True, (150, 150, 150))
            self.screen.blit(model_text, (self.screen_w - 400, chat_y + 10))

        # Show instructions
        if not self.chatting:
            help_text = "Arrow/WASD: Move | Enter: Chat | 1-9: Select Model | Esc: Quit"
            help_surf = self.font_small.render(help_text, True, (100, 100, 100))
            self.screen.blit(help_surf, (10, self.screen_h - 20))

    def show_model_info(self):
        """Print model info to console"""
        print("\n" + "="*60)
        print("AVAILABLE LLM MODELS")
        print("="*60)
        for i, model in enumerate(self.models, 1):
            print(f"{i}. {model['name']} (ID: 0x{model['id']:08X})")
        print("="*60 + "\n")


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║           pxOS INFINITE MAP CHAT                          ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print("Navigate an infinite 2D map.")
    print("Each tile can have its own conversation with a local LLM.")
    print()
    print("Controls:")
    print("  Arrow Keys / WASD - Navigate")
    print("  Enter - Start chatting on current tile")
    print("  1-9 - Select LLM model")
    print("  Tab - Show model info")
    print("  Esc - Quit")
    print()
    input("Press Enter to start...")

    app = InfiniteMapChat()
    app.run()


if __name__ == "__main__":
    main()
