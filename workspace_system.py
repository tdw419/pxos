#!/usr/bin/env python3
"""
Pixel OS Workspace System
Manages virtual desktops and window organization across workspaces
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum

class WorkspaceLayout(Enum):
    FREE = "free"           # Windows positioned freely
    TILED = "tiled"         # Auto-tiled layout
    STACKED = "stacked"     # All windows stacked (like tabs)
    FULLSCREEN = "fullscreen"  # One window fullscreen

@dataclass
class Workspace:
    """Virtual desktop workspace"""
    id: str
    name: str
    windows: Set[str] = field(default_factory=set)
    layout: WorkspaceLayout = WorkspaceLayout.FREE
    background_color: tuple = (25, 25, 30)
    icon: str = "📁"

class WorkspaceManager:
    """
    Manages multiple virtual desktops in Pixel OS
    Each workspace contains a set of windows
    """

    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.workspaces: Dict[str, Workspace] = {}
        self.active_workspace: Optional[str] = None
        self.workspace_order: List[str] = []

        # Initialize default workspaces
        self._create_default_workspaces()

    def _create_default_workspaces(self):
        """Create default workspace layout"""
        default_workspaces = [
            ("ws_dev", "Development", "💻", (20, 30, 40)),
            ("ws_system", "System", "⚙️", (40, 25, 25)),
            ("ws_media", "Media", "🎵", (30, 20, 40)),
            ("ws_personal", "Personal", "🏠", (25, 35, 30))
        ]

        for ws_id, name, icon, bg_color in default_workspaces:
            workspace = Workspace(
                id=ws_id,
                name=name,
                icon=icon,
                background_color=bg_color
            )
            self.workspaces[ws_id] = workspace
            self.workspace_order.append(ws_id)

        self.active_workspace = "ws_dev"
        print(f"📁 Created {len(self.workspaces)} default workspaces")

    def create_workspace(self, name: str, icon: str = "📁") -> str:
        """Create a new workspace"""
        ws_id = f"ws_{len(self.workspaces)}"
        workspace = Workspace(
            id=ws_id,
            name=name,
            icon=icon
        )
        self.workspaces[ws_id] = workspace
        self.workspace_order.append(ws_id)

        print(f"✨ Created workspace: {name} ({ws_id})")
        return ws_id

    def delete_workspace(self, ws_id: str) -> bool:
        """Delete a workspace and move its windows to another workspace"""
        if ws_id not in self.workspaces:
            return False

        if len(self.workspaces) <= 1:
            print("❌ Cannot delete last workspace")
            return False

        workspace = self.workspaces[ws_id]

        # Move windows to next workspace
        next_ws = self._get_next_workspace(ws_id)
        if workspace.windows:
            self.workspaces[next_ws].windows.update(workspace.windows)
            print(f"📦 Moved {len(workspace.windows)} windows to {self.workspaces[next_ws].name}")

        # Delete workspace
        del self.workspaces[ws_id]
        self.workspace_order.remove(ws_id)

        # Update active workspace if needed
        if self.active_workspace == ws_id:
            self.active_workspace = next_ws

        print(f"🗑️  Deleted workspace: {workspace.name}")
        return True

    def switch_workspace(self, ws_id: str) -> Optional[dict]:
        """Switch to a different workspace"""
        if ws_id not in self.workspaces:
            return None

        old_ws = self.active_workspace
        self.active_workspace = ws_id

        # Return info about the transition
        result = {
            "from": old_ws,
            "to": ws_id,
            "from_name": self.workspaces[old_ws].name if old_ws else None,
            "to_name": self.workspaces[ws_id].name,
            "windows_to_hide": list(self.workspaces[old_ws].windows) if old_ws else [],
            "windows_to_show": list(self.workspaces[ws_id].windows)
        }

        print(f"🔄 Switched workspace: {result['from_name']} → {result['to_name']}")
        return result

    def move_window_to_workspace(self, window_name: str, target_ws: str) -> bool:
        """Move a window from one workspace to another"""
        if target_ws not in self.workspaces:
            return False

        # Remove from current workspace
        for workspace in self.workspaces.values():
            if window_name in workspace.windows:
                workspace.windows.remove(window_name)

        # Add to target workspace
        self.workspaces[target_ws].windows.add(window_name)

        print(f"📦 Moved window '{window_name}' to {self.workspaces[target_ws].name}")
        return True

    def add_window_to_workspace(self, window_name: str, ws_id: Optional[str] = None):
        """Add a window to a workspace (defaults to active)"""
        if ws_id is None:
            ws_id = self.active_workspace

        if ws_id not in self.workspaces:
            return False

        self.workspaces[ws_id].windows.add(window_name)
        print(f"➕ Added window '{window_name}' to {self.workspaces[ws_id].name}")
        return True

    def remove_window(self, window_name: str):
        """Remove a window from all workspaces"""
        for workspace in self.workspaces.values():
            if window_name in workspace.windows:
                workspace.windows.remove(window_name)
                print(f"➖ Removed window '{window_name}' from {workspace.name}")

    def get_visible_windows(self) -> Set[str]:
        """Get windows that should be visible in active workspace"""
        if not self.active_workspace:
            return set()
        return self.workspaces[self.active_workspace].windows

    def get_workspace_info(self, ws_id: str) -> Optional[dict]:
        """Get information about a workspace"""
        if ws_id not in self.workspaces:
            return None

        workspace = self.workspaces[ws_id]
        return {
            "id": workspace.id,
            "name": workspace.name,
            "icon": workspace.icon,
            "window_count": len(workspace.windows),
            "windows": list(workspace.windows),
            "layout": workspace.layout.value,
            "is_active": ws_id == self.active_workspace
        }

    def get_all_workspaces(self) -> List[dict]:
        """Get info about all workspaces"""
        return [self.get_workspace_info(ws_id)
                for ws_id in self.workspace_order]

    def navigate_workspace(self, direction: str) -> Optional[str]:
        """Navigate to next/prev workspace"""
        if not self.active_workspace:
            return None

        current_idx = self.workspace_order.index(self.active_workspace)

        if direction == "next":
            new_idx = (current_idx + 1) % len(self.workspace_order)
        elif direction == "prev":
            new_idx = (current_idx - 1) % len(self.workspace_order)
        elif direction == "up":
            new_idx = max(0, current_idx - 2)
        elif direction == "down":
            new_idx = min(len(self.workspace_order) - 1, current_idx + 2)
        else:
            return None

        target_ws = self.workspace_order[new_idx]
        self.switch_workspace(target_ws)
        return target_ws

    def set_workspace_layout(self, ws_id: str, layout: WorkspaceLayout) -> bool:
        """Set the layout mode for a workspace"""
        if ws_id not in self.workspaces:
            return False

        self.workspaces[ws_id].layout = layout
        print(f"🎨 Set {self.workspaces[ws_id].name} layout to {layout.value}")
        return True

    def _get_next_workspace(self, current_ws: str) -> str:
        """Get the next workspace ID"""
        current_idx = self.workspace_order.index(current_ws)
        next_idx = (current_idx + 1) % len(self.workspace_order)
        return self.workspace_order[next_idx]

    def print_state(self):
        """Print current workspace state"""
        print("\n" + "=" * 70)
        print("🗂️  WORKSPACE MANAGER STATE")
        print("=" * 70)

        for i, ws_id in enumerate(self.workspace_order):
            workspace = self.workspaces[ws_id]
            active = "→ " if ws_id == self.active_workspace else "  "
            print(f"{active}[{i+1}] {workspace.icon} {workspace.name}")
            print(f"     Windows: {len(workspace.windows)}")
            if workspace.windows:
                for window in workspace.windows:
                    print(f"       • {window}")
            print(f"     Layout: {workspace.layout.value}")

        print("=" * 70)

    def generate_overview_layout(self) -> List[dict]:
        """
        Generate layout for workspace overview (like macOS Mission Control)
        Returns position/size for each workspace preview
        """
        gap = 50
        preview_width = (self.screen_width - 3 * gap) // 2
        preview_height = (self.screen_height - 3 * gap) // 2

        positions = [
            (gap, gap),  # Top-left
            (gap + preview_width + gap, gap),  # Top-right
            (gap, gap + preview_height + gap),  # Bottom-left
            (gap + preview_width + gap, gap + preview_height + gap)  # Bottom-right
        ]

        layout = []
        for i, ws_id in enumerate(self.workspace_order[:4]):  # Max 4 in 2x2 grid
            workspace = self.workspaces[ws_id]
            x, y = positions[i] if i < len(positions) else (0, 0)

            layout.append({
                "workspace_id": ws_id,
                "name": workspace.name,
                "icon": workspace.icon,
                "position_x": x,
                "position_y": y,
                "width": preview_width,
                "height": preview_height,
                "window_count": len(workspace.windows),
                "is_active": ws_id == self.active_workspace
            })

        return layout


# Demonstration
if __name__ == "__main__":
    print("🗂️  PIXEL OS WORKSPACE SYSTEM DEMONSTRATION")
    print("=" * 70)

    wsm = WorkspaceManager()

    # Show initial state
    wsm.print_state()

    # Add windows to workspaces
    print("\n📦 Adding windows to workspaces...")
    wsm.add_window_to_workspace("Terminal1", "ws_dev")
    wsm.add_window_to_workspace("VSCode", "ws_dev")
    wsm.add_window_to_workspace("Browser", "ws_dev")

    wsm.add_window_to_workspace("SystemMonitor", "ws_system")
    wsm.add_window_to_workspace("Settings", "ws_system")

    wsm.add_window_to_workspace("MusicPlayer", "ws_media")
    wsm.add_window_to_workspace("VideoPlayer", "ws_media")

    wsm.add_window_to_workspace("Email", "ws_personal")

    wsm.print_state()

    # Switch workspaces
    print("\n🔄 Switching to System workspace...")
    result = wsm.switch_workspace("ws_system")
    print(f"   Hide: {result['windows_to_hide']}")
    print(f"   Show: {result['windows_to_show']}")

    # Navigate workspaces
    print("\n⬅️  Navigating to next workspace...")
    wsm.navigate_workspace("next")

    # Move window between workspaces
    print("\n📦 Moving Browser to Media workspace...")
    wsm.move_window_to_workspace("Browser", "ws_media")

    wsm.print_state()

    # Create new workspace
    print("\n✨ Creating custom workspace...")
    new_ws = wsm.create_workspace("Gaming", "🎮")
    wsm.add_window_to_workspace("SteamClient", new_ws)

    # Set layout
    print("\n🎨 Setting tiled layout for Development...")
    wsm.set_workspace_layout("ws_dev", WorkspaceLayout.TILED)

    # Generate overview layout
    print("\n🔍 Generating workspace overview layout...")
    overview = wsm.generate_overview_layout()
    for preview in overview:
        active = " [ACTIVE]" if preview['is_active'] else ""
        print(f"   {preview['icon']} {preview['name']}{active}")
        print(f"      Position: ({preview['position_x']}, {preview['position_y']})")
        print(f"      Size: {preview['width']}x{preview['height']}")
        print(f"      Windows: {preview['window_count']}")

    # Get workspace info
    print("\n📊 Development workspace info:")
    dev_info = wsm.get_workspace_info("ws_dev")
    print(f"   Name: {dev_info['name']}")
    print(f"   Windows: {dev_info['windows']}")
    print(f"   Layout: {dev_info['layout']}")

    print("\n" + "=" * 70)
    print("✅ Workspace system demonstration complete!")
    print("""
This workspace manager provides:
  ✓ Multiple virtual desktops (default: 4)
  ✓ Window organization across workspaces
  ✓ Workspace switching with visibility control
  ✓ Window movement between workspaces
  ✓ Layout modes (free, tiled, stacked, fullscreen)
  ✓ Workspace navigation (next/prev/up/down)
  ✓ Overview mode layout generation
  ✓ Dynamic workspace creation/deletion

Integration: Works with window_manager.py to organize windows
and event_system.py for keyboard shortcuts (Ctrl+Alt+Arrow)!
    """)
