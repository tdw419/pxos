#!/usr/bin/env python3
"""
Pixel OS Event System
Demonstrates how user interactions map to PixelProcess updates
"""

class PixelOSEventSystem:
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.dragged_process = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.hovered_process = None

    def on_mouse_move(self, x, y, processes):
        """Handle mouse movement - update hover states and dragging"""
        self.mouse_x = x
        self.mouse_y = y

        events = []

        # Update dragging if active
        if self.mouse_down and self.dragged_process:
            new_x = x - self.drag_offset_x
            new_y = y - self.drag_offset_y

            # Generate update event for dragged window
            events.append({
                "type": "update_process",
                "process": self.dragged_process,
                "updates": {
                    "position_x": new_x,
                    "position_y": new_y
                }
            })

        # Update hover state
        new_hover = self._find_process_at(x, y, processes)
        if new_hover != self.hovered_process:
            if self.hovered_process:
                events.append({
                    "type": "hover_end",
                    "process": self.hovered_process
                })
            if new_hover:
                events.append({
                    "type": "hover_start",
                    "process": new_hover
                })
            self.hovered_process = new_hover

        return events

    def on_mouse_down(self, x, y, processes):
        """Handle mouse button press - start dragging window"""
        self.mouse_down = True
        events = []

        # Find topmost process at click location
        clicked_name = self._find_process_at(x, y, processes)

        if clicked_name and self._is_draggable(clicked_name):
            self.dragged_process = clicked_name
            clicked = processes[clicked_name]
            # Calculate offset from process origin to click point
            self.drag_offset_x = x - clicked['position_x']
            self.drag_offset_y = y - clicked['position_y']

            events.append({
                "type": "drag_start",
                "process": clicked_name,
                "offset_x": self.drag_offset_x,
                "offset_y": self.drag_offset_y
            })

            # Bring window to front
            events.append({
                "type": "focus",
                "process": clicked_name
            })

        return events

    def on_mouse_up(self, x, y, processes):
        """Handle mouse button release - stop dragging"""
        self.mouse_down = False
        events = []

        if self.dragged_process:
            events.append({
                "type": "drag_end",
                "process": self.dragged_process
            })
            self.dragged_process = None
            self.drag_offset_x = 0
            self.drag_offset_y = 0

        # Handle click events
        clicked_name = self._find_process_at(x, y, processes)
        if clicked_name:
            events.append({
                "type": "click",
                "process": clicked_name,
                "x": x,
                "y": y
            })

        return events

    def on_key_press(self, key, modifiers):
        """Handle keyboard input"""
        events = []

        # Super key - open launcher
        if key == "Super_L" or key == "Super_R":
            events.append({
                "type": "toggle_launcher"
            })

        # Workspace switching (Ctrl+Alt+Arrow)
        if "ctrl" in modifiers and "alt" in modifiers:
            if key in ["Left", "Right", "Up", "Down"]:
                events.append({
                    "type": "switch_workspace",
                    "direction": key.lower()
                })

        return events

    def _find_process_at(self, x, y, processes):
        """Find topmost process at given coordinates"""
        # Sort by z-index descending (top to bottom)
        sorted_procs = sorted(
            processes.items(),
            key=lambda p: p[1].get('z_index', 0),
            reverse=True
        )

        for name, proc in sorted_procs:
            px = proc.get('position_x', 0)
            py = proc.get('position_y', 0)
            pw = proc.get('width', 0)
            ph = proc.get('height', 0)

            if px <= x < px + pw and py <= y < py + ph:
                return name

        return None

    def _is_draggable(self, process_name):
        """Check if process can be dragged (windows, not system UI)"""
        non_draggable = [
            "PlasmaBackground", "Taskbar", "DesktopIcons",
            "WindowManager", "WorkspaceSwitcher"
        ]
        return process_name not in non_draggable


class EventProcessor:
    """Processes events and applies updates to PixelProcesses"""

    def __init__(self, update_process_callback):
        self.update_process = update_process_callback
        self.focused_process = None
        self.z_index_counter = 5000  # Start above system UI

    def process_events(self, events, processes):
        """Process a list of events and update processes"""
        for event in events:
            event_type = event.get('type')

            if event_type == 'update_process':
                # Direct position update
                process_name = event['process']
                updates = event['updates']
                self._apply_updates(process_name, updates)
                print(f"📍 Updated {process_name}: {updates}")

            elif event_type == 'drag_start':
                process_name = event['process']
                print(f"🖱️  Started dragging: {process_name}")

            elif event_type == 'drag_end':
                process_name = event['process']
                print(f"✋ Stopped dragging: {process_name}")

            elif event_type == 'focus':
                process_name = event['process']
                self._bring_to_front(process_name, processes)
                self.focused_process = process_name
                print(f"🎯 Focused: {process_name}")

            elif event_type == 'hover_start':
                process_name = event['process']
                # Could update shader uniform to show hover state
                print(f"👆 Hover: {process_name}")

            elif event_type == 'hover_end':
                process_name = event['process']
                print(f"👋 Hover end: {process_name}")

            elif event_type == 'click':
                process_name = event['process']
                x, y = event['x'], event['y']
                print(f"🖱️  Clicked {process_name} at ({x}, {y})")

            elif event_type == 'toggle_launcher':
                print(f"🚀 Toggle launcher")
                # Could show/hide launcher process

            elif event_type == 'switch_workspace':
                direction = event['direction']
                print(f"🗂️  Switch workspace: {direction}")

    def _apply_updates(self, process_name, updates):
        """Apply updates to a process"""
        if self.update_process:
            self.update_process(process_name, updates)

    def _bring_to_front(self, process_name, processes):
        """Bring process to front by updating z-index"""
        self.z_index_counter += 1
        if self.update_process:
            self.update_process(process_name, {
                'z_index': self.z_index_counter
            })


# Demonstration
if __name__ == "__main__":
    print("🎮 PIXEL OS EVENT SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Mock processes
    processes = {
        "TerminalWindow": {
            "position_x": 100, "position_y": 100,
            "width": 600, "height": 400, "z_index": 100
        },
        "FileBrowserWindow": {
            "position_x": 400, "position_y": 200,
            "width": 600, "height": 500, "z_index": 101
        },
        "Taskbar": {
            "position_x": 0, "position_y": 980,
            "width": 1920, "height": 100, "z_index": 2000
        }
    }

    # Mock update callback
    def mock_update(name, updates):
        for key, value in updates.items():
            processes[name][key] = value

    event_system = PixelOSEventSystem()
    processor = EventProcessor(mock_update)

    print("\n📋 Simulation: Dragging Terminal Window")
    print("-" * 60)

    # Mouse down on terminal
    events = event_system.on_mouse_down(250, 120, processes)
    processor.process_events(events, processes)

    # Drag to new position
    for x in range(250, 500, 50):
        events = event_system.on_mouse_move(x, 150, processes)
        processor.process_events(events, processes)

    # Release mouse
    events = event_system.on_mouse_up(500, 150, processes)
    processor.process_events(events, processes)

    print(f"\n✅ Terminal window moved from (100, 100) to ({processes['TerminalWindow']['position_x']}, {processes['TerminalWindow']['position_y']})")

    print("\n📋 Simulation: Keyboard Shortcuts")
    print("-" * 60)

    # Super key press
    events = event_system.on_key_press("Super_L", [])
    processor.process_events(events, processes)

    # Workspace switch
    events = event_system.on_key_press("Right", ["ctrl", "alt"])
    processor.process_events(events, processes)

    print("\n" + "=" * 60)
    print("🎉 Event system demonstration complete!")
    print("""
This system demonstrates:
  ✓ Mouse event handling (move, down, up)
  ✓ Window dragging with offset calculation
  ✓ Hover state tracking
  ✓ Z-index management (bring to front)
  ✓ Keyboard shortcuts
  ✓ Event → PixelProcess update pipeline

Integration: Events trigger update_PixelProcess calls
to modify shader positions in real-time!
    """)
