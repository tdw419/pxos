#!/usr/bin/env python3
"""
Pixel OS Visualizer - Live demonstration of GPU-native OS
Shows processes, windows, workspaces, and events in real-time
"""

import tkinter as tk
from tkinter import ttk
import json
import time
from typing import Dict, List, Tuple, Optional
import threading

# Import our Pixel OS components
from event_system import PixelOSEventSystem, EventProcessor
from window_manager import WindowManager, WindowState
from workspace_system import WorkspaceManager

class PixelOSVisualizer:
    """
    Visual demonstration of Pixel OS
    Renders the OS state in real-time with interactive elements
    """

    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.scale = 0.6  # Scale for display

        # Initialize Pixel OS components
        self.event_system = PixelOSEventSystem()
        self.window_manager = WindowManager(width, height)
        self.workspace_manager = WorkspaceManager(width, height)

        # Processes registry (name -> visual properties)
        self.processes: Dict[str, dict] = {}

        # Event processor with our update callback
        self.event_processor = EventProcessor(self._update_process)

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Pixel OS - GPU-Native Operating System Visualization")
        self.root.geometry(f"{int(width * self.scale)}x{int(height * self.scale)}")
        self.root.configure(bg='#1a1a1a')

        # Create canvas for rendering
        self.canvas = tk.Canvas(
            self.root,
            width=int(width * self.scale),
            height=int(height * self.scale),
            bg='#0a0a0a',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Pixel OS v1.0 - GPU-Native Operating System",
            bg='#2a2a2a',
            fg='#00ff00',
            font=('Consolas', 10),
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind events
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.root.bind('<Key>', self._on_key_press)

        # Rendering state
        self.canvas_items: Dict[str, List[int]] = {}  # process_name -> canvas item IDs
        self.dragging = False
        self.drag_start = (0, 0)

        print("🎨 Pixel OS Visualizer initialized")
        print(f"   Display: {int(width * self.scale)}x{int(height * self.scale)} (scaled {self.scale}x)")

    def create_process(self, name: str, title: str, x: int, y: int,
                      width: int, height: int, color: str = '#2a4a6a',
                      process_type: str = 'window', z_index: int = 100):
        """Create a visual process"""

        # Register in window manager
        self.window_manager.create_window(name, title, x, y, width, height)

        # Add to workspace
        self.workspace_manager.add_window_to_workspace(name)

        # Store visual properties
        self.processes[name] = {
            'title': title,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'color': color,
            'process_type': process_type,
            'z_index': z_index,
            'visible': True
        }

        # Render on canvas
        self._render_process(name)

        print(f"✨ Created process: {name} at ({x}, {y})")

    def _render_process(self, name: str):
        """Render a process on the canvas"""
        if name not in self.processes:
            return

        # Clear existing items
        if name in self.canvas_items:
            for item_id in self.canvas_items[name]:
                self.canvas.delete(item_id)

        proc = self.processes[name]
        if not proc['visible']:
            return

        items = []

        # Scale coordinates
        x = int(proc['x'] * self.scale)
        y = int(proc['y'] * self.scale)
        w = int(proc['width'] * self.scale)
        h = int(proc['height'] * self.scale)

        if proc['process_type'] == 'window':
            # Window background
            bg = self.canvas.create_rectangle(
                x, y, x + w, y + h,
                fill=proc['color'],
                outline='#4a4a4a',
                width=2,
                tags=(name, 'window')
            )
            items.append(bg)

            # Title bar
            title_bar = self.canvas.create_rectangle(
                x, y, x + w, y + 30,
                fill='#3a5a7a',
                outline='#4a4a4a',
                tags=(name, 'titlebar')
            )
            items.append(title_bar)

            # Title text
            title_text = self.canvas.create_text(
                x + 10, y + 15,
                text=proc['title'],
                fill='white',
                font=('Arial', 10, 'bold'),
                anchor='w',
                tags=(name, 'title')
            )
            items.append(title_text)

            # Close button
            close_btn = self.canvas.create_rectangle(
                x + w - 25, y + 8, x + w - 8, y + 22,
                fill='#ff4444',
                outline='#ff6666',
                tags=(name, 'close')
            )
            items.append(close_btn)

            close_x = self.canvas.create_text(
                x + w - 16, y + 15,
                text='×',
                fill='white',
                font=('Arial', 12, 'bold'),
                tags=(name, 'close')
            )
            items.append(close_x)

        elif proc['process_type'] == 'background':
            # Background process (full screen effect)
            bg = self.canvas.create_rectangle(
                x, y, x + w, y + h,
                fill=proc['color'],
                outline='',
                tags=(name, 'background')
            )
            items.append(bg)

        elif proc['process_type'] == 'taskbar':
            # Taskbar
            bar = self.canvas.create_rectangle(
                x, y, x + w, y + h,
                fill='#1a1a1a',
                outline='#3a3a3a',
                tags=(name, 'taskbar')
            )
            items.append(bar)

            # Clock
            clock_text = self.canvas.create_text(
                x + w - 80, y + h // 2,
                text=time.strftime("%H:%M:%S"),
                fill='#00ff00',
                font=('Consolas', 12),
                tags=(name, 'clock')
            )
            items.append(clock_text)

        self.canvas_items[name] = items

        # Update Z-order
        self._update_z_order()

    def _update_z_order(self):
        """Update canvas item stacking based on z_index"""
        # Sort processes by z_index
        sorted_procs = sorted(
            self.processes.items(),
            key=lambda x: x[1]['z_index']
        )

        # Raise items in order
        for name, proc in sorted_procs:
            if name in self.canvas_items:
                for item_id in self.canvas_items[name]:
                    self.canvas.tag_raise(item_id)

    def _update_process(self, name: str, updates: dict):
        """Update process properties and re-render"""
        if name in self.processes:
            self.processes[name].update(updates)
            self._render_process(name)

            # Update status
            self._update_status()

    def _on_mouse_down(self, event):
        """Handle mouse button down"""
        # Convert to OS coordinates
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        # Generate event
        events = self.event_system.on_mouse_down(x, y, self.processes)
        self.event_processor.process_events(events, self.processes)

        self.dragging = True
        self.drag_start = (event.x, event.y)

    def _on_mouse_drag(self, event):
        """Handle mouse drag"""
        if not self.dragging:
            return

        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        events = self.event_system.on_mouse_move(x, y, self.processes)
        self.event_processor.process_events(events, self.processes)

    def _on_mouse_up(self, event):
        """Handle mouse button up"""
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        events = self.event_system.on_mouse_up(x, y, self.processes)
        self.event_processor.process_events(events, self.processes)

        self.dragging = False

    def _on_mouse_move(self, event):
        """Handle mouse movement"""
        if self.dragging:
            return

        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        events = self.event_system.on_mouse_move(x, y, self.processes)
        # Don't process hover events for now (too noisy)

    def _on_key_press(self, event):
        """Handle key press"""
        key = event.keysym
        modifiers = []

        if event.state & 0x4:  # Control
            modifiers.append('ctrl')
        if event.state & 0x8:  # Alt
            modifiers.append('alt')
        if event.state & 0x1:  # Shift
            modifiers.append('shift')

        events = self.event_system.on_key_press(key, modifiers)
        self.event_processor.process_events(events, self.processes)

    def _update_status(self):
        """Update status bar"""
        active_ws = self.workspace_manager.active_workspace
        ws_info = self.workspace_manager.get_workspace_info(active_ws)

        focused = self.window_manager.focused_window or "None"

        status = (f"Pixel OS v1.0 | "
                 f"Workspace: {ws_info['name']} | "
                 f"Windows: {len(self.window_manager.windows)} | "
                 f"Focused: {focused}")

        self.status_bar.config(text=status)

    def create_demo_environment(self):
        """Create demo Pixel OS environment"""
        print("\n🏗️  Building demo Pixel OS environment...")

        # Background
        self.create_process(
            "Background",
            "Plasma Background",
            0, 0,
            self.width, self.height,
            color='#1a0a2e',
            process_type='background',
            z_index=0
        )

        # Terminal window
        self.create_process(
            "Terminal",
            "Terminal - /home/user",
            100, 100,
            800, 500,
            color='#0a0a0a',
            process_type='window',
            z_index=100
        )

        # File browser
        self.create_process(
            "FileBrowser",
            "Files - Documents",
            400, 200,
            700, 550,
            color='#1a1a1a',
            process_type='window',
            z_index=101
        )

        # System monitor
        self.create_process(
            "SystemMonitor",
            "System Monitor",
            1100, 150,
            600, 450,
            color='#0a1a0a',
            process_type='window',
            z_index=102
        )

        # Taskbar
        self.create_process(
            "Taskbar",
            "Taskbar",
            0, self.height - 80,
            self.width, 80,
            color='#1a1a1a',
            process_type='taskbar',
            z_index=2000
        )

        self._update_status()
        print("✅ Demo environment created!")

    def run(self):
        """Run the visualizer"""
        print("\n🚀 Starting Pixel OS Visualizer...")
        print("   Click and drag windows to move them")
        print("   Windows will bring to front on click")
        print("   Close window to exit\n")

        # Create demo environment
        self.create_demo_environment()

        # Update clock periodically
        def update_clock():
            if 'Taskbar' in self.canvas_items:
                self._render_process('Taskbar')
            self.root.after(1000, update_clock)

        update_clock()

        # Start Tkinter main loop
        self.root.mainloop()


def main():
    """Main entry point"""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║              🎨  P I X E L   O S   L I V E  🎨                ║")
    print("║                                                                ║")
    print("║          GPU-Native Operating System Visualization             ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    visualizer = PixelOSVisualizer(1920, 1080)
    visualizer.run()


if __name__ == "__main__":
    main()
