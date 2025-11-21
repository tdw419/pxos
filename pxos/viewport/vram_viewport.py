"""
pxos/viewport/vram_viewport.py

VRAM Viewport - a live 'screen' into SimulatedVRAM.

This creates a window that acts as a viewport into the VRAM texture,
letting you watch the OS being written in real-time.

Features:
- Pan around VRAM with arrow keys
- Zoom in/out with +/- keys
- Reset view with 'r'
- Auto-refresh to show live updates

Usage:
    from pxos.vram_sim import SimulatedVRAM
    from pxos.viewport.vram_viewport import launch_vram_viewer

    vram = SimulatedVRAM(512, 512)
    # ... agent writes to vram ...
    launch_vram_viewer(vram)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

from pxos.vram_sim import SimulatedVRAM


@dataclass
class ViewportState:
    """Current viewport camera state."""
    vx: int = 0       # top-left X in VRAM
    vy: int = 0       # top-left Y in VRAM
    vw: int = 512     # viewport width in VRAM pixels
    vh: int = 512     # viewport height in VRAM pixels
    zoom: int = 1     # screen pixels per VRAM pixel


class VRAMViewport:
    """Interactive viewport into VRAM."""

    def __init__(self,
                 root: tk.Tk,
                 vram: SimulatedVRAM,
                 refresh_ms: int = 100):
        self.root = root
        self.vram = vram
        self.refresh_ms = refresh_ms

        # Initial viewport: full VRAM
        self.state = ViewportState(
            vx=0,
            vy=0,
            vw=vram.width,
            vh=vram.height,
            zoom=1,
        )

        # Create canvas
        self.canvas = tk.Canvas(root, width=512, height=512, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Store PhotoImage reference to avoid GC
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Create status label
        self.status_label = tk.Label(
            root,
            text="",
            bg='black',
            fg='white',
            font=('Courier', 10),
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

        # Bind keys for pan/zoom
        root.bind("<Left>", self.on_left)
        root.bind("<Right>", self.on_right)
        root.bind("<Up>", self.on_up)
        root.bind("<Down>", self.on_down)
        root.bind("+", self.on_zoom_in)
        root.bind("=", self.on_zoom_in)  # often shift+ = for '+'
        root.bind("-", self.on_zoom_out)
        root.bind("_", self.on_zoom_out)
        root.bind("r", self.on_reset)
        root.bind("q", lambda e: root.quit())

        # Handle resizing
        root.bind("<Configure>", self.on_resize)

        # Start refresh loop
        self.schedule_refresh()

    # ---------- Event handlers ----------

    def on_left(self, event=None):
        """Pan left."""
        self.state.vx = max(0, self.state.vx - 16)

    def on_right(self, event=None):
        """Pan right."""
        self.state.vx = min(
            max(0, self.vram.width - self.state.vw),
            self.state.vx + 16,
        )

    def on_up(self, event=None):
        """Pan up."""
        self.state.vy = max(0, self.state.vy - 16)

    def on_down(self, event=None):
        """Pan down."""
        self.state.vy = min(
            max(0, self.vram.height - self.state.vh),
            self.state.vy + 16,
        )

    def on_zoom_in(self, event=None):
        """Zoom in."""
        self.state.zoom = min(16, self.state.zoom + 1)

    def on_zoom_out(self, event=None):
        """Zoom out."""
        self.state.zoom = max(1, self.state.zoom - 1)

    def on_reset(self, event=None):
        """Reset view to full VRAM."""
        self.state.vx = 0
        self.state.vy = 0
        self.state.vw = self.vram.width
        self.state.vh = self.vram.height
        self.state.zoom = 1

    def on_resize(self, event):
        """Handle window resize."""
        if event.width <= 1 or event.height <= 1:
            return
        # Adjust viewport size to window size / zoom
        self.state.vw = max(1, event.width // self.state.zoom)
        self.state.vh = max(1, (event.height - 30) // self.state.zoom)  # -30 for status bar

    # ---------- Rendering ----------

    def schedule_refresh(self):
        """Schedule next refresh."""
        self.root.after(self.refresh_ms, self.refresh)

    def refresh(self):
        """Refresh viewport from current VRAM state."""
        # Clamp viewport to VRAM bounds
        self.state.vx = max(0, min(self.state.vx, self.vram.width - 1))
        self.state.vy = max(0, min(self.state.vy, self.vram.height - 1))
        self.state.vw = max(1, min(self.state.vw, self.vram.width - self.state.vx))
        self.state.vh = max(1, min(self.state.vh, self.vram.height - self.state.vy))

        # Copy VRAM slice (thread-safe copy)
        vx, vy, vw, vh = (
            self.state.vx,
            self.state.vy,
            self.state.vw,
            self.state.vh,
        )

        # Shallow copy of that region
        sub = np.array(self.vram.buffer[vy:vy+vh, vx:vx+vw, :], copy=True)

        # Create PIL image
        img = Image.fromarray(sub, mode="RGBA")

        # Scale according to zoom
        zoom = self.state.zoom
        new_w = max(1, vw * zoom)
        new_h = max(1, vh * zoom)
        img = img.resize((new_w, new_h), resample=Image.NEAREST)

        # Update canvas
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        # Update status
        status = (
            f"VRAM: {self.vram.width}x{self.vram.height} | "
            f"View: ({vx},{vy}) {vw}x{vh} | "
            f"Zoom: {zoom}x | "
            f"Keys: ←→↑↓ pan, +/- zoom, R reset, Q quit"
        )
        self.status_label.config(text=status)

        # Schedule next frame
        self.schedule_refresh()


def launch_vram_viewer(vram: SimulatedVRAM,
                       refresh_ms: int = 100,
                       title: str = "VRAM Viewport",
                       window_size: tuple = (512, 512)):
    """
    Launch a Tkinter window that acts as a 'screen' into VRAM.

    This blocks until the window is closed.
    To run alongside other code, launch in a separate thread.
    """
    root = tk.Tk()
    root.title(title)
    root.geometry(f"{window_size[0]}x{window_size[1]}")
    viewer = VRAMViewport(root, vram, refresh_ms=refresh_ms)
    root.mainloop()


def launch_vram_viewer_thread(vram: SimulatedVRAM,
                               refresh_ms: int = 100,
                               title: str = "VRAM Viewport"):
    """
    Launch viewport in a separate daemon thread.

    Returns immediately. Viewport runs in background.
    """
    import threading

    def run_viewer():
        launch_vram_viewer(vram, refresh_ms, title)

    t = threading.Thread(target=run_viewer, daemon=True)
    t.start()
    return t
