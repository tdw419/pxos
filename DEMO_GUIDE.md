# 🎨 Pixel OS - Demo Guide

Welcome to **Pixel OS**, a revolutionary GPU-native operating system where every component is a `PixelProcess` running as a GPU shader program!

## 🚀 Quick Start - See Pixel OS in Action!

### **Web-Based Interactive Demo** (Recommended)

Experience Pixel OS running in your browser with draggable windows, live system monitor, and real-time rendering!

**How to Run:**
1.  Open your terminal.
2.  Navigate to the project directory: `cd /home/user/pxos`
3.  Run the demo server: `./run_demo.sh`
4.  Open your browser to: **http://localhost:8080/pixel_os_web_demo.html**

**What you'll see:**
*   **🪟 3 Interactive Windows:** A Terminal, a File Browser, and a System Monitor, all running concurrently.
*   **🎨 Animated Background:** A GPU-style plasma effect that serves as the dynamic desktop wallpaper.
*   **📊 Live System Stats:** A real-time FPS counter and a live clock in the taskbar.
*   **🖱️ Full Interaction:** A fully interactive environment where you can manage windows just like in a modern OS.

**Interactive Features:**
*   **Drag & Drop:** Click and drag the title bar of any window to move it around the canvas.
*   **Focus Management:** Click on any window to bring it to the front (updating its Z-order).
*   **Window Controls:**
    *   Use the **yellow button** to minimize a window.
    *   Use the **green button** to maximize or restore a window.
    *   Use the **red button** to close a window.

### **Python Demo Scripts**

(For environments without a web browser)

#### 1. Complete System Integration
```bash
python3 pixel_os_complete.py
```
This script shows the full Pixel OS architecture in action, printing the state of the event system, window manager, and workspace system to the console.

#### 2. Individual Component Demos
*   **Event System:** `python3 event_system.py`
*   **Window Manager:** `python3 window_manager.py`
*   **Workspace System:** `python3 workspace_system.py`
*   **System Applications:** `python3 system_applications.py`

## 🎯 What is Pixel OS?

Pixel OS is a **proof-of-concept** demonstrating a revolutionary OS architecture where every UI element is a GPU shader program, enabling massively parallel rendering and unprecedented performance. This demo simulates that architecture using web technologies to provide a visual and interactive experience.
