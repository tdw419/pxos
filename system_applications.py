#!/usr/bin/env python3
"""
Pixel OS System Applications
Detailed descriptions for File Browser, System Monitor, and other core apps
"""

# File Browser Application
file_browser_description = """
GPU-Native File Browser Application (800x600px window)

LAYOUT:
┌─────────────────────────────────────────────────────────┐
│ Title Bar: "Files - /home/user/Documents"              │
├──────────┬──────────────────────────────────────────────┤
│ SIDEBAR  │ MAIN VIEW                                    │
│          │                                              │
│ 📁 Home  │  Name ↓        Modified ↓      Size ↓       │
│ 💻 Desktop│  ┌───────────────────────────────────────┐ │
│ 📄 Docs  │  │ 📁 Projects   2024-01-15    4.2 GB    │ │
│ 🖼️  Pics │  │ 📁 Work       2024-01-10    1.8 GB    │ │
│ 🎵 Music │  │ 📄 notes.txt  2024-01-20    15 KB     │ │
│ 📥 Down  │  │ 📄 todo.md    2024-01-18    8 KB      │ │
│ 🗑️  Trash│  └───────────────────────────────────────┘ │
│          │                                              │
│          │  [Scrollbar on right]                       │
└──────────┴──────────────────────────────────────────────┘

VISUAL DESIGN:

1. Title Bar (height: 40px):
   - Background: gradient from rgb(60,65,70) to rgb(50,55,60)
   - Title text: white, 16px, centered
   - Window controls (right): minimize, maximize, close buttons
   - Each button: 30x30px circle on hover

2. Sidebar (width: 180px):
   - Background: rgb(40,45,50)
   - Items: 40px height each
   - Icon: 24x24px, left margin 15px
   - Label: white text, 14px
   - Hover: background rgb(50,55,60)
   - Selected: blue accent rgb(60,120,220) with left border (4px)

3. Main View:
   - Background: rgb(30,35,40)
   - Header row: rgb(45,50,55), 35px height
   - Column headers: white text, bold, 12px
   - Sort indicators: up/down arrow icons

4. File/Folder Rows:
   - Height: 45px each
   - Icon: 32x32px file/folder icon
   - Name: white text, 14px
   - Modified: gray text rgb(150,150,150), 12px
   - Size: gray text rgb(150,150,150), 12px
   - Hover: highlight row with rgb(45,50,55)
   - Selected: blue background rgb(50,100,180)
   - Double-click: open folder/file

5. Scrollbar (width: 12px):
   - Track: rgb(35,40,45)
   - Thumb: rgb(80,85,90), rounded
   - Thumb hover: rgb(100,105,110)

INTERACTIONS:
- Click folder: navigate into folder, update breadcrumb
- Right-click: show context menu (Open, Copy, Delete, Properties)
- Double-click file: open with default app
- Drag file: show drag preview, allow drop to folders
- Scroll: smooth kinetic scrolling
- Resize columns: drag column dividers
- Sort: click column headers to sort ascending/descending

ANIMATIONS:
- Folder open: slide-in from right (250ms)
- Row hover: brightness increase (100ms)
- Selection: scale slightly (1.02x) with blue highlight
- Smooth scrolling with easing
"""

system_monitor_description = """
GPU-Native System Monitor Application (700x500px window)

LAYOUT:
┌─────────────────────────────────────────────────────────┐
│ Title Bar: "System Monitor"                            │
├─────────────────────────────────────────────────────────┤
│ [CPU] [Memory] [GPU] [Network] [Processes]             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  CPU Usage: 45%                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ▁▂▃▅▄▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇▆▅▄▃▂▁ (real-time graph)  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Core 0: ████████░░ 80%                                │
│  Core 1: ██████░░░░ 60%                                │
│  Core 2: ███░░░░░░░ 30%                                │
│  Core 3: █████░░░░░ 50%                                │
│                                                         │
│  Temperature: 65°C                                      │
│  Frequency: 3.4 GHz                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘

TAB VIEWS:

1. CPU Tab:
   - Large real-time line graph (600x200px)
   - X-axis: time (last 60 seconds)
   - Y-axis: 0-100% usage
   - Line color: green gradient
   - Grid lines: subtle gray
   - Per-core bars with percentage
   - Current stats: temp, frequency, load average

2. Memory Tab:
   - Circular usage indicator (200x200px donut chart)
   - Used: blue segment
   - Available: gray segment
   - Center text: "8.2 GB / 16 GB"
   - Breakdown bars:
     * Applications: 5.2 GB
     * System: 2.1 GB
     * Cache: 0.9 GB
   - Swap usage bar below

3. GPU Tab:
   - GPU name: "AMD Radeon RX 7900 XTX"
   - Utilization graph (similar to CPU)
   - VRAM usage: 3.2 GB / 24 GB
   - Clock speed: 2400 MHz
   - Temperature: 72°C
   - Power draw: 250W
   - Fan speed: 45%

4. Network Tab:
   - Download speed graph (green line)
   - Upload speed graph (blue line)
   - Current rates in MB/s
   - Total transferred today
   - Connection list:
     * WiFi: Connected (Signal: 85%)
     * Ethernet: Disconnected

5. Processes Tab:
   - Table view with columns:
     * Name | CPU% | Memory | PID | Status
   - Sortable columns
   - Top 20 processes by resource usage
   - Right-click: End Process, Details

VISUAL DESIGN:
- Dark theme: background rgb(25,30,35)
- Accent color: cyan rgb(0,180,220)
- Text: white primary, gray secondary
- Graphs: smooth antialiased lines
- Real-time updates: 30 FPS (GPU-accelerated)
- Smooth animations for value changes

GRAPH RENDERING (GPU-optimized):
- Use shader for line drawing
- Vertex buffer for graph points
- Update buffer each frame
- Gradient fill under line
- Animated entrance (sweep from left)
- Smooth interpolation between points

COLOR CODING:
- CPU/GPU usage:
  * 0-60%: green rgb(80,200,120)
  * 60-80%: yellow rgb(220,200,80)
  * 80-100%: red rgb(220,80,80)
- Memory: blue rgb(80,150,220)
- Network: green/blue dual lines
- Temperature:
  * <70°C: green
  * 70-85°C: yellow
  * >85°C: red

INTERACTIONS:
- Hover over graph: show tooltip with exact value
- Click tab: switch view with fade transition
- Scroll in process list: smooth scrolling
- Right-click process: context menu
- Resize window: graphs scale proportionally
"""

terminal_application_description = """
GPU-Native Terminal Emulator (900x600px window)

LAYOUT:
┌─────────────────────────────────────────────────────────┐
│ Title Bar: "Terminal - /home/user"                     │
├─────────────────────────────────────────────────────────┤
│ user@pixelos:~/projects$ ls -la                        │
│ total 48                                                │
│ drwxr-xr-x 3 user user 4096 Jan 20 10:30 .             │
│ drwxr-xr-x 8 user user 4096 Jan 15 14:22 ..            │
│ -rw-r--r-- 1 user user  220 Jan 10 09:15 .bashrc       │
│ drwxr-xr-x 2 user user 4096 Jan 20 10:30 src           │
│ -rw-r--r-- 1 user user 1234 Jan 19 16:45 README.md     │
│ user@pixelos:~/projects$ █                             │
│                                                         │
└─────────────────────────────────────────────────────────┘

VISUAL DESIGN:

1. Color Scheme (Monokai-inspired):
   - Background: rgb(30,31,34)
   - Text: rgb(248,248,242)
   - Cursor: rgb(253,151,31) (blinking)
   - Selection: rgba(73,72,62,0.8)

2. ANSI Color Support:
   - Black: rgb(39,40,34)
   - Red: rgb(249,38,114)
   - Green: rgb(166,226,46)
   - Yellow: rgb(253,151,31)
   - Blue: rgb(102,217,239)
   - Magenta: rgb(174,129,255)
   - Cyan: rgb(161,239,228)
   - White: rgb(248,248,242)
   - Bright variants: +30% brightness

3. Typography:
   - Font: Monospace (Fira Code or JetBrains Mono)
   - Size: 14px
   - Line height: 1.4
   - Character width: fixed 8px (monospace)
   - Ligature support for code symbols

4. Cursor:
   - Block style (default)
   - Blinking rate: 500ms on/off
   - Color: bright orange
   - Alternative styles: underline, bar (I-beam)

FEATURES:

1. Text Rendering (GPU-accelerated):
   - Texture atlas for all ASCII characters
   - Batch rendering for performance
   - Smooth sub-pixel rendering
   - Support for Unicode/UTF-8

2. Scrollback Buffer:
   - Store last 10,000 lines
   - Smooth GPU-accelerated scrolling
   - Search capability (Ctrl+Shift+F)
   - Copy/paste support

3. Input Handling:
   - Key capture for all keyboard input
   - Special key combinations (Ctrl+C, etc)
   - Paste with formatting preservation
   - History navigation (up/down arrows)

4. Shell Integration:
   - Execute commands via subprocess
   - Stream output in real-time
   - ANSI escape sequence parsing
   - Color code interpretation

INTERACTIONS:
- Type: characters appear at cursor
- Enter: execute command, show output
- Backspace: delete character before cursor
- Ctrl+C: interrupt running process
- Ctrl+L: clear screen
- Mouse selection: highlight text for copy
- Right-click: context menu (Copy, Paste, Clear)
- Scroll: navigate history with smooth animation
- Resize: reflow text to new width

ANIMATIONS:
- Cursor blink: smooth fade in/out
- Text appearance: instant (high performance)
- Scroll: kinetic with easing
- Command completion: subtle highlight
"""

text_editor_description = r"""
GPU-Native Text Editor (1000x700px window)

LAYOUT:
┌─────────────────────────────────────────────────────────┐
│ File  Edit  View  Search  Run                          │
├──────────┬──────────────────────────────────────────────┤
│ FILES    │ main.py                                  ×   │
│          ├──────────────────────────────────────────────┤
│ 📁 src/  │ 1  #!/usr/bin/env python3                    │
│  📄 main │ 2  # Pixel OS demo                           │
│  📄 utils│ 3  # GPU Native Operating System             │
│  📄 test │ 4                                             │
│          │ 5                                             │
│ 📁 docs/ │ 6  def main():                               │
│          │ 7      print("Hello, Pixel OS!")             │
│          │ 8                                             │
│          │ 9  if __name__ == "__main__":                │
│          │10      main()█                               │
│          │11                                             │
└──────────┴──────────────────────────────────────────────┘
│ Ln 10, Col 11 | Python | UTF-8 | Spaces: 4            │
└─────────────────────────────────────────────────────────┘

VISUAL DESIGN:

1. Color Theme (VS Code Dark+):
   - Background: rgb(30,30,30)
   - Line numbers: rgb(133,133,133)
   - Cursor line: rgba(255,255,255,0.1)
   - Selection: rgba(38,79,120,0.6)
   - Syntax highlighting:
     * Keywords: rgb(197,134,192)
     * Strings: rgb(206,145,120)
     * Comments: rgb(106,153,85)
     * Functions: rgb(220,220,170)
     * Numbers: rgb(181,206,168)

2. Gutter (line numbers):
   - Width: 60px
   - Background: rgb(25,25,25)
   - Numbers: right-aligned, gray
   - Active line number: bright white

3. Minimap (right side):
   - Width: 100px
   - Miniature view of entire file
   - Current viewport: highlighted region
   - GPU-rendered at low resolution
   - Click to jump to location

4. Status Bar (bottom):
   - Height: 30px
   - Background: rgb(0,122,204)
   - Info: line/col, language, encoding, indent

FEATURES:

1. Syntax Highlighting:
   - GPU shader-based highlighting
   - Language detection from extension
   - Support for: Python, JavaScript, C++, Rust, etc.
   - Real-time re-highlighting on edit

2. Code Intelligence:
   - Auto-completion dropdown
   - Bracket matching (highlight pairs)
   - Auto-indent on Enter
   - Multi-cursor editing

3. Search & Replace:
   - Ctrl+F: find dialog
   - Regex support
   - Highlight all matches
   - Case-sensitive toggle
   - Replace with preview

4. File Tree:
   - Collapsible folders
   - File icons by type
   - Right-click: New, Delete, Rename
   - Drag & drop to move files

GPU RENDERING OPTIMIZATIONS:
- Text atlas with all characters
- Batch draw calls for lines
- Viewport culling (only render visible)
- Incremental syntax parsing
- Shader-based highlighting

INTERACTIONS:
- Type: insert at cursor
- Click: move cursor to position
- Drag: select text
- Double-click: select word
- Triple-click: select line
- Ctrl+S: save file
- Ctrl+Z/Y: undo/redo
- Scroll: smooth GPU-animated
- Resize: reflow with animation
"""

print("🖥️  PIXEL OS SYSTEM APPLICATIONS")
print("=" * 70)

applications = [
    {
        "name": "FileBrowser",
        "title": "Files",
        "description": file_browser_description,
        "default_size": (800, 600),
        "process_type": "perceptron",
        "universe": "neural"
    },
    {
        "name": "SystemMonitor",
        "title": "System Monitor",
        "description": system_monitor_description,
        "default_size": (700, 500),
        "process_type": "gravity",
        "universe": "cosmic"
    },
    {
        "name": "Terminal",
        "title": "Terminal",
        "description": terminal_application_description,
        "default_size": (900, 600),
        "process_type": "perceptron",
        "universe": "neural"
    },
    {
        "name": "TextEditor",
        "title": "Code Editor",
        "description": text_editor_description,
        "default_size": (1000, 700),
        "process_type": "perceptron",
        "universe": "neural"
    }
]

print("\n📦 System Applications:")
print("-" * 70)

for i, app in enumerate(applications, 1):
    print(f"\n{i}. {app['title']}")
    print(f"   Process Name: {app['name']}")
    print(f"   Default Size: {app['default_size'][0]}x{app['default_size'][1]}px")
    print(f"   Type: {app['process_type']}")
    print(f"   Universe: {app['universe']}")
    print(f"   Description: {len(app['description'])} characters")

print("\n" + "=" * 70)
print("✅ System applications defined!")
print()
print("Each application is a detailed GPU shader description that includes:")
print("  • Complete visual design specification")
print("  • Layout with precise measurements")
print("  • Color schemes and typography")
print("  • Interactive behaviors")
print("  • GPU rendering optimizations")
print("  • Animation specifications")
print()
print("These descriptions would be passed to generateShaderBytecode")
print("to create GPU-native PixelProcess applications!")
print()
print("Total: 4 fully-specified system applications ready for GPU rendering")
