#!/bin/bash
# Pixel OS Web Demo Server

echo "🎨 Starting Pixel OS Web Demo Server..."
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║              🎨  P I X E L   O S   L I V E  🎨                ║"
echo "║                                                                ║"
echo "║          GPU-Native Operating System Visualization             ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Serving from: /home/user/pxos/"
echo "🌐 Open your browser to:"
echo ""
echo "   http://localhost:8080/pixel_os_web_demo.html"
echo ""
echo "✨ Features:"
echo "   • Drag windows to move them"
echo "   • Click windows to focus"
echo "   • Use window controls (minimize/maximize/close)"
echo "   • Real-time FPS counter"
echo "   • Live clock in taskbar"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd /home/user/pxos
python3 -m http.server 8080
