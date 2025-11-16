#!/bin/bash
#
# pxFS Demo Script
# Demonstrates the pixel filesystem capabilities
#

set -e

echo "========================================="
echo "   pxOS Pixel Filesystem Demo"
echo "========================================="
echo ""

# Clean up previous demo
echo "🧹 Cleaning up previous demo..."
rm -rf .pxfs demo_files demo_restored pixel_map.html 2>/dev/null || true

# Create demo files
echo "📁 Creating demo file structure..."
mkdir -p demo_files/docs demo_files/images demo_files/code

cat > demo_files/README.txt <<EOF
Welcome to pxOS Pixel Filesystem!

This is a demo file to show how pxFS compresses
entire file structures into pixel representations.

Every file becomes a single pixel on an infinite map.
EOF

cat > demo_files/docs/manual.txt <<EOF
pxFS User Manual
================

1. Compress files with: pxfs_cli.py compress-file <path>
2. Compress directories with: pxfs_cli.py compress-dir <path>
3. Visualize with: pxfs_cli.py visualize
4. Decompress with: pxfs_cli.py decompress-dir <path>
EOF

cat > demo_files/code/hello.py <<'EOF'
#!/usr/bin/env python3
"""
Hello World for pxOS
This entire file is compressed to a single pixel!
"""

def main():
    print("Hello from a pixel!")
    print("This Python file was compressed to RGB values")
    print("and stored in content-addressable storage.")

if __name__ == "__main__":
    main()
EOF

# Generate some binary data
python3 -c "import random; open('demo_files/images/noise.dat', 'wb').write(bytes([random.randint(0, 255) for _ in range(50000)]))"

# Create a duplicate to show deduplication
cp demo_files/README.txt demo_files/docs/README_COPY.txt

echo "✓ Created demo file structure"
echo ""

# Show directory tree
echo "📂 Demo directory structure:"
find demo_files -type f | sed 's/demo_files//' | sed 's/^/  /'
echo ""

# Compress the directory
echo "🗜️  Compressing files to pixels..."
python3 pxfs_cli.py compress-dir demo_files/
echo ""

# Show statistics
echo "📊 Pixel Filesystem Statistics:"
python3 pxfs_cli.py stats
echo ""

# List pixels
echo "🔍 Pixels in chunk (0,0):"
python3 pxfs_cli.py list
echo ""

# Show ASCII visualization
echo "🎨 ASCII Pixel Map:"
python3 pxfs_cli.py visualize-ascii
echo ""

# Generate HTML visualization
echo "🌐 Generating interactive HTML map..."
python3 pxfs_cli.py visualize -o pixel_map.html
echo "   ✓ Saved to: pixel_map.html"
echo "   Open in browser: file://$(pwd)/pixel_map.html"
echo ""

# Verify integrity
echo "✅ Verifying pixel integrity..."
python3 pxfs_cli.py verify
echo ""

# Decompress to verify round-trip
echo "📤 Decompressing pixels back to files..."
python3 pxfs_cli.py decompress-dir "$(pwd)/demo_files" -o demo_restored
echo ""

# Compare original and restored
echo "🔍 Comparing original vs restored..."
if diff -r demo_files/ demo_restored/; then
    echo "   ✓ Perfect match! All files restored correctly."
else
    echo "   ✗ Differences found!"
    exit 1
fi
echo ""

# Show compression stats for largest file
echo "📈 Compression details for largest file:"
largest=$(find demo_files -type f -exec ls -l {} \; | sort -k5 -rn | head -1 | awk '{print $9}')
echo "   File: $largest"
ls -lh "$largest" | awk '{print "   Original size:", $5}'

# Calculate stored size (approximate)
content_dir=".pxfs/content/objects"
if [ -d "$content_dir" ]; then
    stored=$(du -sh "$content_dir" | awk '{print $1}')
    echo "   Stored size: $stored"
fi
echo ""

# Summary
echo "========================================="
echo "   Demo Complete!"
echo "========================================="
echo ""
echo "What just happened:"
echo "  1. ✓ Created demo files"
echo "  2. ✓ Compressed to pixels"
echo "  3. ✓ Generated visual map"
echo "  4. ✓ Verified integrity"
echo "  5. ✓ Restored files perfectly"
echo ""
echo "Next steps:"
echo "  • Open pixel_map.html in your browser"
echo "  • Try compressing your own files"
echo "  • Read README_PIXEL_FS.md for details"
echo "  • Check SPECIFICATION.md for the formal spec"
echo ""
echo "pxOS: Every file is a pixel. Every pixel is a file."
echo "========================================="
