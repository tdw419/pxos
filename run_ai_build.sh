#!/bin/bash
##
# Automated pxOS Builder - Quick Start Script
#
# This script runs the AI-powered pxOS build system with LM Studio.
#
# Prerequisites:
#   1. LM Studio running on localhost:1234 with a model loaded
#   2. Python 3.6+ with dependencies (requests, numpy, pillow)
#   3. QEMU (optional, for testing)
#
# Usage:
#   ./run_ai_build.sh                    # Interactive mode
#   ./run_ai_build.sh --auto             # Automated build with default goals
#   ./run_ai_build.sh --demo             # Demo the LM Studio bridge
##

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
LM_STUDIO_URL="http://localhost:1234/v1"
NETWORK_PATH="pxvm/networks/pxos_autobuild.png"
PXOS_DIR="pxos-v1.0"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   pxOS AI-Powered Build System${NC}"
echo -e "${BLUE}   LM Studio + Self-Expanding Pixel Networks${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo

# Check if LM Studio is running
echo -e "${YELLOW}🔍 Checking LM Studio connection...${NC}"
if curl -s --max-time 2 "${LM_STUDIO_URL}/models" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ LM Studio is running${NC}"
else
    echo -e "${RED}❌ ERROR: Cannot connect to LM Studio at ${LM_STUDIO_URL}${NC}"
    echo
    echo "Please ensure:"
    echo "  1. LM Studio is installed and running"
    echo "  2. A model is loaded in LM Studio"
    echo "  3. The server is listening on localhost:1234"
    echo
    exit 1
fi

# Check Python dependencies
echo -e "${YELLOW}🔍 Checking Python dependencies...${NC}"
python3 -c "import requests, numpy, PIL" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python dependencies satisfied${NC}"
else
    echo -e "${YELLOW}⚠️  Installing Python dependencies...${NC}"
    pip3 install requests numpy pillow --quiet
fi

echo

# Parse command line arguments
MODE="menu"
if [ "$1" == "--auto" ]; then
    MODE="auto"
elif [ "$1" == "--demo" ]; then
    MODE="demo"
elif [ "$1" == "--interactive" ]; then
    MODE="interactive"
fi

# Menu system
if [ "$MODE" == "menu" ]; then
    echo -e "${BLUE}Choose an operation:${NC}"
    echo
    echo "  1) Demo LM Studio Bridge (test the AI learning system)"
    echo "  2) Interactive Primitive Generator (generate code interactively)"
    echo "  3) Orchestrator - Full Pipeline (ONE command does everything)"
    echo "  4) Legacy Auto Build (old auto_build_pxos.py)"
    echo "  5) Exit"
    echo
    read -p "Enter choice [1-5]: " choice

    case $choice in
        1) MODE="demo" ;;
        2) MODE="interactive" ;;
        3) MODE="orchestrator" ;;
        4) MODE="auto" ;;
        5) echo "Exiting..."; exit 0 ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
fi

echo

# Execute based on mode
case $MODE in
    demo)
        echo -e "${GREEN}🎯 Running LM Studio Bridge Demo${NC}"
        echo
        python3 pxvm/integration/lm_studio_bridge.py --demo --network "$NETWORK_PATH"
        ;;

    interactive)
        echo -e "${GREEN}🎨 Starting Interactive Primitive Generator${NC}"
        echo
        python3 tools/ai_primitive_generator.py --interactive --network "$NETWORK_PATH"
        ;;

    orchestrator)
        echo -e "${GREEN}🚀 Starting Orchestrator - Full Pipeline${NC}"
        echo
        python3 pxos_orchestrator.py --auto
        ;;

    auto)
        echo -e "${GREEN}🚀 Starting Legacy Auto Build${NC}"
        echo
        python3 tools/auto_build_pxos.py \
            --auto \
            --network "$NETWORK_PATH" \
            --pxos-dir "$PXOS_DIR" \
            --max-iterations 10
        ;;

    auto-test)
        echo -e "${GREEN}🚀 Starting Automated Build + Testing (Orchestrator)${NC}"
        echo
        python3 pxos_orchestrator.py --auto
        ;;
esac

EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Operation completed successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    echo "Next steps:"
    echo "  • Check the generated files in ${PXOS_DIR}/"
    echo "  • Review build_report.json for details"
    echo "  • Test with: cd ${PXOS_DIR} && ./tests/boot_qemu.sh"
    echo "  • View AI learning network: ${NETWORK_PATH}"
else
    echo -e "${RED}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}❌ Operation failed with exit code ${EXIT_CODE}${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    echo "Troubleshooting:"
    echo "  • Check that LM Studio is still running"
    echo "  • Review error messages above"
    echo "  • Check build_report.json for details"
fi

echo
