#!/bin/bash
# Pixel-LLM Test Runner
# Runs all tests with coverage reporting

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "🧪 PIXEL-LLM TEST SUITE"
echo "========================================================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗ pytest not found${NC}"
    echo "Install: pip3 install pytest pytest-cov"
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "📦 Test Environment:"
echo "   Python: $(python3 --version)"
echo "   pytest: $(pytest --version | head -n1)"
echo ""

# Run tests with coverage (if pytest-cov is available)
echo "🚀 Running tests..."
echo ""

# Check if pytest-cov is available
if python3 -c "import pytest_cov" 2>/dev/null; then
    # Run with coverage
    python3 -m pytest pixel_llm/tests/ \
        -v \
        --cov=pixel_llm/core \
        --cov-report=term-missing \
        --cov-report=html:pixel_llm/tests/htmlcov \
        --tb=short \
        "$@"

    TEST_RESULT=$?

    if [ $TEST_RESULT -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================================================"
        echo "✅ ALL TESTS PASSED"
        echo "========================================================================${NC}"
        echo ""
        echo "📊 Coverage report:"
        echo "   Terminal: See above"
        echo "   HTML: pixel_llm/tests/htmlcov/index.html"
        echo ""
        exit 0
    else
        echo ""
        echo -e "${RED}========================================================================"
        echo "❌ TESTS FAILED"
        echo "========================================================================${NC}"
        echo ""
        echo "💡 Tips:"
        echo "   - Run specific test: pytest pixel_llm/tests/test_pixelfs_basic.py -v"
        echo "   - Show full traceback: pytest pixel_llm/tests/ -vv"
        echo "   - Stop on first failure: pytest pixel_llm/tests/ -x"
        echo ""
        exit 1
    fi
else
    # Run without coverage
    echo -e "${YELLOW}⚠️  pytest-cov not installed, running without coverage${NC}"
    echo "   Install: pip3 install pytest-cov"
    echo ""

    python3 -m pytest pixel_llm/tests/ \
        -v \
        --tb=short \
        "$@"

    TEST_RESULT=$?

    if [ $TEST_RESULT -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================================================"
        echo "✅ ALL TESTS PASSED"
        echo "========================================================================${NC}"
        echo ""
        echo "💡 Install pytest-cov for coverage reports:"
        echo "   pip3 install pytest-cov"
        echo ""
        exit 0
    else
        echo ""
        echo -e "${RED}========================================================================"
        echo "❌ TESTS FAILED"
        echo "========================================================================${NC}"
        echo ""
        exit 1
    fi
fi
