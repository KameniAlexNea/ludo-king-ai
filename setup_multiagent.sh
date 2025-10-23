#!/bin/bash
# Quick setup script for multi-agent Ludo training

set -e

echo "=========================================="
echo "Multi-Agent Ludo Training Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install pettingzoo>=1.24.0 supersuit>=3.9.0

echo ""
echo "Installing project dependencies..."
pip install -e .

# Run test
echo ""
echo "=========================================="
echo "Testing Multi-Agent Environment"
echo "=========================================="
python test_multiagent.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start training:"
    echo "   python src/train_multiagent.py --total-steps 1000000 --n-envs 16"
    echo ""
    echo "2. Monitor progress:"
    echo "   tensorboard --logdir=training/logs"
    echo ""
    echo "3. Compare with single-agent:"
    echo "   python src/train.py --total-steps 1000000 --n-envs 16"
    echo ""
    echo "For more information, see:"
    echo "  - MULTIAGENT_GUIDE.md (comprehensive guide)"
    echo "  - MULTIAGENT_SUMMARY.md (quick reference)"
    echo ""
else
    echo ""
    echo "❌ Setup failed during testing"
    echo "Please check the error messages above"
    exit 1
fi
