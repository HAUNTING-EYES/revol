#!/bin/bash
# Setup script for WaveNetNeuro

echo "🌊 WaveNetNeuro Setup"
echo "====================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install torch numpy matplotlib seaborn --user

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start:"
echo "   1. Test model:      python3 wavenet_neuro.py"
echo "   2. Train & compare: python3 train_wavenet.py"
echo "   3. Visualize:       python3 visualize_dynamics.py"
echo ""
echo "📖 Read README.md for full documentation"
