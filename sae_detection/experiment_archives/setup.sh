#!/bin/bash
# Setup script for Project Lightbright v2

echo "🔧 Setting up Project Lightbright v2..."

# Install requirements
pip install -r requirements.txt

# Pre-download models (optional, will download on first run anyway)
echo "📥 Pre-downloading models (optional, Ctrl+C to skip)..."
python -c "
from huggingface_hub import snapshot_download
print('Downloading Gemma-27B tokenizer...')
snapshot_download('google/gemma-2-27b', allow_patterns=['tokenizer*', 'config.json'])
print('Done!')
"

echo "✅ Setup complete!"
echo ""
echo "Run experiments with:"
echo "  python run_experiments.py --all"
echo ""
echo "Or run individually:"
echo "  python run_experiments.py --exp1  # Find features"
echo "  python run_experiments.py --exp5  # Hard negative test (key!)"
