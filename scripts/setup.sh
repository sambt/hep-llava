#!/bin/bash
# PhysLLaVA Environment Setup
# Run this on the cluster node before training.

set -e

echo "=== PhysLLaVA Environment Setup ==="

# 1. Create conda environment
echo "Creating conda environment..."
conda create -n physllava python=3.11 -y
conda activate physllava

# 2. Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# 4. Install flash-attention (optional but recommended)
echo "Installing flash-attention..."
pip install flash-attn --no-build-isolation || echo "flash-attn install failed, will use SDPA"

# 5. Verify GPU
echo "Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

echo "=== Setup Complete ==="
