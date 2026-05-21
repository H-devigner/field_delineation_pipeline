#!/bin/bash
# ── Field Delineation Pipeline — Setup Script ────────────────
# Run this after cloning the repo to install all dependencies.
#
# Prerequisites:
#   - conda environment with Python 3.10
#   - CUDA 11.8 toolkit installed in the conda env
#   (see docs/detectron2_install.md for the full guide)
#
# Usage:
#   conda activate detectron2_env
#   chmod +x setup.sh && ./setup.sh

set -e

echo "═══════════════════════════════════════════════════"
echo "  Field Delineation Pipeline — Setup"
echo "═══════════════════════════════════════════════════"

# ── 1. Init submodules (detectron2) ──────────────────────────
echo ""
echo "→ Initializing git submodules (detectron2)..."
git submodule update --init --recursive

# ── 2. Install Python dependencies ──────────────────────────
echo ""
echo "→ Installing Python dependencies..."
pip install -r requirements.txt

# ── 3. Check torch is installed ──────────────────────────────
echo ""
echo "→ Checking torch..."
python -c "import torch; print(f'  ✅ torch {torch.__version__} cuda {torch.version.cuda}')" 2>/dev/null || {
    echo "  ❌ torch not found!"
    echo "  Install PyTorch for CUDA 11.8 first:"
    echo "    pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118"
    echo "  Or see docs/detectron2_install.md for the full guide."
    exit 1
}

# ── 4. Install detectron2 from local submodule ───────────────
# IMPORTANT: --no-build-isolation is required because detectron2's
# setup.py does `import torch` at build time. Without this flag,
# pip creates a temp build env without torch → build fails.
echo ""
echo "→ Installing detectron2 (--no-build-isolation)..."
cd detectron2
rm -rf build **/*.so detectron2/_C*.so 2>/dev/null || true
pip install -e . --no-build-isolation
cd ..

# ── 5. Verify installation ──────────────────────────────────
echo ""
echo "→ Verifying installation..."
python -c "
import detectron2; print(f'  ✅ detectron2 {detectron2.__version__}')
import torch; print(f'  ✅ torch {torch.__version__} cuda {torch.version.cuda}')
import numpy as np; print(f'  ✅ numpy {np.__version__}')
gpus = torch.cuda.device_count()
print(f'  ✅ GPUs: {gpus}')
if gpus > 0:
    for i in range(gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f'     GPU {i}: {name} ({mem:.0f} GB)')
" 2>/dev/null || echo "  ⚠️  Some imports failed — check your environment"

# ── 6. Create data directories ──────────────────────────────
echo ""
echo "→ Creating data directories..."
mkdir -p data/training/{images,labels,annotations}
mkdir -p data/delineation/{images,masks,delineated,temp}

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Commands:"
echo "    python scripts/train.py -c configs/train.yaml"
echo "    python scripts/infer.py -c configs/inference.yaml \\"
echo "      -i data/delineation/images/<region> \\"
echo "      -o data/delineation/delineated/<region>.gpkg"
echo "═══════════════════════════════════════════════════"
