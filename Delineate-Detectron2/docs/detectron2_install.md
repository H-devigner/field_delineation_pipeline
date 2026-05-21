# Detectron2 Installation Guide — H100 + CUDA 11.8

Reproducible checklist for installing Detectron2 on H100 GPUs with CUDA 13 driver using PyTorch cu118.

## 1. Create Conda Environment

```bash
conda create -n detectron2_env python=3.10 -y
conda activate detectron2_env
```

## 2. Install PyTorch CUDA 11.8 + Pin NumPy

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
conda install "numpy<2" -y
```

Verify:
```bash
python -c "import numpy as np; import torch; print('numpy', np.__version__); print('torch', torch.__version__, torch.version.cuda)"
# → numpy 1.26.4, torch 2.1.0+cu118
```

## 3. Install CUDA 11.8 Toolkit in Conda

```bash
conda config --add channels nvidia
conda config --set channel_priority strict

CONDA_NO_PLUGINS=true conda install -y -c nvidia cuda-nvcc=11.8 cuda-cudart=11.8
CONDA_NO_PLUGINS=true conda install -y -c conda-forge ninja
```

Verify: `nvcc --version` → CUDA 11.8

## 4. Fix CUDA Header Paths

NVIDIA conda packages place headers under `targets/x86_64-linux/`. Create activate/deactivate scripts:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda118.sh" << 'EOF'
export _OLD_CUDA_HOME="$CUDA_HOME"
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export _OLD_CPATH="$CPATH"
export _OLD_C_INCLUDE_PATH="$C_INCLUDE_PATH"
export _OLD_CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH"
export _OLD_LIBRARY_PATH="$LIBRARY_PATH"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda118.sh" << 'EOF'
export CUDA_HOME="$_OLD_CUDA_HOME"
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
export CPATH="$_OLD_CPATH"
export C_INCLUDE_PATH="$_OLD_C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$_OLD_CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$_OLD_LIBRARY_PATH"
unset _OLD_CUDA_HOME _OLD_LD_LIBRARY_PATH _OLD_CPATH _OLD_C_INCLUDE_PATH _OLD_CPLUS_INCLUDE_PATH _OLD_LIBRARY_PATH
EOF
```

Reload:
```bash
conda deactivate && conda activate detectron2_env
which nvcc && nvcc --version
```

## 5. Install Detectron2

```bash
cd detectron2
rm -rf build **/*.so detectron2/_C*.so 2>/dev/null || true
pip install -e . --no-build-isolation
cd ..
```

> **Why `--no-build-isolation`?** Detectron2's `setup.py` does `import torch` at build time. Without this flag, pip creates a temp build env without torch → `ModuleNotFoundError`.

## 6. Verify

```bash
python -c "import detectron2, torch; print('detectron2 OK'); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

## 7. Save Environment (for reprodicibility)

```bash
conda env export --from-history > detectron2_env.yml
pip freeze > requirements-pip.txt
```
