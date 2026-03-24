# multimodal-MoE

Multimodal Mixture-of-Experts for 3D object detection, combining BEVFusion
(LiDAR-only and camera+LiDAR) with the Zenseact Open Dataset (ZOD).

## Environment setup

**Prerequisites**: Linux, CUDA 12.1 compatible GPU, conda (Miniconda/Miniforge).

### Option A — conda (recommended)

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate multimodal-moe

# 2. Install mmcv with pre-built CUDA ops (must match torch + CUDA versions)
mim install mmcv==2.2.0

# 3. Clone and install the patched mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout fe25f7a5                          # v1.4.0 base
git apply /path/to/multimodal-MoE/third_party/mmdetection3d_thesis.patch
pip install -e .
cd ..
```

### Option B — pip only

```bash
# 1. Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv && source .venv/bin/activate

# 2. Install requirements (PyTorch CUDA 12.1 wheels)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
            -r requirements.txt

# 3. Install mmcv (pre-built CUDA wheels, not pip-installable directly)
mim install mmcv==2.2.0

# 4. Clone and install patched mmdetection3d (same as Option A step 3)
```

### Verify installation

```bash
python -c "
import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
import mmcv;  print('mmcv:', mmcv.__version__)
import mmdet; print('mmdet:', mmdet.__version__)
import mmdet3d; print('mmdet3d:', mmdet3d.__version__)
"
```

Expected output:
```
PyTorch: 2.4.0+cu121 | CUDA: 12.1
mmcv: 2.2.0
mmdet: 3.3.0
mmdet3d: 1.4.0
```

### Important notes

- **numpy must be <2.0** (pinned to 1.26.4). mmcv and mmdet3d are not compatible
  with numpy 2.x.
- **mmcv** cannot be installed via plain `pip install`. Use `mim install` which
  fetches the correct pre-built wheel for your torch + CUDA combination.
- **mmdet3d** is installed from a local clone with thesis-specific patches. See
  `third_party/mmdetection3d_changes.md` for the full list of modifications.
- **PYTHONPATH**: When running mmdetection3d tools directly, set
  `PYTHONPATH=/path/to/mmdetection3d` so that `projects.BEVFusion` is importable.
- **Cluster deployment (MeluXina / A100)**: The same versions work on A100 GPUs.
  If the cluster uses a different CUDA toolkit (e.g., 11.8), replace `cu121` with
  `cu118` in the PyTorch index URL and ensure `mim install mmcv==2.2.0` picks up
  the matching wheel. Headless nodes may need `opencv-python-headless` instead of
  `opencv-python`.