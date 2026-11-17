#!/usr/bin/env bash
# Step 1 setup on the GPU VM: install GVHMR (pinned) + checkpoints.
# Expects ~/body_models/{smplx,smpl}/ to have been uploaded already (licensed files, not in git).
# Run under nohup; writes ~/step1_setup.log and ~/step1_setup.done (or .failed).
set -euo pipefail
exec > >(tee -a ~/step1_setup.log) 2>&1
trap 'echo "FAILED at line $LINENO"; touch ~/step1_setup.failed' ERR

GVHMR_COMMIT=ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
HF_REPO=camenduru/GVHMR                       # mirror; Google Drive was rate-limited
HF_REV=21b32d5389e2e59c0737d4c4095bbc0b8c23f66b

echo "== $(date) system packages"
# build-essential + python3.10-dev: cython_bbox/chumpy build from source.
# python3-tk: GVHMR body_model.py has a stray "from turtle import forward" (imports tkinter).
sudo apt-get -qq update && sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg git tmux build-essential python3.10-dev python3-tk >/dev/null

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== GVHMR @ $GVHMR_COMMIT"
[ -d ~/GVHMR ] || git clone -q https://github.com/zju3dv/GVHMR.git ~/GVHMR
cd ~/GVHMR && git fetch -q && git checkout -q $GVHMR_COMMIT

echo "== Python 3.10 env (GVHMR's pytorch3d wheel is cp310 / cu121 / torch 2.3.0)"
[ -d .venv ] || uv venv -q --python 3.10 .venv
source .venv/bin/activate
# chumpy (and cython_bbox) build from source and import pip/numpy/cython at build time without
# declaring them, so install the build tools first and build without isolation.
uv pip install -q pip "setuptools>=68" wheel cython numpy==1.23.5
uv pip install -q --no-build-isolation -r requirements.txt
uv pip install -q -e . huggingface_hub

echo "== checkpoints from HF mirror ($HF_REPO @ $HF_REV), skipping DPVO (static camera)"
mkdir -p inputs/checkpoints
python - <<EOF
from huggingface_hub import hf_hub_download
for f in ["gvhmr/gvhmr_siga24_release.ckpt", "hmr2/epoch=10-step=25000.ckpt",
          "vitpose/vitpose-h-multi-coco.pth", "yolo/yolov8x.pt"]:
    hf_hub_download("$HF_REPO", f, revision="$HF_REV", local_dir="inputs/checkpoints")
EOF
cd inputs/checkpoints
sha256sum -c - <<'EOF'
4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91  gvhmr/gvhmr_siga24_release.ckpt
2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e  hmr2/epoch=10-step=25000.ckpt
50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc  vitpose/vitpose-h-multi-coco.pth
c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b  yolo/yolov8x.pt
EOF

echo "== body models"
mkdir -p body_models/smplx body_models/smpl
for g in NEUTRAL MALE FEMALE; do
  ln -sf ~/body_models/smplx/SMPLX_$g.npz body_models/smplx/SMPLX_$g.npz
  ln -sf ~/body_models/smpl/SMPL_$g.pkl  body_models/smpl/SMPL_$g.pkl
done
ls -lL body_models/*/

echo "== torch / CUDA check"
cd ~/GVHMR && python -c "import torch, pytorch3d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== $(date) setup done"
touch ~/step1_setup.done
