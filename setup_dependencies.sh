#!/usr/bin/env bash
# ------------------------------------------------------------------
# install-cuda-venv.sh  –  One-shot CUDA + Python-3.10 venv setup
# ------------------------------------------------------------------
set -euo pipefail

# === ---------- configurable bits ---------- ===
CUDA_VERSION=12.3           # “12.3”, “11.8”, …  (major.minor)
CUDA_PATCH=0                # “0”, “1” … (the sub-revision)
TARGET_PY=python3.10        # must already be on PATH; apt install python3.10
VENV_DIR=$HOME/py310-cuda   # where venv will live
REQ_FILE=$PWD/packages_py.txt
# === -------------------------------------- ===

echo "==> Installing dependencies …"
sudo apt-get update -qq
sudo apt-get install -y build-essential dkms curl gnupg lsb-release ${TARGET_PY} ${TARGET_PY}-venv

# ---------------------------------------------------------------
# 1) CUDA toolkit – repo pin + install
# ---------------------------------------------------------------
echo "==> Registering NVIDIA GPG key and apt repo for CUDA ${CUDA_VERSION}"
CUDA_DEB="cuda-keyring_1.1-1_all.deb"
wget -qO /tmp/${CUDA_DEB} https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/${CUDA_DEB}
sudo dpkg -i /tmp/${CUDA_DEB}

echo "==> Installing cuda-toolkit-${CUDA_VERSION}"
sudo apt-get update -qq
sudo apt-get install -y cuda-toolkit-${CUDA_VERSION//./-}

# (Optional) pin the driver to the same branch:
# sudo apt-get install -y cuda-drivers-${CUDA_VERSION%.*}

# ---------------------------------------------------------------
# 2) Environment variables
# ---------------------------------------------------------------
if ! grep -q 'export PATH=.*cuda' ~/.bashrc; then
  echo '### CUDA paths (added by install-cuda-venv.sh)' >> ~/.bashrc
  echo "export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH" >> ~/.bashrc
  echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH

# ---------------------------------------------------------------
# 3) Python 3.10 virtual environment
# ---------------------------------------------------------------
echo "==> Creating venv at ${VENV_DIR}"
${TARGET_PY} -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
pip install -r "${REQ_FILE}"

echo
echo "✅  All done!"
echo "➡️  To activate later:   source ${VENV_DIR}/bin/activate"
echo "➡️  Verify CUDA:         nvcc --version   or   nvidia-smi"
