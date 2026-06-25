#!/usr/bin/env bash
#SBATCH --job-name=ddim-mnist
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=02:00:00          # 20 epochs of MNIST on an H100 is minutes; verify cap: sinfo -p highgpu -o "%l"

# Train the DDIM UNet on MNIST. Config lives in config.yml (device: cuda).
# Submit from the repo root:  sbatch slurm/train.sbatch
set -euo pipefail
mkdir -p slurm/logs
# proof-of-life BEFORE anything slow
echo "[proof-of-life] $(date) host=$(hostname) job=${SLURM_JOB_ID:-NA} cwd=$(pwd)"
# robust conda init for a NON-interactive batch shell (conda is not initialized
# unless ~/.bashrc is sourced, so 'conda activate' otherwise no-ops/hangs)
source ~/.bashrc
conda activate ddim
export PYTHONUNBUFFERED=1            # stream python stdout/stderr live (no block-buffering)
export LD_LIBRARY_PATH="$HOME/.conda/envs/ddim/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi -L 2>/dev/null || true
echo "[env] python=$(which python)"
cd "$SLURM_SUBMIT_DIR"

python train.py
echo "training done -> checkpoint/mnist.pth"
