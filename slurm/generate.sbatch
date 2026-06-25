#!/usr/bin/env bash
#SBATCH --job-name=ddim-sample
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=00:30:00

# Sample a grid of digits from a trained checkpoint.
# Submit from the repo root (after training):  sbatch slurm/generate.sbatch
set -euo pipefail
mkdir -p slurm/logs data/result
echo "[proof-of-life] $(date) host=$(hostname) job=${SLURM_JOB_ID:-NA} cwd=$(pwd)"
source ~/.bashrc
conda activate ddim
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/.conda/envs/ddim/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi -L 2>/dev/null || true
echo "[env] python=$(which python)"
cd "$SLURM_SUBMIT_DIR"

python generate.py -cp "checkpoint/mnist.pth" --device cuda --sampler ddim --steps 50 \
    -bs 256 --nrow 16 --result_only -sp "data/result/mnist_result.png" --to_grayscale
echo "wrote data/result/mnist_result.png"
