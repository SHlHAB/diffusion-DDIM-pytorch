# Running on the UCF Newton cluster

SLURM scripts for training/sampling the DDIM model on MNIST on Newton's H100 GPUs
(`highgpu` partition).

## 1. One-time environment setup (on a login node)

```bash
git clone https://github.com/SHlHAB/diffusion-DDIM-pytorch.git
cd diffusion-DDIM-pytorch

# create the conda env the sbatch scripts activate (name must be `ddim`)
conda create -n ddim python=3.10 -y
conda activate ddim

# install a CUDA-enabled PyTorch (cu121 wheels work on the H100 nodes),
# then the rest of the deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If you prefer, swap the conda env name in `slurm/*.sbatch` (the `conda activate ddim`
line and the `LD_LIBRARY_PATH` path) for one you already have.

## 2. Train

```bash
# submit from the repo root -- $SLURM_SUBMIT_DIR must be the repo root
sbatch slurm/train.sbatch
```

- Reads `config.yml` (`device: cuda`, 50 epochs, batch 128). MNIST auto-downloads
  to `data/` on first run.
- Checkpoints save to `checkpoint/mnist.pth` every epoch, so you can resume:
  set `consume: True` in `config.yml` and resubmit.
- Logs land in `slurm/logs/ddim-mnist_<jobid>.out` / `.err`.

## 3. Sample

```bash
sbatch slurm/generate.sbatch        # writes data/result/mnist_result.png
```

## Monitor

```bash
squeue --me
tail -f slurm/logs/ddim-mnist_*.out
sinfo -p highgpu -o "%l"            # check the partition's max walltime
```

## Notes

- The code resolves the device via `utils.tools.get_device`, so `device: cuda`
  in `config.yml` auto-falls back to MPS/CPU off-cluster (e.g. a local Mac).
- An H100 is far more than this small 32x32 model needs; data loading, not
  compute, is the likely bottleneck. Raise `epochs` (cheap here) or `num_workers`
  in `config.yml` if you want.
