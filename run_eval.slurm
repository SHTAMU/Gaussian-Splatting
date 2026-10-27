#!/bin/bash
#SBATCH --job-name=3dgs_eval
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=eval_%j.out

# 1. Load compiler, MPI, and CUDA runtime
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.8.0
module load Anaconda3

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gaussian_splatting

# Point Torch cache to SCRATCH to avoid quota limits in home directory
export TORCH_HOME="${SCRATCH:-$HOME}/.cache/torch"

# 2. Render novel test views from the trained model
echo "=== Step 1: Rendering test viewpoints (render.py) ==="
python render.py \
    -m output/bronze_statue

# 3. Compute quantitative metrics: PSNR, SSIM, and LPIPS
echo "=== Step 2: Computing quantitative metrics (metrics.py) ==="
python metrics.py \
    -m output/bronze_statue

echo "=== Evaluation completed successfully! ==="
