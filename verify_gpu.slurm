#!/bin/bash
#SBATCH --job-name=verify_gpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=verify_gpu_%j.out

# 1. Load compiler, MPI, and CUDA modules
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.8.0
module load Anaconda3

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gaussian_splatting

# 2. Inspect allocated GPU hardware
nvidia-smi

# 3. Run PyTorch CUDA and environment diagnostic script
python test_env.py
