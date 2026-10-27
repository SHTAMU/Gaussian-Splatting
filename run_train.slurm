#!/bin/bash
#SBATCH --job-name=3dgs_train
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=train_%j.out

# 1. Load compiler modules and CUDA runtime
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.8.0
module load Anaconda3

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gaussian_splatting

# 2. Check allocated GPU resources
nvidia-smi

# 3. Train 3D Gaussian Splatting model
# -s: source dataset directory
# -m: output directory for trained checkpoints and point cloud PLY
# --eval: enable standard 1-in-8 train/test split evaluation protocol
python train.py \
    -s data \
    -m output/bronze_statue \
    --eval
