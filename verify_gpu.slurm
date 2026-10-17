#!/bin/bash
#SBATCH --job-name=verify_gpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=verify_gpu_%j.out

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.8.0

# 1. 顯示分配到的 GPU 硬體資訊
nvidia-smi

# 2. 使用虛擬環境中的 Python 絕對路徑直接執行
/scratch/user/shihyuyang/conda_envs/gaussian_splatting/bin/python test_env.py