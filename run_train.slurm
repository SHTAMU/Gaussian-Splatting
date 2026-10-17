#!/bin/bash
#SBATCH --job-name=3dgs_train
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=train_%j.out

# 1. 載入編譯環境與 CUDA
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.8.0

# 2. 顯示分配到的 GPU
nvidia-smi

# 3. 執行 3D Gaussian Splatting 訓練
# -s 指定資料集目錄
# -m 指定訓練權重與點雲的輸出目錄
# 預設會迭代 30,000 步（20 張影像在 A100 約只需 10~15 分鐘）
/scratch/user/shihyuyang/conda_envs/gaussian_splatting/bin/python train.py \
    -s data_hw1 \
    -m output/hw1_model \
    --eval