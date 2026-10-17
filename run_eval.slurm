#!/bin/bash
#SBATCH --job-name=3dgs_eval
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=eval_%j.out

# 1. 載入編譯與 CUDA 環境
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.8.0

# 讓運算節點去 Scratch 讀取 LPIPS 權重，避免再去 /home 找而報錯
export TORCH_HOME=/scratch/user/shihyuyang/.cache/torch

# 2. 渲染新視角 (renders)
echo "=== 開始執行渲染 (render.py) ==="
/scratch/user/shihyuyang/conda_envs/gaussian_splatting/bin/python render.py \
    -m output/hw1_model

# 3. 計算評估指標 PSNR, SSIM, LPIPS (metrics)
echo "=== 開始計算評估指標 (metrics.py) ==="
/scratch/user/shihyuyang/conda_envs/gaussian_splatting/bin/python metrics.py \
    -m output/hw1_model

echo "=== 評估完成！ ==="