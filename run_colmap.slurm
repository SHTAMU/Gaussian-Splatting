#!/bin/bash
#SBATCH --job-name=colmap_hw1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=colmap_%j.out

module purge
module load GCC/12.2.0 OpenMPI/4.1.4
module load COLMAP/3.8

export QT_QPA_PLATFORM=offscreen

# 加上 --no_gpu 參數，改用 CPU 處理 20 張影像特徵，避免無頭伺服器 OpenGLContextManager 崩潰
/scratch/user/shihyuyang/conda_envs/gaussian_splatting/bin/python convert.py -s data_hw1 --no_gpu