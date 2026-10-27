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

# Run COLMAP conversion with CPU-based feature extraction to avoid OpenGLContextManager crashes
python convert.py -s data --no_gpu
