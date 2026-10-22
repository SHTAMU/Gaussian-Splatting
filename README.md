# High-Fidelity 3D Scene Reconstruction & Evaluation Pipeline via 3D Gaussian Splatting

[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20A100%20(40GB)-green.svg)](https://www.nvidia.com)
[![Platform](https://img.shields.io/badge/HPC-TAMU%20Grace%20Cluster%20(SLURM)-blue.svg)](https://hpc.tamu.edu)
[![Framework](https://img.shields.io/badge/Method-3DGS%20%7C%20COLMAP%20SfM-orange.svg)](https://github.com/graphdeco-inria/gaussian-splatting)
[![Python](https://img.shields.io/badge/Python-3.8%2B%20%7C%20CUDA%2011.8-yellow.svg)](https://pytorch.org/)

**Author:** Shih-Yu Yang (Frank)  
**Institution:** Department of Electrical and Computer Engineering, Texas A&M University  
**Contact:** [shihyuyang@tamu.edu](mailto:shihyuyang@tamu.edu)

---

## 📌 Executive Summary

This repository presents an end-to-end, reproducible **3D Scene Reconstruction and Novel View Synthesis (NVS)** pipeline leveraging **Structure-from-Motion (COLMAP)** and **3D Gaussian Splatting (3DGS)**. 

Targeting high-performance compute environments, this project integrates automated **SLURM batch workflows** on the **TAMU Grace HPC Cluster (NVIDIA A100-PCIE-40GB)** to handle sparse outdoor multi-view imagery. The pipeline encompasses sparse feature matching, camera pose estimation, differentiable 3D Gaussian radiance field optimization (30,000 iterations), held-out view evaluation, and quantitative error diagnostics (PSNR, SSIM, LPIPS).

---

## 🔬 Pipeline Architecture

[Raw Multi-View Images]
│
▼  (Task 1: Camera Pose Estimation)
[COLMAP SfM Pipeline] ──> Feature Extraction & Exhaustive Matching ──> Sparse 3D Point Cloud & Poses
│
▼  (Task 2: Scene Splatting Optimization)
[3DGS Optimization]   ──> Adaptive Density Control & Spherical Harmonics (30k Iterations on A100)
│
▼  (Task 3: Quantitative & Qualitative Evaluation)
[Held-out Evaluation] ──> 1-in-8 Test Split (PSNR / SSIM / LPIPS) + SIBR Interactive Viewer Inspection


---

## 🚀 Key Engineering & Implementation Highlights

1. **HPC SLURM Automation (`*.slurm` scripts):**
   - Engineered scalable SLURM execution scripts (`run_colmap.slurm`, `run_train.slurm`, `run_eval.slurm`) tailored for distributed A100 compute nodes, preventing VRAM out-of-memory spikes and orchestrating automated dependency chains.
2. **Strict 1-in-8 Evaluation Protocol:**
   - Evaluated under standard novel view synthesis benchmarks: every 8th camera view (`images: 0, 8, 16`) was held out as unseen ground truth, ensuring zero-leakage test metrics.
3. **Reproducibility & Verification Tools:**
   - Built custom diagnostic utilities (`check_data.py`, `test_env.py`, `verify_gpu.slurm`) to validate CUDA runtime bindings, tensor memory allocation, and COLMAP undistorted image-camera pose correspondences prior to optimization.

---

## 📊 Quantitative Benchmarks & Results

Optimization was executed on a real-world outdoor bronze statue sequence (20 multi-view images total: 17 training, 3 test views).

| Metric | Result (Ours) | Baseline Expectation | Interpretation |
| :--- | :---: | :---: | :--- |
| **PSNR (Peak Signal-to-Noise Ratio)** | **21.10 dB** | $> 20.0\text{ dB}$ | High structural reconstruction fidelity |
| **SSIM (Structural Similarity)** | **0.799** | $> 0.750$ | Preserves high-frequency spatial context |
| **LPIPS (Perceptual Distance)** | **0.207** | $< 0.250$ | Low perceptual error (VGG perceptual metric) |
| **Training Latency** | **~6m 28s** | ~30–45 mins (standard GPU) | **30,000 iters accelerated on NVIDIA A100** |

---

## 🖼️ Qualitative Results & Novel View Synthesis

### 1. Held-Out Novel View Synthesis (Test Camera)
Comparison between the held-out ground truth view and the novel view synthesized by 3DGS (30,000 iterations):

| Ground Truth (Test View) | 3DGS Novel-View Rendering (Ours) |
| :---: | :---: |
| <img src="assets/Ground%20truth.png" width="450"/> | <img src="assets/3DGS%20Rendering.png" width="450"/> |

### 2. Global Radiance Field Geometry
Visual inspection of the reconstructed 3D Gaussian scene (ellipsoids) via the SIBR interactive viewer:

<p align="center">
  <img src="assets/reconstructed%203D%20Gaussian%20scene.png" width="750"/>
</p>

---

## 🔍 Technical Diagnosis & Failure Mode Analysis

While the pipeline achieved strong overall structural fidelity, critical artifacts were observed and technically diagnosed:

1. **Under-Constrained Ray Optimization under Sparse Baseline:**
   - *Observation:* Thin structures (e.g., horse ears/head edges) exhibit partial opacity loss and slight geometric blur.
   - *Root Cause:* Training with only 17 images provides insufficient angular overlap for wide-baseline parallax. Certain spatial voxels are intersected by fewer than 2 camera rays, leading to depth ambiguity during gradient backpropagation.
2. **Needle-like Floaters (Spatial Anisotropy Distortion):**
   - *Observation:* Inspecting the global 3D viewer reveals elongated, needle-like Gaussian ellipsoids stretching into empty background regions.
   - *Root Cause:* The 3DGS adaptive density control mechanism splits and clones Gaussians based on 2D view-space positional gradients (`densify_grad_threshold = 0.0002`). Without multi-view geometric depth regularization, Gaussians overfit to match training view colors by extending along null-space viewing frustums.

---

## 🛠️ Environment Setup & Quickstart

### 1. Environment Initialization

```bash
# Clone repository recursively with submodules
git clone https://github.com/SHTAMU/Gaussian-Splatting.git --recursive
cd Gaussian-Splatting

# Setup Conda environment
conda env create -f environment.yml
conda activate gaussian_splatting

# Validate GPU & CUDA runtime
python test_env.py


### 2. SLURM HPC Execution (TAMU Grace Cluster)
# Step 1: Run COLMAP SfM pose estimation
sbatch run_colmap.slurm

# Step 2: Run 3DGS training (with 1-in-8 evaluation split)
sbatch run_train.slurm

# Step 3: Render held-out test views and compute PSNR/SSIM/LPIPS
sbatch run_eval.slurm

📚 References
Kerbl, B., Kopanas, G., Leimkuehler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics (SIGGRAPH 2023).

Schönberger, J. L., & Frahm, J. M. (2016). Structure-from-Motion Revisited. IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016).
