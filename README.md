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

This repository contains an end-to-end pipeline for 3D scene reconstruction and novel view synthesis (NVS). The pipeline uses Structure-from-Motion (COLMAP) and 3D Gaussian Splatting (3DGS).

We designed this project for high-performance computing (HPC) environments. The project uses automated SLURM scripts on the TAMU Grace HPC Cluster. The cluster uses NVIDIA A100-PCIE-40GB GPUs. The pipeline reconstructs scenes from sparse outdoor images. 

The complete workflow includes:
- Feature matching and camera pose recovery
- Optimization of the 3D Gaussian scene (30,000 iterations)
- Evaluation of held-out test views
- Quantitative error measurement (PSNR, SSIM, and LPIPS).

---

## 🔬 Pipeline Architecture

```text
┌──────────────────────────┐
│  Raw Multi-View Images   │ (Outdoor Sequence: 17 Training Views + 3 Held-Out Views)
└─────────────┬────────────┘
              │
              ▼  [Stage 1: Camera Pose & Sparse Structure Recovery]
┌──────────────────────────┐
│    COLMAP SfM Engine     │ ──► SIFT Feature Extraction & Exhaustive Feature Matching
└─────────────┬────────────┘ ──► Sparse Bundle Adjustment: Camera Poses (R, t) & Point Cloud
              │
              ▼  [Stage 2: Differentiable 3D Radiance Field Optimization]
┌──────────────────────────┐
│ 3D Gaussian Splatting    │ ──► Anisotropic Covariance Parameterization & Fast Tile Rasterization
│  (30k Iters on A100)     │ ──► Adaptive Density Control (Gradient-driven Clone, Split & Prune)
└─────────────┬────────────┘ ──► View-Dependent Radiance Transfer via Spherical Harmonics (SH)
              │
              ▼  [Stage 3: Novel View Synthesis & Quantitative Metrology]
┌──────────────────────────┐
│   Held-Out Evaluation    │ ──► 1-in-8 Test Split Rendering (Strict Zero Data Leakage)
│  & Visual Diagnostics    │ ──► Quantitative Error Analysis: PSNR (21.10 dB), SSIM (0.799), LPIPS (0.207)
└──────────────────────────┘ ──► Real-Time Spatial Inspection via SIBR Interactive Viewers


---

## 🚀 Key Engineering Highlights

1. **SLURM Batch Automation (`*.slurm` scripts):**
   - We developed SLURM batch scripts (`run_colmap.slurm`, `run_train.slurm`, `run_eval.slurm`) for distributed NVIDIA A100 nodes.
   - These scripts prevent GPU out-of-memory errors and manage execution sequences automatically.

2. **Standard Evaluation Protocol (1-in-8 Split):**
   - We evaluated novel view synthesis with a standard benchmark.
   - Every eighth camera image (`images: 0, 8, 16`) was kept as a test view. This step prevents data leakage during training.

3. **System Diagnostic Tools:**
   - We created diagnostic scripts (`check_data.py`, `test_env.py`, `verify_gpu.slurm`).
   - These tools verify the CUDA environment, GPU memory state, and camera poses before optimization.

---

## 📊 Quantitative Benchmarks

We optimized the model on an outdoor sequence of a bronze statue. The dataset contains 20 images (17 training views and 3 test views).

| Metric | Result (Ours) | Target Baseline | Analysis |
| :--- | :---: | :---: | :--- |
| **PSNR (Peak Signal-to-Noise Ratio)** | **21.10 dB** | $> 20.0\text{ dB}$ | High structural reconstruction fidelity |
| **SSIM (Structural Similarity)** | **0.799** | $> 0.750$ | Correct preservation of high-frequency contours |
| **LPIPS (Perceptual Distance)** | **0.207** | $< 0.250$ | Low perceptual error (VGG perceptual network) |
| **Optimization Latency** | **~6m 28s** | ~35–45 mins | Accelerated run on NVIDIA A100 (30,000 iters) |

---

## 🖼️ Qualitative Results

### 1. Held-Out Novel View Synthesis
This section compares the held-out test view and the 3DGS rendering after 30,000 iterations:

| Ground Truth (Test View) | 3DGS Novel-View Rendering (Ours) |
| :---: | :---: |
| <img src="assets/Ground%20truth.png" width="450"/> | <img src="assets/3DGS%20Rendering.png" width="450"/> |

### 2. Global Radiance Field Geometry
The image below shows the 3D Gaussian ellipsoids in the SIBR interactive viewer:

<p align="center">
  <img src="assets/reconstructed%203D%20Gaussian%20scene.png" width="750"/>
</p>

---

## 🔍 Technical Diagnosis and Artifact Analysis

1. **Under-Constrained Ray Optimization in Sparse Baselines:**
   - *Observation:* Thin structures (such as horse ears and fine contours) show lower opacity and localized blur.
   - *Cause:* The dataset contains only 17 training images. This small number of images provides limited angular overlap. Fewer than two camera rays observe some spatial regions. This condition causes depth ambiguity during backpropagation.

2. **Needle-Like Floaters (Density Overfitting):**
   - *Observation:* The global 3D viewer shows elongated, needle-like Gaussian shapes in empty background areas.
   - *Cause:* The density control splits Gaussians based on 2D position gradients (`densify_grad_threshold = 0.0002`). Without geometric depth constraints, Gaussians expand along camera viewing rays to match training colors.

---

## 🛠️ Environment Setup & Quickstart

### 1. Environment Setup

```bash
# Clone the repository with submodules
git clone [https://github.com/SHTAMU/Gaussian-Splatting.git](https://github.com/SHTAMU/Gaussian-Splatting.git) --recursive
cd Gaussian-Splatting

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate gaussian_splatting

# Verify the GPU and CUDA environment
python test_env.py
```

### 2. SLURM HPC Execution (TAMU Grace Cluster)

```bash
# Step 1: Run COLMAP SfM pose estimation
sbatch run_colmap.slurm

# Step 2: Run 3DGS training (with 1-in-8 evaluation split)
sbatch run_train.slurm

# Step 3: Render held-out test views and compute PSNR/SSIM/LPIPS
sbatch run_eval.slurm
```

---

## 📚 References

- Kerbl, B., Kopanas, G., Leimkuehler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics (SIGGRAPH 2023).
- Schönberger, J. L., & Frahm, J. M. (2016). *Structure-from-Motion Revisited*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016).
