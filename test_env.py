import sys
import torch

print("=" * 50)
print("Python Version:", sys.version.split()[0])
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Current GPU Device:", torch.cuda.get_device_name(0))
    print("GPU Memory Total:", f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("[ERROR] PyTorch can not read GPU！")
    sys.exit(1)

try:
    import diff_gaussian_rasterization
    print("[SUCCESS] diff-gaussian-rasterization load success！")
except ImportError as e:
    print("[ERROR] diff-gaussian-rasterization load fail:", e)

try:
    import simple_knn
    print("[SUCCESS] simple-knn load success！")
except ImportError as e:
    print("[ERROR] simple-knn load fail:", e)

print("=" * 50)
print("environment success！")
