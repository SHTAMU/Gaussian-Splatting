import os
from PIL import Image

scene_path = "data_hw1"
input_dir = os.path.join(scene_path, "input")

if not os.path.exists(input_dir):
    print(f"[ERROR] 找不到資料夾: {input_dir}")
    exit(1)

valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
image_files = [f for f in os.listdir(input_dir) if f.endswith(valid_extensions)]

print("=" * 50)
print(f"場景目錄: {os.path.abspath(scene_path)}")
print(f"偵測到的影像總數: {len(image_files)} 張")

if len(image_files) == 0:
    print("[ERROR] input 資料夾內沒有支援的影像！請確認解壓路徑。")
    exit(1)

print("\n正在驗證前 3 張影像可讀性與解析度...")
for i, img_name in enumerate(sorted(image_files)[:3]):
    img_path = os.path.join(input_dir, img_name)
    with Image.open(img_path) as img:
        print(f"  [{i+1}] {img_name} - 尺寸 (寬x高): {img.size}, 色彩模式: {img.mode}")

print("=" * 50)
print("資料檢查完成！已具備執行 Task 2（COLMAP 相機姿態估算）的條件。")