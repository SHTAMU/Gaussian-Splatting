import os
from PIL import Image

scene_path = "data_hw1"
input_dir = os.path.join(scene_path, "input")

if not os.path.exists(input_dir):
    print(f"[ERROR] can not find folder: {input_dir}")
    exit(1)

valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
image_files = [f for f in os.listdir(input_dir) if f.endswith(valid_extensions)]

print("=" * 50)
print(f"scene table: {os.path.abspath(scene_path)}")
print(f"detected photo number: {len(image_files)} 張")

if len(image_files) == 0:
    print("[ERROR] input folder no support photo！check unzip path。")
    exit(1)

print("\nverify first 3 phtot readibility and resolution...")
for i, img_name in enumerate(sorted(image_files)[:3]):
    img_path = os.path.join(input_dir, img_name)
    with Image.open(img_path) as img:
        print(f"  [{i+1}] {img_name} - size (wxh): {img.size}, color mode: {img.mode}")

print("=" * 50)
print("data check complete！can run Task 2（COLMAP）")
