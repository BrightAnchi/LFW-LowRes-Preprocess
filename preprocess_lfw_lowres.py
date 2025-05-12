
# 📁 Step 1: 掛載 LFW 資料集（Google Drive）
from google.colab import drive
import os
import cv2
import zipfile
import urllib.request
from tqdm import tqdm
from pathlib import Path

!mkdir -p /content/datasets

LFW_ORIGINAL_PATH = "/content/drive/MyDrive/ColabNotebooksDBS/LabelledFacesintheWildDataset/lfw-deepfunneled123"

!rm -rf /content/datasets/lfw-deepfunneled
!ln -s "$LFW_ORIGINAL_PATH" /content/datasets/lfw-deepfunneled
!ls /content/datasets/lfw-deepfunneled | head -10

# 📁 Step 2: 處理成低解析人臉影像
import cv2
import shutil
import os
from tqdm import tqdm

src_dir = "/content/datasets/lfw-deepfunneled"
dst_orig_dir = "/content/datasets/LFW_original"
dst_lowres_dir = "/content/datasets/LFW_lowres"
target_lowres_size = (16, 16)

os.makedirs(dst_orig_dir, exist_ok=True)
os.makedirs(dst_lowres_dir, exist_ok=True)

success_count = 0
fail_count = 0

for person_name in tqdm(os.listdir(src_dir), desc="處理人物資料夾"):
    person_path = os.path.join(src_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    for filename in os.listdir(person_path):
        if not filename.lower().endswith('.jpg'):
            continue

        src_img_path = os.path.join(person_path, filename)
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"⚠️ 無法讀取圖片: {src_img_path}")
            fail_count += 1
            continue

        dst_orig_subdir = os.path.join(dst_orig_dir, person_name)
        dst_lowres_subdir = os.path.join(dst_lowres_dir, person_name)
        os.makedirs(dst_orig_subdir, exist_ok=True)
        os.makedirs(dst_lowres_subdir, exist_ok=True)

        shutil.copy(src_img_path, os.path.join(dst_orig_subdir, filename))

        h, w = img.shape[:2]
        lowres_img = cv2.resize(img, target_lowres_size, interpolation=cv2.INTER_AREA)
        lowres_img = cv2.resize(lowres_img, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(dst_lowres_subdir, filename), lowres_img)
        success_count += 1

print(f"✅ 成功處理: {success_count} 張")
print(f"❌ 失敗處理: {fail_count} 張")
