import os
import shutil
import json
from pathlib import Path
import hashlib
from tqdm import tqdm

# ✅ Paths (use Pathlib for clarity and safety)
base_dir = Path('C:/Users/priya/Downloads/archive (1)/2020-02-14_InfraredSolarModules/InfraredSolarModules')
images_dir = base_dir / 'images'
metadata_path = base_dir / 'module_metadata.json'
output_dir = Path('D:/Solar Panel Research/sorted_images')  # ✅ Updated to match current location
output_dir.mkdir(parents=True, exist_ok=True)

# ✅ Load metadata
with metadata_path.open('r') as f:
    metadata = json.load(f)

# ✅ Define classes and create output dirs
classes = sorted(set(entry['anomaly_class'] for entry in metadata.values()))
phases = ['train', 'validation', 'test']
for phase in phases:
    for class_name in classes:
        (output_dir / phase / class_name).mkdir(parents=True, exist_ok=True)

# ✅ Hash-based reproducible split
def get_split(uniq_id):
    hash_val = int(hashlib.md5(uniq_id.encode()).hexdigest(), 16)
    ratio = hash_val % 10
    if ratio < 8:
        return 'train'
    elif ratio == 8:
        return 'validation'
    else:
        return 'test'

# ✅ Process all images with progress bar
print("📂 Sorting images by anomaly class...")
for image_id, data in tqdm(metadata.items(), desc="Processing images"):
    class_name = data['anomaly_class']
    image_name = Path(data['image_filepath']).name
    src_path = images_dir / image_name

    if src_path.exists():
        phase = get_split(image_id)
        dest_path = output_dir / phase / class_name / image_name
        shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
    else:
        print(f"⚠️ File not found: {src_path}")

print("✅ Image sorting complete.")
