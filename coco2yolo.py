import json
import os
from collections import defaultdict

# Paths to your filtered COCO JSON and the output directories.
coco_file = "output/coco_data/filtered_coco_annotations.json"
images_dir = "images"   # Folder where images are stored.
labels_dir = "labels"   # Folder where YOLO labels will be written.
os.makedirs(labels_dir, exist_ok=True)

# Load the COCO annotations.
with open(coco_file, "r") as f:
    data = json.load(f)

# Create a mapping from original COCO category IDs to new sequential YOLO class indices.
cat2idx = {}
for new_idx, cat in enumerate(data["categories"]):
    cat2idx[cat["id"]] = new_idx

# Build a dictionary of image info by image ID.
img_dict = {img["id"]: img for img in data["images"]}

# Group annotations by image_id.
ann_by_img = defaultdict(list)
for ann in data["annotations"]:
    ann_by_img[ann["image_id"]].append(ann)

# For each image, create a YOLO-format text file.
for image_id, anns in ann_by_img.items():
    img_info = img_dict[image_id]
    img_width = img_info["width"]
    img_height = img_info["height"]
    # Assume the file name points to the image file in images_dir.
    base_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
    label_file = os.path.join(labels_dir, base_name + ".txt")
    
    with open(label_file, "w") as f_out:
        for ann in anns:
            # COCO bounding box is [x_min, y_min, width, height]
            x_min, y_min, w, h = ann["bbox"]
            # Convert to YOLO format: normalized center coordinates and box dimensions.
            x_center = (x_min + w/2) / img_width
            y_center = (y_min + h/2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            # Get the YOLO class index.
            label = cat2idx[ann["category_id"]]
            f_out.write(f"{label} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("Conversion complete. YOLO labels are in the 'labels' directory.")
