import json

# Specify the category names you want to keep.
# Change these to the categories you want.
desired_categories = ["WING_LEFT", "Drone", "ENGINE_LEFT", "forklift", "scissor lift"]

# Load the original COCO annotations file.
with open("output/coco_data/coco_annotations.json", "r") as f:
    data = json.load(f)

# Filter categories to include only those in desired_categories.
filtered_categories = [cat for cat in data["categories"] if cat["name"] in desired_categories]

# Get a set of allowed category IDs.
allowed_cat_ids = {cat["id"] for cat in filtered_categories}

# Filter annotations to include only those with an allowed category_id.
filtered_annotations = [ann for ann in data["annotations"] if ann["category_id"] in allowed_cat_ids]

# Optionally, filter images: keep only images that have at least one annotation.
image_ids = {ann["image_id"] for ann in filtered_annotations}
filtered_images = [img for img in data["images"] if img["id"] in image_ids]

# Construct the new filtered COCO dataset.
filtered_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "categories": filtered_categories,
    "images": filtered_images,
    "annotations": filtered_annotations
}

# Write the filtered data to a new JSON file.
with open("output/coco_data/filtered_coco_annotations.json", "w") as f:
    json.dump(filtered_data, f, indent=4)

print("Filtered annotations saved to filtered_coco_annotations.json")
