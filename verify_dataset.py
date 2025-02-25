import os
import glob
import cv2
import matplotlib.pyplot as plt

# Paths to your dataset directories.
images_dir = "dataset/images"
labels_dir = "dataset/labels"

# (Optional) List of class names corresponding to your YOLO dataset.
# Make sure the order matches the mapping used during conversion.
classes = ["WING_LEFT", "Drone", "ENGINE_LEFT", "forklift", "scissor lift"]

# Get a list of image files (adjust extension if needed).
image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

for image_file in image_files:
    # Load the image.
    image = cv2.imread(image_file)
    if image is None:
        continue
    height, width = image.shape[:2]
    
    # Build the corresponding label file name.
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    label_file = os.path.join(labels_dir, base_name + ".txt")
    
    # If the label file exists, read and draw each annotation.
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, box_w, box_h = parts
                class_id = int(class_id)
                x_center = float(x_center)
                y_center = float(y_center)
                box_w = float(box_w)
                box_h = float(box_h)
                
                # Convert normalized coordinates to absolute pixel values.
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                box_w_abs = box_w * width
                box_h_abs = box_h * height
                x_min = int(x_center_abs - box_w_abs / 2)
                y_min = int(y_center_abs - box_h_abs / 2)
                x_max = int(x_center_abs + box_w_abs / 2)
                y_max = int(y_center_abs + box_h_abs / 2)
                
                # Draw the bounding box.
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_text = classes[class_id] if class_id < len(classes) else str(class_id)
                cv2.putText(image, label_text, (x_min, max(y_min - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the image using matplotlib.
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image: {base_name}")
    plt.axis("off")
    plt.show()
