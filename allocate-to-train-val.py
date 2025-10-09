import os
import random
import shutil

# Define the source directories for images and labels
label_source_dir = "K:/DRONE/image-processing/fixedlabels"
image_source_dir = "K:/DRONE/image-processing/images"

# Define the base output directory for the dataset
output_base_dir = "K:/DRONE/YOLO-2025-lsc/dataset"

# Define the specific output directories for images and labels
train_images_dir = os.path.join(output_base_dir, "images", "train")
val_images_dir = os.path.join(output_base_dir, "images", "val")
train_labels_dir = os.path.join(output_base_dir, "labels", "train")
val_labels_dir = os.path.join(output_base_dir, "labels", "val")

# Create the output directories if they don't already exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get a list of all image and label files
all_images = [f for f in os.listdir(image_source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
all_labels = [f for f in os.listdir(label_source_dir) if f.lower().endswith('.txt')]

# --- Step 1: Handle empty and non-empty label files ---

images_with_empty_labels = []
images_with_non_empty_labels = []

for image_name in all_images:
    # Construct the corresponding label filename
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(label_source_dir, label_name)

    if os.path.exists(label_path):
        # Check if the label file is empty
        if os.path.getsize(label_path) == 0:
            images_with_empty_labels.append(image_name)
        else:
            images_with_non_empty_labels.append(image_name)

# Keep 10% of the images with empty labels
num_empty_to_keep = int(len(images_with_empty_labels) * 0.10)
selected_empty_images = random.sample(images_with_empty_labels, num_empty_to_keep)

# Combine the images with non-empty labels and the selected empty ones
final_image_list = images_with_non_empty_labels + selected_empty_images
random.shuffle(final_image_list)

# --- Step 2: Allocate images to train and validation sets ---

split_ratio = 0.8  # 80% for training
split_index = int(len(final_image_list) * split_ratio)

train_images = final_image_list[:split_index]
val_images = final_image_list[split_index:]

# --- Function to copy image and label files ---

def copy_files(image_list, image_dest_dir, label_dest_dir):
    """
    Copies a list of images and their corresponding labels to destination directories.
    """
    for image_name in image_list:
        # Define source paths for image and label
        source_image_path = os.path.join(image_source_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        source_label_path = os.path.join(label_source_dir, label_name)

        # Define destination paths
        dest_image_path = os.path.join(image_dest_dir, image_name)
        dest_label_path = os.path.join(label_dest_dir, label_name)

        # Copy the image file
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, dest_image_path)
        else:
            print(f"Warning: Image file not found - {source_image_path}")


        # Copy the corresponding label file
        if os.path.exists(source_label_path):
            shutil.copy(source_label_path, dest_label_path)
        else:
             print(f"Warning: Label file not found for image - {image_name}")


# --- Copy files to their respective folders ---

print("Copying training files...")
copy_files(train_images, train_images_dir, train_labels_dir)

print("Copying validation files...")
copy_files(val_images, val_images_dir, val_labels_dir)

print("\n--- Dataset Creation Summary ---")
print(f"Total images processed: {len(final_image_list)}")
print(f"Number of training images: {len(train_images)}")
print(f"Number of validation images: {len(val_images)}")
print("--------------------------------")
print(f"Training images and labels are in: {train_images_dir} and {train_labels_dir}")
print(f"Validation images and labels are in: {val_images_dir} and {val_labels_dir}")
print("\nScript finished successfully!")