import os
import random
import shutil

# Set your source folders and destination folders
image_folder = 'C:/Users/ramdu/Downloads/aam BG'
label_folder = 'C:/Users/ramdu/Downloads/aam labels'  # Add this for .txt files
train_folder = 'C:/Users/ramdu/Downloads/train'
test_folder = 'C:/Users/ramdu/Downloads/test'

# Make sure destination folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all image files (you can adjust extensions if needed)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

# Shuffle images randomly
random.shuffle(images)

# Calculate the split index
split_index = int(len(images) * 0.8)

# Move images and corresponding label files
for i, img in enumerate(images):
    src_img_path = os.path.join(image_folder, img)
    txt_filename = os.path.splitext(img)[0] + '.txt'
    src_txt_path = os.path.join(label_folder, txt_filename)

    if i < split_index:
        dst_img_folder = os.path.join(os.path.dirname(train_folder), "images", os.path.basename(train_folder))
        dst_txt_folder = os.path.join(os.path.dirname(train_folder), "labels", os.path.basename(train_folder))
        os.makedirs(dst_img_folder, exist_ok=True)
        os.makedirs(dst_txt_folder, exist_ok=True)
        dst_img_path = os.path.join(dst_img_folder, img)
        dst_txt_path = os.path.join(train_folder, txt_filename)
    else:
        dst_img_folder = os.path.join(os.path.dirname(test_folder), "images", os.path.basename(test_folder))
        dst_txt_folder = os.path.join(os.path.dirname(test_folder), "labels", os.path.basename(test_folder))
        os.makedirs(dst_img_folder, exist_ok=True)
        os.makedirs(dst_txt_folder, exist_ok=True)
        dst_img_path = os.path.join(test_folder, img)
        dst_txt_path = os.path.join(test_folder, txt_filename)

    # Move image
    shutil.copy(src_img_path, dst_img_path)

    # Move corresponding .txt file if it exists
    if os.path.exists(src_txt_path):
        shutil.copy(src_txt_path, dst_txt_path)
    else:
        print(f"Warning: Label file not found for {img}")

print(f"Moved {split_index} images and their labels to {train_folder} and {len(images) - split_index} to {test_folder}.")
