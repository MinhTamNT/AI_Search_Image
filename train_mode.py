import os
import shutil
import random

# Cấu hình
input_folder = "flickr30k_images/flickr30k_images"  # Thư mục chứa tất cả ảnh
output_folder = "dataset"  # Thư mục đích chứa train/ và val/
val_split = 0.2  # Tỷ lệ ảnh validation (20%)

train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)

for img_file in image_files:

    class_label = img_file.split('_')[0]

    train_class_folder = os.path.join(train_folder, class_label)
    val_class_folder = os.path.join(val_folder, class_label)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(val_class_folder, exist_ok=True)

    # Phân loại ảnh vào train hoặc val
    if random.random() < val_split:
        shutil.move(os.path.join(input_folder, img_file), os.path.join(val_class_folder, img_file))
    else:
        shutil.move(os.path.join(input_folder, img_file), os.path.join(train_class_folder, img_file))

print("🎉 Hoàn thành! Dữ liệu đã được chia vào train/ và val/")
