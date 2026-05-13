import os
import shutil

# =========================
# CLASS GROUPS
# =========================

strong_classes = [1, 6, 7, 10, 12, 14]
weak_classes = [0, 2, 3, 4, 5, 8, 9, 11,13]

# =========================
# PATHS
# =========================

base_path = "."

image_train = os.path.join(base_path, "images/train")
image_val = os.path.join(base_path, "images/val")

label_train = os.path.join(base_path, "labels/train")
label_val = os.path.join(base_path, "labels/val")

# OUTPUT DATASETS
strong_path = "dataset_strong"
weak_path = "dataset_weak"

# =========================
# CREATE FOLDERS
# =========================

for dataset in [strong_path, weak_path]:
    os.makedirs(os.path.join(dataset, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "labels/val"), exist_ok=True)

# =========================
# FUNCTION
# =========================

def process_split(image_dir, label_dir, output_base, class_list, split):
    
    for label_file in os.listdir(label_dir):

        if not label_file.endswith(".txt"):
            continue

        src_label = os.path.join(label_dir, label_file)

        filtered_lines = []

        with open(src_label, "r") as f:
            lines = f.readlines()

        for line in lines:
            class_id = int(line.split()[0])

            if class_id in class_list:
                filtered_lines.append(line)

        # skip file if no matching classes
        if not filtered_lines:
            continue

        # save filtered label
        dst_label = os.path.join(
            output_base,
            f"labels/{split}",
            label_file
        )

        with open(dst_label, "w") as f:
            f.writelines(filtered_lines)

        # copy matching image
        image_name = label_file.replace(".txt", ".jpg")

        src_image = os.path.join(image_dir, image_name)

        dst_image = os.path.join(
            output_base,
            f"images/{split}",
            image_name
        )

        if os.path.exists(src_image):
            shutil.copy(src_image, dst_image)

# =========================
# PROCESS TRAIN
# =========================

process_split(
    image_train,
    label_train,
    strong_path,
    strong_classes,
    "train"
)

process_split(
    image_train,
    label_train,
    weak_path,
    weak_classes,
    "train"
)

# =========================
# PROCESS VAL
# =========================

process_split(
    image_val,
    label_val,
    strong_path,
    strong_classes,
    "val"
)

process_split(
    image_val,
    label_val,
    weak_path,
    weak_classes,
    "val"
)

print("DONE SPLITTING DATASETS")