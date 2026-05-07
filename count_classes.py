import os
import yaml
from collections import Counter

# ---- PATHS (CHANGE IF NEEDED) ----
data_yaml_path = "data.yaml"
label_paths = ["labels/train", "labels/val"]

# ---- LOAD CLASS NAMES ----
with open(data_yaml_path, "r") as f:
    data = yaml.safe_load(f)

names = data["names"]  # {0: 'caries', 1: 'implant', ...}
print(type(names))

# ---- COUNTERS ----
class_counts = Counter()
image_counts = Counter()

# ---- PROCESS LABEL FILES ----
for path in label_paths:
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                lines = f.readlines()

                classes_in_image = set()

                for line in lines:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    classes_in_image.add(class_id)

                for cls in classes_in_image:
                    image_counts[cls] += 1

# ---- PRINT RESULTS ----
print("\n📊 CLASS DISTRIBUTION\n")

for cls_id in sorted(class_counts, key=class_counts.get, reverse=True):
    class_name = names[cls_id]

    instance_count = class_counts.get(cls_id, 0)
    image_count = image_counts.get(cls_id, 0)

    print(f"{cls_id:2d} | {class_name:25s} | Instances: {instance_count:5d} | Images: {image_count:4d}")