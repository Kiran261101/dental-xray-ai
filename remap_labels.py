import os

# =========================
# REMAP DICTIONARIES
# =========================

strong_remap = {
    1: 0,
    6: 1,
    7: 2,
    10: 3,
    12: 4,
    14: 5
}

weak_remap = {
    0: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    8: 5,
    9: 6,
    11: 7,
    13: 8
}

# =========================
# FUNCTION
# =========================

def remap_dataset(label_dir, remap_dict):

    for file_name in os.listdir(label_dir):

        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(label_dir, file_name)

        new_lines = []

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            old_class = int(parts[0])

            new_class = remap_dict[old_class]

            parts[0] = str(new_class)

            new_lines.append(" ".join(parts) + "\n")

        with open(file_path, "w") as f:
            f.writelines(new_lines)

# =========================
# RUN STRONG
# =========================

remap_dataset(
    "dataset_strong/labels/train",
    strong_remap
)

remap_dataset(
    "dataset_strong/labels/val",
    strong_remap
)

# =========================
# RUN WEAK
# =========================

remap_dataset(
    "dataset_weak/labels/train",
    weak_remap
)

remap_dataset(
    "dataset_weak/labels/val",
    weak_remap
)

print("DONE REMAPPING")