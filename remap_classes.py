import os

# OLD CLASS IDs -> NEW CLASS IDs
class_mapping = {
    3: 0,    # BONE LOSS
    8: 1,    # CROWDING
    1: 2,    # ATTRITION
    11: 3,   # DENTAL CARIES
    24: 4,   # MISSING TOOTH
    39: 5,   # SPACING
    9: 6,    # CROWNS
    36: 7,   # ROTATED
    27: 8,   # PERIAPICAL INFECTION
    14: 9,   # ERUPTING TOOTH
    30: 10,  # RCT TOOTH
    35: 11,  # ROOT STUMP
    17: 12,  # IMPACTED TOOTH
    31: 13,  # RESTORATIONS
    42: 14   # SUPRAERUPTED TOOTH
}

folders = [
    "labels/train",
    "labels/val"
]

for label_dir in folders:

    if not os.path.exists(label_dir):
        continue

    for file in os.listdir(label_dir):

        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:

            parts = line.strip().split()

            old_class_id = int(parts[0])

            if old_class_id in class_mapping:

                new_class_id = class_mapping[old_class_id]

                parts[0] = str(new_class_id)

                new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

print("Class remapping completed.")