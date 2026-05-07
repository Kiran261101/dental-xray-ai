import os

# KEEP ONLY THESE CLASS IDS
keep_classes = [3,8,1,11,24,39,9,36,27,14,30,35,17,31,42]

label_dir = "labels/train"

for file in os.listdir(label_dir):

    if not file.endswith(".txt"):
        continue

    path = os.path.join(label_dir, file)

    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        cls_id = int(line.split()[0])

        if cls_id in keep_classes:
            new_lines.append(line)

    with open(path, "w") as f:
        f.writelines(new_lines)

print("Done filtering labels")