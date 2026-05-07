import json
import os

# paths
json_path = "Annotations/exported_json.json"
output_dir = "labels"
image_dir = "images"

os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(output_dir):
    path = os.path.join(output_dir, f)

    if os.path.isfile(path):
        os.remove(path)
    else:
        print("Skipping folder:",path)

with open(json_path) as f:
    data = json.load(f)

# fix wrong class names
fix_map = {
    "MEATAL FRAMES": "METAL FRAMES",
    "STEEL LIGASTURES": "STEEL LIGATURES"
    
}

cleaned = set()

for cat in data["categories"]:
    name = cat["name"].strip().upper().replace("|"," ")
    name = " ".join(name.split())

    if name in fix_map:
        name = fix_map[name]

    cleaned.add(name)

print("\nCLEANED CLASSES:", len(cleaned))
for i,name in enumerate(sorted(cleaned)):
    print(i, name)

unique_names = sorted(list(set(cleaned)))

name_to_id = {name: i for i, name in enumerate(unique_names)}





for i, name in enumerate(unique_names):
    print(i, name)


        
    

images = {img["id"]: img for img in data["images"]}


annotations = data["annotations"]

for ann in annotations:
    image_id = ann["image_id"]
    img = images[image_id]

    file_name = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]

    
    x, y, w, h = ann["bbox"]

    # convert to YOLO format
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height

    orig_name = next(cat["name"] for cat in data["categories"] if cat["id"] == ann["category_id"])
    
    orig_name = orig_name.strip().upper().replace("|"," ")
    orig_name = " ".join(orig_name.split())

    if orig_name in fix_map:
        orig_name = fix_map[orig_name]

    if orig_name not in name_to_id:
        print("ERROR CLASS:", orig_name, "FILE:", file_name)
        continue

    class_id = name_to_id[orig_name]

    txt_name = os.path.splitext(file_name)[0] + ".txt"
    txt_file = os.path.join(output_dir, txt_name)

    
    with open(txt_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6F} {w:.6f} {h:.6f}\n")

import os

print("\n=== DEBUG CHECK ===")
print("Total label files created:", len(os.listdir(output_dir)))

import os
print("CURRENT DIR:", os.getcwd())
print("FOLDER HERE:", os.listdir())
import os

print("CHECK images folder:", os.listdir("images"))   # 👈 ADD THIS

img_dir = "images"
label_dir = output_dir
img_dir = "images"   # change if needed
label_dir = output_dir

images = set(f.split('.')[0] for f in os.listdir(img_dir))
labels = set(f.split('.')[0] for f in os.listdir(label_dir))

missing = images - labels

print("Missing labels count:", len(missing))

print("\nSome missing files:")
for name in list(missing)[:20]:
    print(name)
for name in missing:
    open(os.path.join(label_dir, name + ".txt"), "w").close()

missing = images - labels

print("Missing labels count:", len(missing))

print("\nSome missing files:")
for name in list(missing)[:20]:
    print(name)

# ✅ create empty labels
for name in missing:
    open(os.path.join(label_dir, name + ".txt"), "w").close()

# ✅ NOW recompute (THIS is what you were missing)
labels = set(f.split('.')[0] for f in os.listdir(label_dir))
missing = images - labels

print("\nAFTER FIX:")
print("Missing labels count:", len(missing))



import shutil

def clean(name):
    return name.replace(" ", "").replace("_", "").lower()

train_images = {clean(f.replace(".jpg", "")): f for f in os.listdir("images/train")}
val_images   = {clean(f.replace(".jpg", "")): f for f in os.listdir("images/val")}

for txt in os.listdir("labels"):
    if not txt.endswith(".txt"):
        continue

    raw_name = txt.replace(".txt", "")
    key = clean(raw_name)

    if key in train_images:
        shutil.move(f"labels/{txt}", f"labels/train/{txt}")

    elif key in val_images:
        shutil.move(f"labels/{txt}", f"labels/val/{txt}")
    
# ---- your existing code above ----


# ✅ CHECK COUNTS
import os

print("\n=== FINAL COUNT CHECK ===")
print("Train images:", len(os.listdir("images/train")))
print("Train labels:", len(os.listdir("labels/train")))

print("\n=== EMPTY LABEL CHECK ===")

empty = 0
for f in os.listdir("labels/train"):
    path = f"labels/train/{f}"
    if os.path.getsize(path) == 0:
        empty += 1
        print("Empty:", f)

print("Total empty:", empty)

print("\n=== VAL CHECK ===")
print("Val images:", len(os.listdir("images/val")))
print("Val labels:", len(os.listdir("labels/val")))

for f in os.listdir("labels/train"):
    path = f"labels/train/{f}"
    if os.path.getsize(path) == 0:
        os.remove(path)

for f in os.listdir("images/train"):
    name = f.replace(".jpg", "")
    if not os.path.exists(f"labels/train/{name}.txt"):
        os.remove(f"images/train/{f}")

print("Train images:", len(os.listdir("images/train")))
print("Train labels:", len(os.listdir("labels/train")))

print(len(os.listdir("images/train")))
print(len(os.listdir("labels/train")))

img_files = set(f.replace(".jpg", "") for f in os.listdir("images/train"))
lbl_files = set(f.replace(".txt", "") for f in os.listdir("labels/train"))

extra_labels = lbl_files - img_files

print("Extra labels:", extra_labels)

for name in extra_labels:
    path = f"labels/train/{name}.txt"
    if os.path.exists(path):
        os.remove(path)
        print("Deleted:", name)

print(len(os.listdir("images/train")))
print(len(os.listdir("labels/train")))