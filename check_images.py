from PIL import Image
import os

folders = [
    "dataset_weak/images/train",
    "dataset_weak/images/val"
]

bad_images = []

for folder in folders:
    for file in os.listdir(folder):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(folder, file)

            try:
                img = Image.open(path)
                img.verify()

                img = Image.open(path)
                img.load()

            except Exception as e:
                print(f"CORRUPT: {path}")
                print(f"  Error: {e}")
                bad_images.append(path)

print("\nDONE")
print(f"Bad images found: {len(bad_images)}")