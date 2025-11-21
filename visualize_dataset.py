import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors

# Load COCO annotations
with open('data/annotations.coco.json', 'r') as f:
    coco = json.load(f)

# Get random 10 images
images = random.sample(coco['images'], min(10, len(coco['images'])))

# Create image_id to annotations mapping
img_to_anns = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    if img_id not in img_to_anns:
        img_to_anns[img_id] = []
    img_to_anns[img_id].append(ann)

# Create category_id to name mapping
cat_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

colors = list(mcolors.BASE_COLORS.values())
# Visualize
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
# pick a random sample of images
random_sample = random.sample(images, len(axes))

for idx, img_info in enumerate(random_sample):
    # Load image
    try :
        img_path = f"data/{img_info['file_name']}"
        print(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e :
        print(f"Error loading image {img_info['file_name']}: {e}")
        continue

    # Draw segmentations
    anns = img_to_anns.get(img_info['id'], [])
    overlay = img.copy()

    for ann in anns:
        cat_id = ann['category_id']
        color = colors[cat_id % len(colors)]

        # Draw segmentation polygon
        if 'segmentation' in ann:
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], True, color, 2)

    # Blend overlay
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    # Display
    axes[idx].imshow(img)
    axes[idx].set_title(f"{img_info['file_name']}\n{len(anns)} annotations", fontsize=8)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization to dataset_visualization.png")
print(f"\nCategories: {list(cat_to_name.values())}")
print(f"Total images: {len(coco['images'])}")
print(f"Total annotations: {len(coco['annotations'])}")
plt.show()
