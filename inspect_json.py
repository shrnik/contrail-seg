import json

with open('data/annotations.coco.json', 'r') as f:
    coco = json.load(f)

print("First 5 images:")
for img in coco['images'][:5]:
    print(img['file_name'])

print("\nFirst 5 annotations:")
for ann in coco['annotations'][:5]:
    print(ann)
