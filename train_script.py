
import ultralytics
from ultralytics import YOLO
import wandb
from datetime import datetime


wandb.login()
DATA_DIR = "./data"


import json
import os
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(json_path, source_img_dir, output_dir, val_split=0.2):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Create directories
    output_dir = Path(output_dir)
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # Map categories to indices (0-indexed)
    # Ensure consistent ordering
    categories = sorted(coco['categories'], key=lambda x: x['id'])
    cat_id_to_idx = {cat['id']: i for i, cat in enumerate(categories)}
    cat_names = [cat['name'] for cat in categories]

    print(f"Categories mapping: {cat_id_to_idx}")
    print(f"Category names: {cat_names}")

    # Group annotations by image
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Split images
    images = coco['images']
    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_split))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def process_images(image_list, split_name):
        for img_info in tqdm(image_list, desc=f"Processing {split_name}"):
            img_id = img_info['id']
            file_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']

            # Copy image
            src_path = Path(source_img_dir) / file_name
            if not src_path.exists():
                # Try adding underscore after 8th char (date)
                if len(file_name) > 8 and file_name[8] != '_':
                    alt_name = file_name[:8] + '_' + file_name[8:]
                    src_path_alt = Path(source_img_dir) / alt_name
                    if src_path_alt.exists():
                        src_path = src_path_alt
                        # print(f"Fixed filename: {file_name} -> {alt_name}")

            dst_path = output_dir / 'images' / split_name / file_name # Keep original name in dest or use fixed?
            # Better to use the fixed name in dest so it matches the label file which is derived from file_name
            # But wait, the label file is derived from file_name (from JSON).
            # If I change the image name on disk, I should also change the label filename.
            # Let's use the found src filename as the basis for dest filename to ensure consistency.

            if src_path.exists():
                dst_path = output_dir / 'images' / split_name / src_path.name
                shutil.copy(src_path, dst_path)
                # Update file_name to match the one on disk for label generation
                file_name = src_path.name
            else:
                print(f"Warning: Image {file_name} not found (tried {src_path}).")
                continue

            # Create label file
            label_path = output_dir / 'labels' / split_name / f"{Path(file_name).stem}.txt"

            anns = img_to_anns.get(img_id, [])
            with open(label_path, 'w') as f:
                for ann in anns:
                    if 'segmentation' not in ann:
                        continue

                    cat_idx = cat_id_to_idx[ann['category_id']]

                    for seg in ann['segmentation']:
                        # seg is a list of coordinates [x1, y1, x2, y2, ...]
                        # Normalize
                        points = np.array(seg).reshape(-1, 2).astype(float)
                        points[:, 0] /= width
                        points[:, 1] /= height

                        # Clip to [0, 1] just in case
                        points = np.clip(points, 0, 1)

                        # Format: class x1 y1 x2 y2 ...
                        line = f"{cat_idx} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
                        f.write(line + "\n")

    process_images(train_images, 'train')
    process_images(val_images, 'val')

    # Create data.yaml
    yaml_content = f"""
path: {output_dir.absolute()}
train: images/train
val: images/val
names:
"""
    for idx, name in enumerate(cat_names):
        yaml_content += f"  {idx}: {name}\n"

    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"Dataset prepared at {output_dir}")
    print(f"data.yaml created at {output_dir / 'data.yaml'}")

convert_coco_to_yolo(
    json_path=f'{DATA_DIR}/annotations.coco.json',
    source_img_dir=DATA_DIR,
    output_dir='datasets/contrail-seg'
)



with wandb.init(project="contrails-seg") as run:
    def on_train_epoch_end(trainer):
        """Callback that runs at the end of each training epoch."""
        metrics = trainer.metrics
        epoch = trainer.epoch

        # Log training metrics
        log_dict = {
            "epoch": epoch,
        }

        # Add box loss metrics
        if "train/box_loss" in metrics:
            log_dict["train/box_loss"] = metrics["train/box_loss"]
        if "train/seg_loss" in metrics:
            log_dict["train/seg_loss"] = metrics["train/seg_loss"]
        if "train/cls_loss" in metrics:
            log_dict["train/cls_loss"] = metrics["train/cls_loss"]
        if "train/dfl_loss" in metrics:
            log_dict["train/dfl_loss"] = metrics["train/dfl_loss"]

        # Add validation metrics if available
        if "metrics/precision(M)" in metrics:
            log_dict["val/precision"] = metrics["metrics/precision(M)"]
        if "metrics/recall(M)" in metrics:
            log_dict["val/recall"] = metrics["metrics/recall(M)"]
        if "metrics/mAP50(M)" in metrics:
            log_dict["val/mAP50"] = metrics["metrics/mAP50(M)"]
        if "metrics/mAP50-95(M)" in metrics:
            log_dict["val/mAP50-95"] = metrics["metrics/mAP50-95(M)"]

        # Log to W&B
        run.log(metrics)
    def on_val_end(trainer):
        """Callback that runs at the end of validation."""
        metrics = trainer.metrics

        # Additional validation metrics can be logged here
        val_log_dict = {}

        if hasattr(metrics, 'box'):
            if hasattr(metrics.box, 'map'):
                val_log_dict["val/box_mAP"] = metrics.box.map
            if hasattr(metrics.box, 'map50'):
                val_log_dict["val/box_mAP50"] = metrics.box.map50

        if hasattr(metrics, 'seg'):
            if hasattr(metrics.seg, 'map'):
                val_log_dict["val/seg_mAP"] = metrics.seg.map
            if hasattr(metrics.seg, 'map50'):
                val_log_dict["val/seg_mAP50"] = metrics.seg.map50

        if val_log_dict:
            run.log(val_log_dict)
    model = YOLO("yolo11x-seg.pt")
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    model.train(
        data=f"./datasets/contrail-seg/data.yaml",
        epochs=100,
        imgsz=640,
        name="contrail-segmentation-runL",
        exist_ok=True,
        batch=-1,
        save_period=5,
        project="/content/drive/MyDrive/contrails_seg/xl",

    )





