from datetime import datetime
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

model_name = "yolov11x"
IMAGE_DIR = "/Users/shrenikborad/Downloads/NNDL/images_uwisc/east/2025-01-09/east"
OUTPUT_VIDEO = f"segmentation_results_{model_name}_1280.mp4"

def run_inference(model_path=f"/Users/shrenikborad/pless/contrail-seg/best_{model_name}.pt", fps=5, max_images=None):
    # Load model
    model = YOLO(model_path)

    paths = sorted(list(Path(IMAGE_DIR).glob("*.jpg")))
    image_times = [datetime.strptime(f.name.split('.')[0], '%H_%M_%S') for f in paths]
    image_df = pd.DataFrame({'time': image_times, 'path': paths})
    image_df = image_df[(image_df['time'] >= datetime.strptime('07:00:00', '%H:%M:%S')) & (image_df['time'] < datetime.strptime('07:15:00', '%H:%M:%S'))]
    paths = image_df['path'].tolist()
    if max_images:
        paths = paths[:max_images]

    print(f"Found {len(paths)} images in {IMAGE_DIR}")

    if len(paths) == 0:
        print("No images found!")
        return

    # Process images and save annotated frames
    batch_size = 8
    total = len(paths)
    frame_size = None
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pbar = tqdm(total=total, desc="Processing images", unit="img")
    for start in range(0, total, batch_size):
        batch = paths[start:start + batch_size]

        # Run batch inference
        results = model.predict(source=batch, conf=0.25, device="mps")

        # results is an iterable of per-image Results
        for j, res in enumerate(results):
            annotated_frame = res.plot(line_width=2, masks=True, font_size=0.5)

            # Convert RGB to BGR for OpenCV
            # if annotated_frame.ndim == 3 and annotated_frame.shape[2] == 3:
            #     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # Initialize video writer with first frame dimensions
            if video_writer is None:
                h, w = annotated_frame.shape[:2]
                frame_size = (w, h)
                video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, frame_size)
                print(f"Video dimensions: {w}x{h}")

            # Resize to match first frame if needed
            if annotated_frame.shape[:2] != (frame_size[1], frame_size[0]):
                annotated_frame = cv2.resize(annotated_frame, frame_size)

            # Write to video
            video_writer.write(annotated_frame)

            pbar.update(1)
    pbar.close()

    if video_writer is not None:
        video_writer.release()
    print(f"Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    run_inference(max_images=None, fps=5)