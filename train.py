from ultralytics import YOLO

def train():
    # Load a model
    # yolo11m-seg.pt will be downloaded automatically
    model = YOLO("yolo11m-seg.pt")

    # Train the model
    # using 50 epochs, image size 640
    # gpu or mps

    results = model.train(
        data="datasets/contrail-seg/data.yaml",
        epochs=20,
        imgsz=640,
        device="cuda"
    )
    # Save the model after training
    model.save("trained_model.pt")


if __name__ == "__main__":
    train()
