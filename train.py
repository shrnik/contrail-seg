from ultralytics import YOLO

def train():
    # Load a model
    # yolo11m-seg.pt will be downloaded automatically
    model = YOLO("yolo11m-seg.pt")

    # Train the model
    # using 50 epochs, image size 640
    results = model.train(
        data="datasets/contrail-seg/data.yaml",
        epochs=50,
        imgsz=640,
        device='mps' if str(model.device) != 'cpu' else 'cpu', # Auto-detect MPS if available, though ultralytics handles 'device' arg well. 
        # On Mac, 'mps' is often available. Ultralytics auto-selects usually.
        # Let's just let it auto-detect or specify if needed.
        # Actually, explicitly setting device='mps' is good if on Apple Silicon.
    )

if __name__ == "__main__":
    train()
