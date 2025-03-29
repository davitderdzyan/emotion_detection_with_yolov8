from ultralytics import YOLO

model = YOLO()
if __name__ == '__main__':
    model.train(
        data="C:/Users/derdz/Downloads/diploma_work/YOLO_format/data.yaml",
        epochs=50,
        imgsz=96,
        # weights=weights_path  # Specify custom destination directory for trained model weights
    )
    # Add these lines at the end
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
