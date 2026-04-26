import os
import time

import cv2
from ultralytics import YOLO


FRAME_PATH = "/dev/shm/argus_frame.jpg"
MIN_JPEG_BYTES = 1024


def read_frame_safely(path: str):
    try:
        size_before = os.path.getsize(path)
        if size_before < MIN_JPEG_BYTES:
            return None

        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return None

        size_after = os.path.getsize(path)
        if size_before != size_after:
            return None

        return frame
    except (FileNotFoundError, PermissionError, OSError):
        return None


def main():
    model = YOLO("yolov8n.pt")

    while True:
        frame = read_frame_safely(FRAME_PATH)
        if frame is None:
            time.sleep(0.005)
            continue

        results = model(frame, verbose=False)
        best_label = None
        best_conf = -1.0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    class_id = int(box.cls[0])
                    best_label = result.names[class_id]

        if best_label is not None:
            print(f"Alert: {best_label.capitalize()} detected", flush=True)


if __name__ == "__main__":
    main()
