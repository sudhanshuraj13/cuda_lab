import os
import time
import cv2
from ultralytics import YOLO

FRAME_PATH = "/dev/shm/argus_frame.jpg"
MIN_JPEG_BYTES = 5000


def read_frame_safely(path: str):
    try:
        if not os.path.exists(path):
            return None

        size_before = os.path.getsize(path)
        if size_before < MIN_JPEG_BYTES:
            return None

        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        size_after = os.path.getsize(path)
        if size_before != size_after:
            return None

        return frame

    except:
        return None


def main():
    print("🔹 Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("✅ Model loaded. Starting detection...")

    last_valid_frame = None

    while True:
        frame = read_frame_safely(FRAME_PATH)

        if frame is None:
            if last_valid_frame is None:
                time.sleep(0.01)
                continue
            frame = last_valid_frame
        else:
            last_valid_frame = frame

        # 🔥 Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 🔥 Convert back to 3-channel (YOLO needs 3 channels)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 🔥 Run YOLO on grayscale image
        results = model(gray_3ch, conf=0.4, verbose=False)

        # 🔥 Draw boxes
        annotated = results[0].plot()

        # 🔥 Show result
        display = cv2.resize(annotated, (800, 600))
        cv2.imshow("YOLO (Black & White)", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()