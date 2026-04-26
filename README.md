# Real-Time Webcam Detection Pipeline

This project is a two-process pipeline for real-time object detection:

- `gpu_processor.cpp` captures frames from webcam device `0`, resizes them to `640x480`, converts them to grayscale, and writes the latest frame to `/dev/shm/argus_frame.jpg`.
- `inference.py` watches that JPEG file, loads it safely with OpenCV, runs YOLOv8 inference, and prints the highest-confidence detected label.

## How It Works

1. The C++ process reads frames from the webcam.
2. The frame is resized and converted to grayscale.
3. The processed frame is written to shared memory as `/dev/shm/argus_frame.jpg`.
4. The Python process reads that image and runs YOLOv8.
5. If an object is detected, the script prints:

```text
Alert: Person detected
```

## Requirements

The current code is written for Linux because it uses `/dev/shm` as a shared-memory file location.

You need:

- Linux or WSL with webcam access
- Python 3.10+
- OpenCV for Python
- Ultralytics YOLO
- OpenCV development libraries for C++
- A C++17 compiler
- Optional: NVIDIA HPC SDK if you want the OpenACC pragmas in `gpu_processor.cpp` to run on GPU

## Python Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `ultralytics` will also install `torch` and related dependencies.
- On the first run, YOLO may download `yolov8n.pt` automatically if it is not already present.

## C++ Build Dependencies

On Ubuntu/Debian, install the common build tools and OpenCV headers:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libopencv-dev
```

## Build The C++ Program

### Option 1: Build with NVIDIA OpenACC support

Use this if you have the NVIDIA HPC SDK installed and want the `#pragma acc` sections to be compiled for acceleration:

```bash
nvc++ -O3 -std=c++17 -acc gpu_processor.cpp -o gpu_processor $(pkg-config --cflags --libs opencv4)
```

### Option 2: Build with g++

This is the simplest option. The code will still compile, but the OpenACC pragmas may be ignored, so the grayscale conversion will effectively run on CPU:

```bash
g++ -O3 -std=c++17 gpu_processor.cpp -o gpu_processor $(pkg-config --cflags --libs opencv4)
```

## How To Run

Run the two programs in separate terminals.

### Terminal 1: Start the frame producer

```bash
./gpu_processor
```

This process continuously captures webcam frames and writes the latest grayscale JPEG to:

```text
/dev/shm/argus_frame.jpg
```

### Terminal 2: Start YOLO inference

```bash
source .venv/bin/activate
python3 inference.py
```

If objects are detected, you will see output like:

```text
Alert: Person detected
Alert: Car detected
```

## Expected File Flow

- Temporary output: `/dev/shm/argus_frame.tmp.jpg`
- Final frame consumed by Python: `/dev/shm/argus_frame.jpg`

The temporary file is renamed into place so the Python process is less likely to read a partially written JPEG.

## Troubleshooting

### Webcam cannot be opened

If you see:

```text
ERROR: Unable to open webcam device 0
```

then either:

- no webcam is available
- the camera index is different
- your environment does not expose webcam access

To try another camera, change this line in `gpu_processor.cpp`:

```cpp
cv::VideoCapture cap(0);
```

### `inference.py` prints nothing

Possible reasons:

- `gpu_processor` is not running
- `/dev/shm/argus_frame.jpg` is not being created
- the model is still downloading on first run
- no detections are being produced above the model's default threshold

### Running on Windows

The current code will not work as-is on plain Windows because it depends on the Linux path `/dev/shm/...`.

To run on Windows, update these paths in both files to the same writable location, for example:

```text
C:/temp/argus_frame.jpg
```

You would need to change:

- `FRAME_PATH` in `inference.py`
- `kOutputPath` in `gpu_processor.cpp`
- `kTempPath` in `gpu_processor.cpp`

## Project Files

- `gpu_processor.cpp`: webcam capture, resize, grayscale conversion, JPEG writer
- `inference.py`: frame reader and YOLO inference loop
- `requirements.txt`: Python dependencies for the inference script
