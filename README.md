# YOLO Object Detection Project

A custom YOLO-based object detection project for detecting 16 different classes including dogs, persons, cats, cars, various foods, and more.

## 📋 Features

- Custom YOLO model training with YOLOv8
- Object detection from local images or URLs
- Automated dataset preparation and splitting
- Easy-to-use Makefile commands
- Detection from clipboard (macOS)
- Annotated image output with bounding boxes

## 🎯 Detected Classes

The model can detect 16 classes:
- **Animals**: dog, cat
- **People**: person, girl
- **Objects**: tv, car
- **Food**: meatballs, marinara sauce, tomato soup, chicken noodle soup, french onion soup, chicken breast, ribs, pulled pork, hamburger
- **Medical**: cavity

## 📁 Project Structure

```
.
├── dataset/                    # Training dataset
│   ├── images/
│   │   ├── train/             # Training images
│   │   └── val/               # Validation images
│   ├── labels/
│   │   ├── train/             # Training labels (YOLO format)
│   │   └── val/               # Validation labels
│   └── data.yaml              # Dataset configuration
├── runs/                       # Training outputs
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt    # Best model weights
│               └── last.pt    # Last checkpoint
├── scripts/
│   ├── prepare_yolo_dataset.py  # Dataset preparation script
│   └── detect.py              # Detection script
├── example_usage.py           # Usage examples
├── Makefile                   # Build automation
└── yolov8n.pt                 # Pre-trained YOLOv8 nano model

```

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install ultralytics opencv-python requests
```

### Training the Model

1. **Prepare your dataset:**
   ```bash
   make prepare-dataset SRC_DIR=~/path/to/your/images
   ```

2. **Check dataset structure:**
   ```bash
   make check-dataset
   ```

3. **Train the model:**
   ```bash
   make train
   ```
   
   Or with custom parameters:
   ```bash
   make train EPOCHS=200 BATCH_SIZE=32 IMG_SIZE=640
   ```

### Running Detection

**Option 1: Detect from clipboard (macOS)**
```bash
# Copy an image URL or file path to clipboard, then:
make detect
```

**Option 2: Detect from specific file**
```bash
make detect-manual IMAGE=path/to/image.jpg
```

**Option 3: Detect with custom output and confidence**
```bash
make detect-manual IMAGE=path/to/image.jpg OUTPUT=result.jpg CONF=0.5
```

## 📖 Usage Examples

### Python API

```python
from scripts.detect import detect_and_annotate, detect_and_save

# Detect from local file
annotated_img, results = detect_and_annotate('image.jpg')

# Detect from URL
annotated_img, results = detect_and_annotate('https://example.com/image.jpg')

# Detect and auto-save
output_path, results = detect_and_save('image.jpg')

# Custom confidence threshold
annotated_img, results = detect_and_annotate('image.jpg', conf=0.5)

# Access detection details
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
    class_id = int(box.cls[0])
    class_name = results.names[class_id]
    print(f"{class_name}: {confidence:.2%}")
```

### Command Line

```bash
# Basic detection
python3 scripts/detect.py --image path/to/image.jpg

# With output path
python3 scripts/detect.py --image image.jpg --output detected.jpg

# From URL
python3 scripts/detect.py --image https://example.com/photo.jpg

# Custom confidence
python3 scripts/detect.py --image image.jpg --conf 0.5
```

## 🔧 Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make prepare-dataset` | Prepare YOLO dataset from source folder |
| `make check-dataset` | Check dataset structure and files |
| `make train` | Train YOLO model |
| `make validate` | Validate trained model |
| `make predict` | Run prediction on validation images |
| `make detect` | Detect from clipboard URL/path |
| `make detect-manual` | Detect with manual IMAGE parameter |
| `make export` | Export model to ONNX format |
| `make clean` | Clean training outputs |
| `make clean-dataset` | Clean dataset folder |

## ⚙️ Configuration

### Makefile Variables

You can override these variables when running make commands:

```bash
make train EPOCHS=200 BATCH_SIZE=32 IMG_SIZE=640
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SRC_DIR` | `~/Downloads/training` | Source folder for dataset preparation |
| `DEST_DIR` | `./dataset` | Destination for prepared dataset |
| `VAL_SPLIT` | `0.2` | Validation split ratio |
| `EPOCHS` | `100` | Number of training epochs |
| `IMG_SIZE` | `640` | Input image size |
| `BATCH_SIZE` | `16` | Training batch size |
| `MODEL` | `yolov8n.pt` | Base YOLO model |

## 📊 Training Results

Training outputs are saved in `runs/detect/train*/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.csv` - Training metrics
- `args.yaml` - Training arguments

## 🎓 Model Performance

The model is trained on a custom dataset with:
- Training images: ~32 images
- Validation images: ~14 images
- Image size: 640x640
- Base model: YOLOv8 nano

Check `runs/detect/train/results.csv` for detailed metrics.

## 🛠️ Troubleshooting

**Error: Clipboard is empty**
- Make sure you've copied an image URL or file path before running `make detect`

**Error: Missing dataset folders**
- Run `make prepare-dataset` first
- Verify dataset structure with `make check-dataset`

**Low detection accuracy**
- Increase training epochs: `make train EPOCHS=200`
- Increase batch size if you have enough memory
- Adjust confidence threshold: `make detect-manual IMAGE=image.jpg CONF=0.3`

## 📝 License

This project uses YOLOv8 from Ultralytics. Please refer to their license terms.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

**Author**: Huan Pham  
**Last Updated**: December 2025
