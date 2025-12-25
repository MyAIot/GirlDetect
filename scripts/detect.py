#!/usr/bin/env python3
"""
YOLO Inference Script
Detect objects in images and return/save annotated images with bounding boxes.

Usage:
    python scripts/detect.py --image path/to/image.jpg --output path/to/output.jpg
    python scripts/detect.py --image path/to/image.jpg  # saves to input_detected.jpg
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import requests
from ultralytics import YOLO


def is_url(path):
    """Check if string is a URL."""
    return isinstance(path, str) and (path.startswith('http://') or path.startswith('https://'))


def load_image_from_url(url):
    """Download image from URL and convert to numpy array."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def detect_and_annotate(image_path, model_path='runs/detect/train/weights/best.pt', conf=0.25):
    """
    Detect objects in an image and return annotated image with bounding boxes.
    
    Args:
        image_path (str): Path to input image or URL
        model_path (str): Path to trained YOLO model weights
        conf (float): Confidence threshold (0-1)
    
    Returns:
        annotated_img (numpy.ndarray): Image with drawn bounding boxes and labels
        results: YOLO results object with detection info
    """
    # Load model
    model = YOLO(model_path)
    
    # Read image from URL or local path
    if is_url(image_path):
        img = load_image_from_url(image_path)
        if img is None:
            raise ValueError(f"Cannot download image from URL: {image_path}")
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
    
    # Run inference
    results = model.predict(source=img, conf=conf, verbose=False)
    
    # Get annotated image (with boxes drawn)
    annotated_img = results[0].plot()  # ultralytics draws boxes automatically
    
    return annotated_img, results[0]


def detect_and_save(image_path, output_path=None, model_path='runs/detect/train/weights/best.pt', conf=0.25):
    """
    Detect objects and save annotated image to file.
    
    Args:
        image_path (str): Path to input image or URL
        output_path (str): Path to save output (default: input_detected.jpg or url_detected.jpg)
        model_path (str): Path to trained model
        conf (float): Confidence threshold
    
    Returns:
        output_path (str): Path where image was saved
        results: Detection results
    """
    # Default output path
    if output_path is None:
        if is_url(image_path):
            output_path = Path('url_detected.jpg')
        else:
            image_path_obj = Path(image_path)
            output_path = image_path_obj.parent / f"{image_path_obj.stem}_detected{image_path_obj.suffix}"
    
    # Detect and annotate
    annotated_img, results = detect_and_annotate(image_path, model_path, conf)
    
    # Save
    cv2.imwrite(str(output_path), annotated_img)
    
    # Print detection summary
    boxes = results.boxes
    img_name = image_path if is_url(image_path) else Path(image_path).name
    print(f"\n✅ Detected {len(boxes)} objects in {img_name}")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        cls_name = results.names[cls_id]
        print(f"  - {cls_name}: {conf_val:.2%}")
    
    print(f"💾 Saved to: {output_path}\n")
    
    return str(output_path), results


def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--image', '-i', required=True, help='Path to input image or URL')
    parser.add_argument('--output', '-o', help='Path to output image (default: input_detected.jpg)')
    parser.add_argument('--model', '-m', default='runs/detect/train/weights/best.pt', 
                        help='Path to model weights')
    parser.add_argument('--conf', '-c', type=float, default=0.25, 
                        help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    detect_and_save(args.image, args.output, args.model, args.conf)


if __name__ == '__main__':
    main()
