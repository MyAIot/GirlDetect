"""
Example usage of detect functions
"""
from scripts.detect import detect_and_annotate, detect_and_save
import cv2

# Example 1: Detect from local file
annotated_img, results = detect_and_annotate('path/to/image.jpg')

# Example 2: Detect from URL
annotated_img, results = detect_and_annotate('https://example.com/image.jpg')
cv2.imwrite('output.jpg', annotated_img)

# Example 3: Detect and auto-save (local file)
output_path, results = detect_and_save('path/to/image.jpg')

# Example 4: Detect and auto-save (URL)
output_path, results = detect_and_save('https://example.com/cat.jpg', output_path='cat_detected.jpg')

# Example 5: Detect with custom confidence threshold
annotated_img, results = detect_and_annotate('https://example.com/dog.jpg', conf=0.5)

# Example 6: Access detection details
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
    confidence = box.conf[0]       # Confidence score
    class_id = int(box.cls[0])     # Class ID
    class_name = results.names[class_id]  # Class name
    print(f"{class_name}: {confidence:.2%} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
