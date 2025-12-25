Quick steps to prepare your YOLO dataset using the helper script

1) Put your images and corresponding .txt labels and `classes.txt` in a source folder, e.g. `raw/` or current dir.
   - images: .jpg/.jpeg/.png
   - labels: same basename .txt (YOLO format)
   - classes.txt: one class name per line

2) Run the script (from repo root):

```bash
python3 scripts/prepare_yolo_dataset.py --src ./raw --dest ./dataset --val 0.2
```

3) The script will:
- create `dataset/images/train`, `dataset/images/val`,
  `dataset/labels/train`, `dataset/labels/val`
- copy images and labels
- generate `dataset/data.yaml` (used by YOLOv5/YOLOv8)

4) Train example (YOLOv8):
```bash
pip install ultralytics
yolo train data=dataset/data.yaml model=yolov8n.pt imgsz=640 epochs=100
```

Notes:
- If some images do not have corresponding .txt, the script will still copy the image and report missing labels.
- `classes.txt` is used to set `nc` and `names` in the generated `data.yaml`.
