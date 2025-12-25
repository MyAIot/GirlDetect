#!/usr/bin/env python3
"""
Prepare YOLO dataset structure from a folder containing images and label txt files.
Usage:
  python scripts/prepare_yolo_dataset.py --src ./raw --dest ./dataset --val 0.2

Expectations:
- Images: .jpg/.jpeg/.png
- Labels: .txt files with same basename as images (YOLO format)
- classes.txt in src folder (one class name per line)

This script will:
- create dest/images/train dest/images/val dest/labels/train dest/labels/val
- split images into train/val (random)
- copy images and their corresponding .txt labels into target folders
- generate data.yaml in dest/data.yaml
- print summary
"""
import argparse
import os
import shutil
import random
from pathlib import Path

def find_images(src):
    exts = ('.jpg','.jpeg','.png')
    return [p for p in src.iterdir() if p.suffix.lower() in exts]


def read_classes(src):
    cls_file = src / 'classes.txt'
    if not cls_file.exists():
        return None
    return [l.strip() for l in cls_file.read_text(encoding='utf-8').splitlines() if l.strip()]


def prepare_dirs(dest):
    (dest / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (dest / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (dest / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (dest / 'labels' / 'val').mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs, images_dst, labels_dst):
    copied = 0
    missing_labels = []
    for img in pairs:
        txt = img.with_suffix('.txt')
        dst_img = images_dst / img.name
        shutil.copy2(img, dst_img)
        if txt.exists():
            dst_txt = labels_dst / txt.name
            shutil.copy2(txt, dst_txt)
        else:
            missing_labels.append(img.name)
        copied += 1
    return copied, missing_labels


def write_data_yaml(dest, nc, names):
    content = f"""train: {str((dest / 'images' / 'train').resolve())}
val: {str((dest / 'images' / 'val').resolve())}
nc: {nc}
names: {names}
"""
    (dest / 'data.yaml').write_text(content, encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='.', help='Source folder containing images, labels and classes.txt')
    p.add_argument('--dest', default='dataset', help='Destination dataset folder')
    p.add_argument('--val', type=float, default=0.2, help='Validation split fraction')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    random.seed(args.seed)

    print('Source:', src)
    print('Destination:', dest)

    imgs = find_images(src)
    if not imgs:
        print('No images found in', src)
        return

    classes = read_classes(src)
    if classes is None:
        print('Warning: classes.txt not found in source. Please create it with one class per line.')
        classes = []

    prepare_dirs(dest)

    imgs_sorted = sorted(imgs)
    random.shuffle(imgs_sorted)
    n_val = max(1, int(len(imgs_sorted) * args.val))
    val_imgs = imgs_sorted[:n_val]
    train_imgs = imgs_sorted[n_val:]

    copied_train, missing_train = copy_pairs(train_imgs, dest / 'images' / 'train', dest / 'labels' / 'train')
    copied_val, missing_val = copy_pairs(val_imgs, dest / 'images' / 'val', dest / 'labels' / 'val')

    write_data_yaml(dest, nc=len(classes), names=classes)

    print('\nSummary:')
    print('Total images:', len(imgs_sorted))
    print('Train images copied:', copied_train, ' (missing labels:', len(missing_train), ')')
    print('Val images copied:', copied_val, ' (missing labels:', len(missing_val), ')')
    if missing_train or missing_val:
        print('\nImages missing label files:')
        for m in missing_train + missing_val:
            print(' -', m)
    print('\nWrote:', dest / 'data.yaml')
    print('Done.')

if __name__ == '__main__':
    main()
