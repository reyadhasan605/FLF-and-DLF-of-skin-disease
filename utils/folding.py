import random
import shutil
from collections import defaultdict
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Split dataset into k-folds for cross-validation")
parser.add_argument('--root_dir', type=str, required=True,
                    help='Path to the root directory containing class folders')
parser.add_argument('--output_root', type=str, required=True,
                    help='Path to the output directory for fold structure')
parser.add_argument('--num_folds', type=int, default=5,
                    help='Number of folds for cross-validation (default: 5)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

args = parser.parse_args()

# Set random seed
random.seed(args.seed)

# Create output directory
os.makedirs(args.output_root, exist_ok=True)

folds = defaultdict(list)

# 1. Sort the list of classes
for class_name in sorted(os.listdir(args.root_dir)):
    class_path = os.path.join(args.root_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # 2. Gather and sort the image filenames
    images = sorted([
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    ])

    # 3. Shuffle the sorted list
    random.shuffle(images)
    fold_size = len(images) // args.num_folds

    for i in range(args.num_folds):
        start = i * fold_size
        end = None if i == args.num_folds - 1 else (i + 1) * fold_size
        folds[i].extend([(img, class_name) for img in images[start:end]])

# Copy into fold directories
for fold_idx in range(args.num_folds):
    fold_dir = os.path.join(args.output_root, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    for img_path, class_name in folds[fold_idx]:
        class_dir = os.path.join(fold_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        dst_path = os.path.join(class_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
        print(f"Copied to: {dst_path}")