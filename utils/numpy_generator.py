import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse


def get_class_mapping(base_dir):
    """Returns class-to-index mapping and sorted class names"""
    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    print(f"Classes found: {class_to_idx}")
    return class_names, class_to_idx


def collect_image_paths_and_labels(base_dir, class_names, class_to_idx):
    """Returns lists of image file paths and corresponding labels"""
    image_paths, labels = [], []
    for cls in class_names:
        cls_folder = os.path.join(base_dir, cls)
        for img_file in os.listdir(cls_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(cls_folder, img_file))
                labels.append(class_to_idx[cls])
    return image_paths, labels


def load_and_preprocess(path, image_size):
    """Load image, convert to RGB, resize, normalize"""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), resample=Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def build_dataset(image_paths, labels, image_size):
    """Preprocess and return X, y arrays"""
    print("Loading and preprocessing images...")
    X_data = np.stack([load_and_preprocess(p, image_size) for p in image_paths])
    y_data = np.array(labels)
    print(f" Processed {len(X_data)} images.")
    return X_data, y_data


def save_numpy_arrays(X_data, y_data, output_dir):
    """Save arrays to specified output directory"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_test.npy"), X_data)
    np.save(os.path.join(output_dir, "y_test.npy"), y_data)
    print(f"Saved X_test.npy and y_test.npy to {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare .npy files from directory-structured dataset")
    parser.add_argument('--data_dir', type=str, default="path/to/folder",
                        help='Path to dataset directory (subfolders = class names)')
    parser.add_argument('--output_dir', type=str, default="path/to/folder",
                        help='Output directory to save X_test.npy and y_test.npy')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size (height and width) to resize images to (default: 224)')
    return parser.parse_args()


def main():
    args = parse_arguments()

    class_names, class_to_idx = get_class_mapping(args.data_dir)
    image_paths, labels = collect_image_paths_and_labels(args.data_dir, class_names, class_to_idx)
    X_data, y_data = build_dataset(image_paths, labels, args.image_size)
    save_numpy_arrays(X_data, y_data, args.output_dir)


if __name__ == "__main__":
    main()
