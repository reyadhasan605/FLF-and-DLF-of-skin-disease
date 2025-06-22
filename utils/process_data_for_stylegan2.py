import argparse
import os
from PIL import Image

def resize_images(source_dir, target_dir, size=(256, 256)):
    os.makedirs(target_dir, exist_ok=True)
    for img_name in os.listdir(source_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(source_dir, img_name)
            img = Image.open(img_path)
            img = img.resize(size, Image.LANCZOS)
            img.save(os.path.join(target_dir, img_name))
    print(f"Resized images saved to: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images in a directory.")
    parser.add_argument("--input_dir", required=True, help="Path to the source image directory.")
    parser.add_argument("--output_dir", required=True, help="Path to save the resized images.")
    parser.add_argument("--size", nargs=2, type=int, default=[256, 256], help="Resize dimensions (width height).")

    args = parser.parse_args()

    resize_images(args.source, args.target, size=tuple(args.size))
