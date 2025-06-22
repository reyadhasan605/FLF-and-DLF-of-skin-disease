import os
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse


def load_images(input_folder, target_size=(767, 576)):
    """Load images from the input folder, resize them to target_size, and return as a NumPy array."""
    # Get a sorted list of image files
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images = []
    for file in image_files:
        img_path = os.path.join(input_folder, file)
        # Load and convert to RGB
        img = Image.open(img_path).convert('RGB')
        # Resize to the target size
        img = img.resize(target_size)
        # Convert to NumPy array
        img = np.array(img)
        images.append(img)
    # Stack into a single NumPy array
    return np.array(images)

def augment_images(input_folder, output_folder, num_images_to_generate, seed=42):
    """Generate reproducible augmented images from the input directory."""
    # Set random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load images in a consistent order
    images = load_images(input_folder)

    # Define the data generator with augmentation parameters

    datagen = ImageDataGenerator(
        brightness_range=[0.7, 1.3],  # Adjust brightness (not in online)
        channel_shift_range=30.0,  # Color shift (not in online)
        vertical_flip=True,  # Only horizontal_flip in online
        featurewise_center=True,  # Advanced normalization
        featurewise_std_normalization=True,
    )

    # Create a generator with no shuffling
    generator = datagen.flow(
        images,
        batch_size=1,
        shuffle=False
    )

    # Ensure output directory exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate and save augmented images
    for i in range(num_images_to_generate):
        batch = generator.next()
        img = batch[0].astype(np.uint8)  # Convert to uint8 for image saving
        img_pil = Image.fromarray(img)
        img_path = output_path / f'aug_{i}.png'
        img_pil.save(img_path)

    saved_images = len(list(output_path.glob('*.png')))
    print(f"Generated and saved {saved_images} augmented images to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image augmentation using Keras ImageDataGenerator")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input image folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output image folder')
    parser.add_argument('--num_images', type=int, required=True, help='Number of augmented images to generate')

    args = parser.parse_args()

    augment_images(args.input_folder, args.output_folder, args.num_images)
