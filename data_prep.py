import os
import sys
import json
import random
from PIL import Image
import numpy as np
import fitz  # PyMuPDF

from effects import (
    add_salt_and_pepper_noise, apply_gaussian_blur, apply_binary_threshold,
    add_motion_blur, add_random_lines_to_greyscale,
    apply_page_rotation, apply_optical_problems, apply_local_distortions,
    add_contrast_change_noise, add_pixelate_noise,
    add_jitter_noise, add_mean_shift_noise
)

# Function to create necessary directories
def create_directories(base_dir, combo_name):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, combo_name), exist_ok=True)

# Function to rename and copy original images
def copy_and_rename_images(src_dir):
    image_paths = []
    count = 1
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                image_paths.append((src_path, f'{count}.png'))
                count += 1
    return image_paths

# Function to convert PDF to images
def pdf_to_images(pdf_path, output_folder, image_format='png', dpi=300):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Increase the resolution by setting the dpi
        zoom = dpi / 72  # 72 is the default dpi
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to greyscale
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = image.convert("L")  # Convert to greyscale

        # Define the image path
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.{image_format}")

        # Save the image
        image.save(image_path)
        print(f"Saved page {page_num + 1} as greyscale image to {image_path}")

# Function to apply distortions to an image and save the results
def apply_and_save_distortions(image_paths, base_dir, noise_type1, noise_type2):
    distortion_levels = {
        'binary_threshold': list(range(50, 256, 25)),  # Binary threshold levels
        'gaussian_blur': list(range(1, 11)),  # Gaussian blur levels
        'optical_problems': [i * 0.1 for i in range(1, 11)],  # Optical problems levels
        'rotation': list(range(-5, 6, 1)),  # Rotation levels
        'contrast_change': [i * 0.2 + 1 for i in range(0, 10)],  # Contrast change levels
        'pixelate': list(range(1, 11)),  # Pixelate levels
        'jitter': list(range(1, 11)),  # Jitter levels
        'mean_shift': list(range(10, 110, 10)),  # Mean shift levels
        'salt_and_pepper': [i * 0.001 for i in range(1, 11)]  # Salt and pepper noise levels
    }

    ground_truth = []

    combo_name = f"{noise_type1}_{noise_type2}"
    create_directories(base_dir, combo_name)
    combo_dir = os.path.join(base_dir, combo_name)

    for src_path, img_name in image_paths:
        image = Image.open(src_path)

        # Save the reference image with ground truth zero
        ref_image_path = os.path.join(combo_dir, img_name)
        image.save(ref_image_path, quality=100)

        if noise_type1 == 'binary_threshold':
            ground_truth.append({'image': img_name, 'distortion_level': [0, 0]})
        elif noise_type2 == 'binary_threshold':
            ground_truth.append({'image': img_name, 'distortion_level': [0, 0]})
        else:
            ground_truth.append({'image': img_name, 'distortion_level': [0, 0]})

        # Generate 20 random combinations of distortion levels for each noise pair
        for _ in range(20):
            level1 = random.choice(distortion_levels[noise_type1])
            level2 = random.choice(distortion_levels[noise_type2])

            distorted_image = image.copy()

            if noise_type1 == 'binary_threshold':
                distorted_image = apply_binary_threshold(distorted_image, level1)
            elif noise_type1 == 'gaussian_blur':
                distorted_image = apply_gaussian_blur(distorted_image, level1)
            elif noise_type1 == 'optical_problems':
                distorted_image = apply_optical_problems(distorted_image, level1)
            elif noise_type1 == 'rotation':
                distorted_image = apply_page_rotation(distorted_image, level1)
            elif noise_type1 == 'contrast_change':
                distorted_image = add_contrast_change_noise(distorted_image, level1)
            elif noise_type1 == 'pixelate':
                distorted_image = add_pixelate_noise(distorted_image, level1)
            elif noise_type1 == 'jitter':
                distorted_image = add_jitter_noise(distorted_image, level1)
            elif noise_type1 == 'mean_shift':
                distorted_image = add_mean_shift_noise(distorted_image, level1)
            elif noise_type1 == 'salt_and_pepper':
                distorted_image = add_salt_and_pepper_noise(distorted_image, level1)

            if noise_type2 == 'binary_threshold':
                distorted_image = apply_binary_threshold(distorted_image, level2)
            elif noise_type2 == 'gaussian_blur':
                distorted_image = apply_gaussian_blur(distorted_image, level2)
            elif noise_type2 == 'optical_problems':
                distorted_image = apply_optical_problems(distorted_image, level2)
            elif noise_type2 == 'rotation':
                distorted_image = apply_page_rotation(distorted_image, level2)
            elif noise_type2 == 'contrast_change':
                distorted_image = add_contrast_change_noise(distorted_image, level2)
            elif noise_type2 == 'pixelate':
                distorted_image = add_pixelate_noise(distorted_image, level2)
            elif noise_type2 == 'jitter':
                distorted_image = add_jitter_noise(distorted_image, level2)
            elif noise_type2 == 'mean_shift':
                distorted_image = add_mean_shift_noise(distorted_image, level2)
            elif noise_type2 == 'salt_and_pepper':
                distorted_image = add_salt_and_pepper_noise(distorted_image, level2)

            distorted_img_name = f'{os.path.splitext(img_name)[0]}_{level1}_{level2}.png'
            distorted_image_path = os.path.join(combo_dir, distorted_img_name)
            distorted_image.save(distorted_image_path, quality=100)

            if noise_type1 == 'binary_threshold':
                ground_truth.append({'image': distorted_img_name, 'distortion_level': [level1, level2]})
            elif noise_type2 == 'binary_threshold':
                ground_truth.append({'image': distorted_img_name, 'distortion_level': [level1, level2]})
            else:
                ground_truth.append({'image': distorted_img_name, 'distortion_level': [level1, level2]})

    # Ensure that JSON is properly written
    try:
        with open(os.path.join(combo_dir, 'ground_truth.json'), 'w') as f:
            json.dump(ground_truth, f, indent=4)
    except Exception as e:
        print(f"Error writing JSON file: {e}")

# Main processing function
def main():
    base_dir = 'dataset'
    src_dir = 'data'  # Folder containing the PDF files
    intermediate_img_folder = 'temp_images'  # Temporary folder to store images converted from PDFs

    # Step 1: Convert all PDFs in src_dir to images
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_output_folder = os.path.join(intermediate_img_folder, os.path.splitext(file)[0])
                pdf_to_images(pdf_path, pdf_output_folder, dpi=200)

    # Step 2: Copy and rename original images from intermediate_img_folder
    image_paths = copy_and_rename_images(intermediate_img_folder)

    # Step 3: Apply distortions and save the results
    noise_type1 = 'binary_threshold'  # Specify the first noise type here
    noise_type2 = 'gaussian_blur'  # Specify the second noise type here

    apply_and_save_distortions(image_paths, base_dir, noise_type1, noise_type2)

if __name__ == '__main__':
    main()
