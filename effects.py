from PIL import Image
import numpy as np
import cv2
import random

# Function to add salt and pepper noise
def add_salt_and_pepper_noise(image, amount=0.1, salt_vs_pepper=0.5):
    """
    Add salt and pepper noise to an image.

    Parameters:
    image (PIL.Image): The input image.
    amount (float): Amount of noise to add.
    salt_vs_pepper (float): Proportion of salt vs pepper noise.

    Returns:
    PIL.Image: The image with added salt and pepper noise.
    """
    noisy = np.array(image)

    if noisy.ndim == 2:
        num_salt = np.ceil(amount * noisy.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * noisy.size * (1.0 - salt_vs_pepper))

        # Add Salt noise (white pixels)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy.shape]
        noisy[coords] = 255

        # Add Pepper noise (black pixels)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy.shape]
        noisy[coords] = 0
    else:
        num_salt = np.ceil(amount * noisy.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * noisy.size * (1.0 - salt_vs_pepper))

        # Add Salt noise (white pixels)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy.shape]
        noisy[coords[0], coords[1], :] = 255

        # Add Pepper noise (black pixels)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy.shape]
        noisy[coords[0], coords[1], :] = 0

    return Image.fromarray(noisy)


# Function to apply Gaussian blur using OpenCV
def apply_gaussian_blur(image, ksize):
    image_np = np.array(image)
    ksize = max(1, ksize // 2 * 2 + 1)  # Ensure kernel size is odd and >= 1
    blurred = cv2.GaussianBlur(image_np, (ksize, ksize), 0)
    return Image.fromarray(blurred)


# Function to apply binary thresholding
def apply_binary_threshold(image, threshold):
    image_np = np.array(image.convert('L'))  # Convert to grayscale
    _, binary = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)


# Function to add motion blur
def add_motion_blur(image, size=10):
    image_np = np.array(image)
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    blurred = cv2.filter2D(image_np, -1, kernel_motion_blur)
    return Image.fromarray(blurred)


# Function to add compression artifacts
def add_compression_artifacts(image, quality=10):
    image_np = np.array(image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image_np, encode_param)
    decoded_image = cv2.imdecode(encoded_image, 1)
    return Image.fromarray(decoded_image)


def add_random_lines_to_greyscale(image, greyscale=128, sparsity_level=5):
    image_np = np.array(image.convert('L'))  # Convert to grayscale if not already
    height, width = image_np.shape
    drawn_lines = set()

    # Define the number of possible lines based on the sparsity level
    sparsity_to_lines = {1: 500, 2: 400, 3: 300, 4: 200, 5: 100, 6: 70, 7: 40, 8: 18, 9: 8, 10: 3}
    total_possible_lines = sparsity_to_lines.get(sparsity_level, 50)  # Default to 50 if level is out of range

    for _ in range(total_possible_lines):
        orientation = np.random.randint(2)
        line_width = np.random.randint(1, 6)

        if orientation == 0:  # Horizontal line
            y = np.random.randint(0, height)
            attempts = 0
            while (orientation, y) in drawn_lines and attempts < 100:
                y = np.random.randint(0, height)
                attempts += 1
            if attempts < 100:
                drawn_lines.add((orientation, y))
                x = 0
                while x < width:
                    if np.random.rand() > 0.5:  # Randomly decide whether to draw or gap
                        segment_size = np.random.randint(5, 150)
                        draw_end = min(x + segment_size, width)
                        if x < width:
                            image_np[max(0, y - line_width // 2):min(height, y + line_width // 2), x:draw_end] = greyscale
                        x += segment_size
                    else:
                        segment_size = np.random.randint(5, 150)
                        x += segment_size

        else:  # Vertical line
            x = np.random.randint(0, width)
            attempts = 0
            while (orientation, x) in drawn_lines and attempts < 100:
                x = np.random.randint(0, width)
                attempts += 1
            if attempts < 100:
                drawn_lines.add((orientation, x))
                y = 0
                while y < height:
                    if np.random.rand() > 0.5:  # Randomly decide whether to draw or gap
                        segment_size = np.random.randint(5, 150)
                        draw_end = min(y + segment_size, height)
                        if y < height:
                            image_np[y:draw_end, max(0, x - line_width // 2):min(width, x + line_width // 2)] = greyscale
                        y += segment_size
                    else:
                        segment_size = np.random.randint(5, 150)
                        y += segment_size

    return Image.fromarray(image_np)


def apply_page_rotation(image, tilt_angle):
    # Convert image to numpy array
    image_np = np.array(image)
    # Get image dimensions
    rows, cols = image_np.shape[:2]
    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), tilt_angle, 1)
    # Apply the rotation
    rotated = cv2.warpAffine(image_np, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)


# Function to apply Floyd-Steinberg dithering
def apply_dithering(image, whitening_threshold=240, black_threshold=50):
    # Convert image to grayscale to create a mask for black text and white background
    grayscale = np.array(image.convert('L'))
    black_text_mask = grayscale < black_threshold  # Threshold to identify black text
    white_background_mask = grayscale > whitening_threshold  # Threshold to identify white background

    # Convert image to RGB for processing
    image_np = np.array(image.convert('RGB'))

    # Apply the dithering algorithm to each color channel separately, excluding black text and white background regions
    for channel in range(3):
        channel_data = image_np[:, :, channel]

        for y in range(channel_data.shape[0]):
            for x in range(channel_data.shape[1]):
                if not (black_text_mask[y, x] or white_background_mask[y, x]):
                    old_pixel = channel_data[y, x]
                    new_pixel = 255 if old_pixel > 128 else 0
                    channel_data[y, x] = new_pixel
                    quant_error = old_pixel - new_pixel

                    if x + 1 < channel_data.shape[1]:
                        channel_data[y, x + 1] = min(255, max(0, channel_data[y, x + 1] + quant_error * 7 / 16))
                    if y + 1 < channel_data.shape[0]:
                        if x - 1 >= 0:
                            channel_data[y + 1, x - 1] = min(255,
                                                             max(0, channel_data[y + 1, x - 1] + quant_error * 3 / 16))
                        channel_data[y + 1, x] = min(255, max(0, channel_data[y + 1, x] + quant_error * 5 / 16))
                        if x + 1 < channel_data.shape[1]:
                            channel_data[y + 1, x + 1] = min(255,
                                                             max(0, channel_data[y + 1, x + 1] + quant_error * 1 / 16))

    # Combine the dithered channels back into an image
    dithered_image = np.stack([image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]], axis=2)

    # Restore black text and white background regions from the original grayscale image
    grayscale_restored = np.array(image.convert('L'))
    dithered_image[black_text_mask] = np.stack([grayscale_restored[black_text_mask]] * 3, axis=-1)
    dithered_image[white_background_mask] = np.stack([grayscale_restored[white_background_mask]] * 3, axis=-1)

    # Convert the result back to grayscale
    dithered_image_gray = Image.fromarray(dithered_image).convert('L')

    return dithered_image_gray


def apply_optical_problems(image, darkness_level=0.5):
    image_np = np.array(image.convert('RGB'))  # Convert image to RGB and then to a NumPy array
    height, width, _ = image_np.shape  # Get the dimensions of the image

    # Create a gradient or noise pattern for uneven lighting
    gradient = np.tile(np.linspace(1, darkness_level, width), (height, 1))
    gradient = np.dstack([gradient] * 3)

    # Apply the gradient to the image
    uneven_lighting_image = (image_np * gradient).astype(np.uint8)

    return Image.fromarray(uneven_lighting_image)


# Function to apply stretching or shrinking effect due to page movement
def apply_local_distortions(image, num_regions=3, max_shift=5, distortion_type=1):
    image_np = np.array(image)
    height, width, _ = image_np.shape

    applied_regions = []

    for _ in range(num_regions):
        # Define a vertical shift magnitude
        shift_magnitude = random.randint(1, max_shift)

        # Ensure that the new region does not overlap with previously applied regions
        while True:
            y_start = random.randint(0, height - shift_magnitude - 1)
            y_end = y_start + shift_magnitude

            overlapping = any(y_start < y < y_end or y_start < y2 < y_end for y, y2 in applied_regions)
            if not overlapping:
                break

        applied_regions.append((y_start, y_end))

        if distortion_type == 1:  # Stretch
            shift_amount = shift_magnitude

            src_points = np.float32([
                [0, y_start],
                [width, y_start],
                [0, y_end]
            ])

            dst_points = np.float32([
                [0, y_start + shift_amount],
                [width, y_start + shift_amount],
                [0, y_end + shift_amount]
            ])

            # Calculate the affine transformation matrix
            M = cv2.getAffineTransform(src_points, dst_points)

            # Create a larger region by concatenating with the region below
            if y_end + shift_magnitude > height:
                larger_region = np.vstack([image_np[y_start:y_end, :], np.zeros((shift_magnitude, width, 3), dtype=image_np.dtype)])
            else:
                larger_region = np.vstack([image_np[y_start:y_end, :], image_np[y_end:y_end + shift_magnitude, :]])

            # Apply transformation to the larger region
            transformed_region = cv2.warpAffine(larger_region, M, (width, y_end - y_start + shift_magnitude), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # Correctly crop the transformed region to avoid repetition
            transformed_region = transformed_region[:y_end - y_start + shift_magnitude, :]

            # Concatenate the transformed region with the unaffected parts of the image
            upper_part = image_np[:y_start, :]
            lower_part = image_np[y_end:, :]
            image_np = np.vstack([upper_part, transformed_region, lower_part])[:height, :]

        elif distortion_type == -1:  # Shrink
            # Remove the desired lines of pixels
            region_above = image_np[:y_start, :]
            region_below = image_np[y_end + shift_magnitude:, :]
            image_np = np.vstack([region_above, region_below])

            # Pad the result to maintain the original height
            padding = np.zeros((shift_magnitude, width, 3), dtype=image_np.dtype)
            image_np = np.vstack([image_np, padding])[:height, :]

    return Image.fromarray(image_np.astype(np.uint8))

def add_mean_shift_noise(image, shift_value=50):
    """
    Add mean shift noise to an image.

    Parameters:
    image (PIL.Image): The input image.
    shift_value (int): The value by which to shift the mean brightness of the image.

    Returns:
    PIL.Image: The image with added mean shift noise.
    """
    image_np = np.array(image, dtype=np.int16)  # Use int16 to avoid overflow
    shifted_image_np = np.clip(image_np + shift_value, 0, 255)  # Apply the shift and clip to valid range
    shifted_image_np = shifted_image_np.astype(np.uint8)  # Convert back to uint8

    return Image.fromarray(shifted_image_np)


def add_jitter_noise(image, max_jitter=5):
    """
    Add jitter noise to an image by randomly shifting pixels in small regions.

    Parameters:
    image (PIL.Image): The input image.
    max_jitter (int): The maximum pixel shift for the jitter.

    Returns:
    PIL.Image: The image with added jitter noise.
    """
    image_np = np.array(image)
    # Check if the image is grayscale or RGB
    if len(image_np.shape) == 2:
        height, width = image_np.shape
        channels = 1
    else:
        height, width, channels = image_np.shape

    jittered_image_np = image_np.copy()

    for y in range(height):
        for x in range(width):
            shift_y = np.random.randint(-max_jitter, max_jitter + 1)
            shift_x = np.random.randint(-max_jitter, max_jitter + 1)

            new_y = np.clip(y + shift_y, 0, height - 1)
            new_x = np.clip(x + shift_x, 0, width - 1)

            jittered_image_np[y, x] = image_np[new_y, new_x]

    return Image.fromarray(jittered_image_np)

def add_pixelate_noise(image, pixel_size=10):
    """
    Add pixelate noise to an image by reducing its resolution.

    Parameters:
    image (PIL.Image): The input image.
    pixel_size (float): The size of the pixel blocks.

    Returns:
    PIL.Image: The image with added pixelate noise.
    """
    image_np = np.array(image)

    # Check if the image is grayscale or RGB
    if len(image_np.shape) == 2:
        height, width = image_np.shape
    else:
        height, width, _ = image_np.shape

    # Reduce resolution
    reduced_height = max(1, int(height / pixel_size))
    reduced_width = max(1, int(width / pixel_size))
    small_image = cv2.resize(image_np, (reduced_width, reduced_height), interpolation=cv2.INTER_LINEAR)

    # Restore original resolution
    pixelated_image_np = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    return Image.fromarray(pixelated_image_np)

def add_high_sharpen_noise(image):
    """
    Add high sharpen noise to an image.

    Parameters:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The image with added high sharpen noise.
    """
    image_np = np.array(image)

    # Define a high-pass filter kernel for sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply the high-pass filter to the image
    sharpened_image_np = cv2.filter2D(image_np, -1, kernel)

    return Image.fromarray(sharpened_image_np)

def add_contrast_change_noise(image, factor=1.5):
    """
    Add contrast change noise to an image.

    Parameters:
    image (PIL.Image): The input image.
    factor (float): The factor by which to change the contrast.
                    Values > 1 will increase contrast, values < 1 will decrease contrast.

    Returns:
    PIL.Image: The image with changed contrast.
    """
    image_np = np.array(image)

    # Apply contrast change
    mean = np.mean(image_np, axis=(0, 1), keepdims=True)
    contrast_image_np = np.clip((image_np - mean) * factor + mean, 0, 255).astype(np.uint8)

    return Image.fromarray(contrast_image_np)



