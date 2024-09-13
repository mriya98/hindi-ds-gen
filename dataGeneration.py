'''
A dataset is generated using fonts supporting Hindi characters in Devanagiri script.
Augmentations are applied to add colours, scaling, and distortion to create a varied dataset.
'''
from albumentations import ElasticTransform
import config
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import GaussianBlur
import uuid

def generate_dataset_from_fonts():
    '''
    Input: None
    Output: A dataset made up of images of Hindi characters and special symbols

    The images are generated using text fonts available in "./fonts" folder, and
    the generated dataset is stored in "./glyph_dataset". The source and output
    directories can be modified on config.py to location in your local drive.

    The images are generated in different sizes to maintain variance. The geberated
    image of each class is stored in a folder with the character unicode as its name.
    '''
    # Root directory for the dataset
    dataset_dir = config.GLPYH_DATASET_DIR
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Define Unicode ranges for special characters and Indic scripts
    unicode_ranges = {
        "devanagari": (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, etc.)
        # Comment the ranges below if punctuations and other symbols are not required
        "math_symbols": (0x2200, 0x22FF),
        "currency_symbols": (0x20A0, 0x20CF),
        "punctuation": (0x2000, 0x206F),
    }

    # Get list of all TTF files
    font_folder = config.FONT_DIR
    font_files = [file for file in os.listdir(font_folder) if os.path.isfile(os.path.join(font_folder, file))]

    # For each font style, generate and save images with random filenames, skipping blank ones
    for font_style in font_files:
        font_path = os.path.join(font_folder, font_style)
        font = ImageFont.truetype(font_path, 64)
        for script, (start, end) in unicode_ranges.items():
            for code in range(start, end + 1):
                char = chr(code)
                unicode_str = f"U+{code:04X}"

                # Generate a random filename for the image using UUID
                random_filename = str(uuid.uuid4()) + ".png"

                # Create an image and draw the character
                width, height = font.getsize(char)
                img = Image.new("RGB", (width+20, height+20), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), char, font=font, fill='black')

                # Check if the character is rendered or not
                # Image is inverted to get the binding box around character if present
                invert_img = ImageOps.invert(img)
                bbox = invert_img.getbbox()
                if not (bbox is None):
                    # Create directory if it doesn't exist
                    char_dir = os.path.join(dataset_dir, unicode_str)
                    if not os.path.exists(char_dir):
                        os.makedirs(char_dir)
                    # Save the image with the random filename
                    img.save(os.path.join(char_dir, random_filename))
    return dataset_dir

def randomize_glyph_color(image):
    image = image.convert("RGBA")  # Ensure image is in RGBA mode

    # Generate random color for the glyph
    r, g, b = [random.randint(0, 255) for _ in range(3)]

    # Replace the color of the glyph (assuming non-white areas are glyphs)
    def change_color(pixel):
        # Check if the pixel is a tuple (RGBA or RGB mode)
        if isinstance(pixel, tuple):
            if len(pixel) == 4:  # RGBA mode
                r_orig, g_orig, b_orig, a = pixel
                if a > 0:  # If the pixel is not fully transparent
                    return (r, g, b, a)
            elif len(pixel) == 3:  # RGB mode
                r_orig, g_orig, b_orig = pixel
                return (r, g, b)
        # If the image is in grayscale (L mode) or any other single-channel mode
        elif isinstance(pixel, int):
            return random.randint(0, 255)  # Return a random grayscale value

        return pixel  # Return the original pixel if no changes are made

    # Apply the color change to the image
    image = image.point(change_color, mode='RGBA')

    return image.convert("RGB")  # Convert back to RGB

def elastic_transform(image):
    transform = ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=None)
    augmented = transform(image=np.array(image))['image']
    return Image.fromarray(augmented)

def add_gaussian_noise(image):
    img_array = np.array(image)
    mean = 0
    stddev = 10
    gaussian = np.random.normal(mean, stddev, img_array.shape)
    noisy_image = img_array + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)
    return Image.fromarray(noisy_image.astype('uint8'))

def change_background_color(image, color):
    image = image.convert("RGBA")
    background = Image.new('RGBA', image.size, color)
    combined = Image.alpha_composite(background, image)
    return combined.convert("RGB")

# Random background color augmentation
def random_bg_color(image):
    random_color = tuple(np.random.randint(0, 256, size=3))  # Random RGB color
    return change_background_color(image, random_color)

def getAugmentedDataset(dataset_dir, num_augmentations=5):
    '''
    Input: dataset directory path (suitable for ImageFolder)
    Output: dataset with augmentations applied

    Args:
        dataset_dir (str): Path to dataset directory
        num_augmentations (int): Minimum number of augmented image to generate for each original image
    '''
    # Define augmentation pipeline
    augmentations = transforms.Compose([
        transforms.RandomApply([transforms.Lambda(random_bg_color)], p=0.4),
        transforms.RandomApply([transforms.Lambda(lambda img: randomize_glyph_color(img))], p=0.5),
        transforms.RandomApply([transforms.Lambda(lambda img: elastic_transform(img))], p=0.2),
        transforms.RandomApply([transforms.Lambda(lambda img: add_gaussian_noise(img))], p=0.5),
        transforms.RandomApply([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        transforms.RandomApply([transforms.Resize((128, 128))], p=1),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-15, 15))], p=0.3),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.Lambda(lambda img: img.convert("RGB").point(lambda p: p * random.uniform(0.5, 1.5)))], p=0.4),
        transforms.ToTensor()
    ])

    # Create a custom dataset to apply augmentations multiple times
    class AugmentedImageFolder(ImageFolder):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            augmented_images = []
            for _ in range(num_augmentations):
                augmented_img = augmentations(img)
                augmented_images.append(augmented_img)
            return augmented_images, target

    dataset = AugmentedImageFolder(root=dataset_dir, transform=None)

    return dataset
