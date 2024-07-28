import os
import shutil
import random
import logging
from PIL import Image, UnidentifiedImageError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(image_path, output_path, size=(1024, 1024)):
    """Resize the image to the specified size and save it to the output path."""
    try:
        with Image.open(image_path) as img:
            resized_img = img.resize(size, Image.LANCZOS)
            resized_img.save(output_path)
            logging.info(f"Resized and saved image to {output_path}")
    except (IOError, UnidentifiedImageError) as e:
        logging.error(f"Could not process file {image_path}: {e}")

def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return filename.lower().endswith(valid_extensions)

def split_dataset(train_dataset_folder, train_ratio):
    """
    Split the dataset into training and testing sets and resize the images.
    
    Args:
        train_dataset_folder (str): Path to the train dataset folder containing 'imgs' and 'masks' subfolders.
        train_ratio (float): Ratio of data to be used for training.
    """
    images_folder = os.path.join(train_dataset_folder, 'imgs')
    masks_folder = os.path.join(train_dataset_folder, 'masks')
    
    preprocessed_folder = 'preprocessed_dataset'
    
    if os.path.exists(preprocessed_folder):
        user_input = input(f"The folder {preprocessed_folder} already exists. Do you want to remove it and create a new one? (Y/N): ")
        if user_input.lower() == 'y':
            shutil.rmtree(preprocessed_folder)
            logging.info(f"Removed existing preprocessed dataset folder: {preprocessed_folder}")
        else:
            logging.info("Operation cancelled by the user.")
            return
    
    train_folder = os.path.join(preprocessed_folder, 'train')
    test_folder = os.path.join(preprocessed_folder, 'test')
    
    # Create necessary directories
    create_directories([train_folder, test_folder])
    
    train_images_folder, train_masks_folder = create_subdirectories(train_folder)
    test_images_folder, test_masks_folder = create_subdirectories(test_folder)

    # List and shuffle image files
    image_files = [f for f in os.listdir(images_folder) if is_image_file(f)]
    random.shuffle(image_files)

    split_index = int(len(image_files) * train_ratio)
    train_files, test_files = image_files[:split_index], image_files[split_index:]

    process_files(train_files, images_folder, masks_folder, train_images_folder, train_masks_folder)
    process_files(test_files, images_folder, masks_folder, test_images_folder, test_masks_folder)

    logging.info(f"Train set: {len(train_files)} images")
    logging.info(f"Test set: {len(test_files)} images")

def create_directories(directories):
    """Create directories if they do not exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def create_subdirectories(base_folder):
    """
    Create images and masks subdirectories in the base folder.
    
    Args:
        base_folder (str): The base folder path.
    
    Returns:
        tuple: Paths to the images and masks subdirectories.
    """
    images_folder = os.path.join(base_folder, 'images')
    masks_folder = os.path.join(base_folder, 'masks')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    return images_folder, masks_folder

def process_files(files, images_folder, masks_folder, target_images_folder, target_masks_folder):
    """Process the files by resizing and saving them to target folders."""
    for file in files:
        resize_image(os.path.join(images_folder, file), os.path.join(target_images_folder, file))
        resize_image(os.path.join(masks_folder, file), os.path.join(target_masks_folder, file))

def flip_images_in_folder(folder_path, flip_horizontal=True, flip_vertical=False):
    """
    Flip all images in the specified folder and save them.
    
    Args:
        folder_path (str): Path to the folder containing images to flip.
        flip_horizontal (bool): Whether to flip the images horizontally.
        flip_vertical (bool): Whether to flip the images vertically.
    """
    if not os.path.exists(folder_path):
        logging.error(f"The folder {folder_path} does not exist.")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and is_image_file(filename):
            try:
                with Image.open(file_path) as img:
                    if flip_horizontal:
                        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        save_flipped_image(file_path, flipped_img, "_h")
                    if flip_vertical:
                        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        save_flipped_image(file_path, flipped_img, "_v")
            except UnidentifiedImageError:
                logging.error(f"Could not identify image file {file_path}")

def save_flipped_image(file_path, flipped_img, suffix):
    """Save the flipped image with a suffix in the filename."""
    name, ext = os.path.splitext(os.path.basename(file_path))
    new_filename = f"{name}{suffix}{ext}"
    new_file_path = os.path.join(os.path.dirname(file_path), new_filename)
    flipped_img.save(new_file_path)
    logging.info(f"Saved flipped image as {new_file_path}")

def main():
    # Define the path to your train dataset
    train_dataset_folder = './train_dataset'

    # Define the split ratio
    train_ratio = 0.95

    # Define flipping options
    flip_horizontal = True
    flip_vertical = False

    # Split the dataset
    split_dataset(train_dataset_folder, train_ratio)

    # Define the paths to your folders for flipping images
    folder1 = 'preprocessed_dataset/train/images'
    folder2 = 'preprocessed_dataset/train/masks'

    # Flip images in both folders
    flip_images_in_folder(folder1, flip_horizontal, flip_vertical)
    flip_images_in_folder(folder2, flip_horizontal, flip_vertical)

if __name__ == "__main__":
    main()
