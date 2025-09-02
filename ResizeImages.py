import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(256, 256)):
    """
    Resize all images in a folder to the given size and save them to another folder.
    
    Args:
        input_folder (str): Path to folder containing images.
        output_folder (str): Path to folder where resized images will be saved.
        size (tuple): Desired output size (width, height).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if not os.path.isfile(input_path):
            continue  # Skip non-files

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")  # ensure consistency
                img = img.resize(size, Image.Resampling.LANCZOS)

                output_path = os.path.join(output_folder, filename)
                img.save(output_path, "JPEG", quality=95)

                print(f"Saved resized image: {output_path}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")


if __name__ == "__main__":
    input_folder = "DDR/filtered_procesed_train"     # Change this to your input folder
    output_folder = "DDR/resized_images"  # Change this to your output folder
    resize_images(input_folder, output_folder)
