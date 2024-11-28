import os
import numpy as np
from PIL import Image, ImageOps

def load_and_convert_to_grayscale_alpha(image_path):
    """Load an image and convert it to grayscale with alpha channel."""
    image = Image.open(image_path).convert("LA")
    gray_channel, alpha_channel = image.split()
    
    # Apply histogram equalization to the grayscale channel
    equalized_gray = ImageOps.equalize(gray_channel)
    
    # Recombine the equalized grayscale channel with the alpha channel
    equalized_image = Image.merge('LA', (equalized_gray, alpha_channel))# Convert to grayscale with alpha
    return equalized_image

def resize_image(image, size=(512, 512)):
    """Resize the image to the specified size."""
    return image.resize(size, Image.LANCZOS)

def combine_masks(mask_paths, size=(512, 512)):
    """Combine multiple mask images into one by overlaying using maximum intensity."""
    
    # Initialize an empty mask with zeros and set dtype to uint8
    combined_mask = np.zeros(size, dtype=np.uint8)

    for path in mask_paths:
        # Load each mask as a single-channel binary mask and resize
        mask = Image.open(path).convert("L")  # Convert to single-channel (binary)
        mask = resize_image(mask, size)

        # Convert mask to numpy array for combining
        mask_array = np.array(mask)

        # Ensure mask_array is of type uint8
        if mask_array.dtype != np.uint8:
            mask_array = mask_array.astype(np.uint8)

        # Combine using maximum intensity to ensure overlaps are preserved
        combined_mask = np.maximum(combined_mask, mask_array)
    
    return combined_mask

def get_placeholder_image(size=(512, 512)):
    """Create an empty black image of specified size."""
    return Image.fromarray(np.zeros(size, dtype=np.uint8))

def save_processed_image(image, output_dir, case_number, category):
    """Save the processed image in the appropriate category folder inside 'Clean Data'."""
    category_dir = os.path.join(output_dir, category.capitalize() + "s")  # "Images", "Tumors", or "Others"
    os.makedirs(category_dir, exist_ok=True)
    output_path = os.path.join(category_dir, f"case{case_number}_{category}.png")
    image.save(output_path)
    return output_path

def generate_background_mask(tumor_array, other_array):
    """Generate a background mask based on tumor and other masks."""
    background_mask_array = np.ones((512, 512), dtype=np.uint8) * 255  # Start with all white (255)
    
    if np.any(tumor_array):  # If there is a tumor present
        background_mask_array[tumor_array] = 0  # Set tumor areas to black (0)
        
    if np.any(other_array):  # If there are other areas present
        background_mask_array[other_array > 0] = 0  # Set other areas to black (0)

    return background_mask_array

def process_case_images(case_number, file1_dir, file2_dir, file3_dir, output_dir):
    """Process images for a single case number across three categories."""
    
    # Paths for each category of image
    image_path = os.path.join(file1_dir, f"case{case_number}.png")
    tumor_path = os.path.join(file2_dir, f"case{case_number}_tumor.png")
    
    other_paths = [os.path.join(file3_dir, f"case{case_number}_other{i}.png") 
                   for i in range(1, 10) if os.path.exists(os.path.join(file3_dir, f"case{case_number}_other{i}.png"))]
    
    # Process and save the main image as grayscale with alpha
    if os.path.exists(image_path):
        main_image = load_and_convert_to_grayscale_alpha(image_path)
        main_image = resize_image(main_image)
    else:
        main_image = get_placeholder_image()
    
    save_processed_image(main_image, output_dir, case_number, "image")
    
    # Process and save the tumor image as a binary mask
    if os.path.exists(tumor_path):
        tumor_image = Image.open(tumor_path).convert("L")  # Keep as single-channel binary
        tumor_image = resize_image(tumor_image)
        tumor_array = np.array(tumor_image) > 0  # Create binary array (True/False)
    else:
        tumor_array = np.zeros((512, 512), dtype=bool)  # All-black placeholder
    
    save_processed_image(tumor_image if os.path.exists(tumor_path) else get_placeholder_image(), output_dir, case_number, "tumor")
    
    # Process and save the combined "other" masks
    if other_paths:
        combined_other_array = combine_masks(other_paths)
        other_binary_array = combined_other_array > 0  # Create binary array for others
    else:
        combined_other_array = np.zeros((512, 512), dtype=np.uint8)  # All-black placeholder
        other_binary_array = np.zeros((512, 512), dtype=bool)         # All-black placeholder
    
    save_processed_image(Image.fromarray(combined_other_array), output_dir, case_number, "other")

    # Generate the background mask using the tumor and other masks
    background_mask_array = generate_background_mask(tumor_array, other_binary_array)
    
    background_mask_image = Image.fromarray(background_mask_array)  # Convert back to image
    
    background_category_dir = os.path.join(output_dir, "Backgrounds")  # New directory for backgrounds
    os.makedirs(background_category_dir, exist_ok=True)
    
    background_output_path = os.path.join(background_category_dir, f"case{case_number}_background.png")
    background_mask_image.save(background_output_path)

def process_all_cases(case_numbers, file1_dir, file2_dir, file3_dir, output_dir):
    """Loop over all case numbers and process each."""
    for case_number in case_numbers:
        process_case_images(case_number, file1_dir, file2_dir, file3_dir, output_dir)

# Example usage
if __name__ == "__main__":
    case_numbers = [f"{i:03}" for i in range(1, 257)]  # List of case numbers to process
    file1_dir = "Datasets/Breast Ultrasound Dataset/Processed Datasets/Images"
    file2_dir = "Datasets/Breast Ultrasound Dataset/Processed Datasets/Tumors"
    file3_dir = "Datasets/Breast Ultrasound Dataset/Processed Datasets/Other"
    output_dir = "Datasets\Breast Ultrasound Dataset\Breast Clean Data Equalized"  # Main directory for cleaned data

    process_all_cases(case_numbers, file1_dir, file2_dir, file3_dir, output_dir)