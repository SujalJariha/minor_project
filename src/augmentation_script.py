import os
import csv
import cv2
import numpy as np

BASE_DIR = "DryFruits_Dataset"
DRY_FRUITS = ["Almond", "Cashew", "Raisin", "Walnut", "Pistachio", "Date"]
GRADES = ["Grade_A", "Grade_B", "Grade_C"]
OUTPUT_SIZE = (720, 720)
ROTATION_ANGLES = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
BC_FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
# ----------------------

def create_folder_structure():
    """Creates the dataset folder structure."""
    for fruit in DRY_FRUITS:
        for grade in GRADES:
            path = os.path.join(BASE_DIR, fruit, grade)
            os.makedirs(path, exist_ok=True)

def adjust_brightness_contrast(image, brightness_factor=1.0, contrast_factor=1.0):
    """
    Adjust brightness and contrast.
    brightness_factor: scales pixel intensity
    contrast_factor: scales difference from mean intensity
    """
    img = image.astype(np.float32)
    mean = np.mean(img)
    img = (img - mean) * contrast_factor + mean
    img = img * brightness_factor
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def zoom_crop(image, zoom_range=(1.1, 1.3)):
    """Random zoom and crop back to 720x720."""
    h, w = image.shape[:2]
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])

    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - OUTPUT_SIZE[0]) // 2
    start_y = (new_h - OUTPUT_SIZE[1]) // 2
    cropped = resized[start_y:start_y+OUTPUT_SIZE[1], start_x:start_x+OUTPUT_SIZE[0]]
    return cropped

def rotate_image(image, angle):
    """Rotate image at given angle while keeping size 720x720."""
    h, w = OUTPUT_SIZE
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rotated

def process_and_augment_images(input_dir):
    """Processes all images from input_dir and saves augmented images into BASE_DIR."""
    label_file = os.path.join(BASE_DIR, "labels.csv")
    
    with open(label_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "type", "grade"])
        
        for fruit in DRY_FRUITS:
            for grade in GRADES:
                input_path = os.path.join(input_dir, fruit, grade)
                output_path = os.path.join(BASE_DIR, fruit, grade)
                
                if not os.path.exists(input_path):
                    continue
                
                count = 1
                for img_file in os.listdir(input_path):
                    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    
                    img_path = os.path.join(input_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error reading {img_path}")
                        continue
                    
                    # Resize to 720x720
                    img_resized = cv2.resize(img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
                    
                    # Save original
                    filename = f"{fruit.lower()}_{grade.split('_')[1]}_{count:03d}.jpg"
                    cv2.imwrite(os.path.join(output_path, filename), img_resized)
                    writer.writerow([filename, fruit, grade])
                    count += 1
                    
                    # Rotations
                    for angle in ROTATION_ANGLES:
                        rotated = rotate_image(img_resized, angle)
                        filename = f"{fruit.lower()}_{grade.split('_')[1]}_{count:03d}.jpg"
                        cv2.imwrite(os.path.join(output_path, filename), rotated)
                        writer.writerow([filename, fruit, grade])
                        count += 1
                    
                    # Flip
                    flipped = cv2.flip(img_resized, 1)
                    filename = f"{fruit.lower()}_{grade.split('_')[1]}_{count:03d}.jpg"
                    cv2.imwrite(os.path.join(output_path, filename), flipped)
                    writer.writerow([filename, fruit, grade])
                    count += 1
                    
                    # Brightness/Contrast
                    for bf in BC_FACTORS:
                        for cf in BC_FACTORS:
                            bc_img = adjust_brightness_contrast(img_resized, bf, cf)
                            filename = f"{fruit.lower()}_{grade.split('_')[1]}_{count:03d}.jpg"
                            cv2.imwrite(os.path.join(output_path, filename), bc_img)
                            writer.writerow([filename, fruit, grade])
                            count += 1
                    
                    # Zoom & Crop
                    zoomed = zoom_crop(img_resized)
                    filename = f"{fruit.lower()}_{grade.split('_')[1]}_{count:03d}.jpg"
                    cv2.imwrite(os.path.join(output_path, filename), zoomed)
                    writer.writerow([filename, fruit, grade])
                    count += 1

    print(f"✅ Dataset created in '{BASE_DIR}' with labels.csv")

if __name__ == "__main__":
    create_folder_structure()
    INPUT_IMAGES_DIR = "input_images"
    process_and_augment_images(INPUT_IMAGES_DIR)