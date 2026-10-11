import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Support both Dataset and dataset directory casing
CAMPUS_DIR = PROJECT_ROOT / "Dataset" / "campus"
if not CAMPUS_DIR.exists():
    CAMPUS_DIR = PROJECT_ROOT / "dataset" / "campus"

IMAGES_DIR = CAMPUS_DIR / "images"
ANNOTATIONS_DIR = CAMPUS_DIR / "annotations"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_directories():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def auto_generate_crack_mask(image_path, filter_size=15, threshold_offset=-2):
    """
    Generate an initial binary crack mask from a concrete surface photograph
    using morphological black-hat filtering and adaptive thresholding.
    White (255) = Crack, Black (0) = Background.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Black-hat transform isolates thin dark features (cracks) on lighter concrete
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filter_size, filter_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        filter_size if filter_size % 2 == 1 else filter_size + 1,
        threshold_offset
    )

    # Morphological opening to suppress salt-and-pepper texture noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)

    # Remove tiny disconnected specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 15:  # Keep components with at least 15 pixels
            mask[labels == i] = 255

    return mask


def process_unannotated_campus_images():
    """
    Scan Dataset/campus/images/ and generate candidate masks for any
    images lacking a corresponding mask in Dataset/campus/annotations/.
    """
    ensure_directories()
    images = [f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print(f"No images found in {IMAGES_DIR}.")
        print("Place your campus concrete photographs in that folder first.")
        return 0

    generated_count = 0
    for img_path in images:
        mask_path = ANNOTATIONS_DIR / f"{img_path.stem}.png"
        if not mask_path.exists():
            mask = auto_generate_crack_mask(img_path)
            cv2.imwrite(str(mask_path), mask)
            generated_count += 1
            print(f"Generated candidate mask: {mask_path.name}")

    print(f"\nProcessing complete: {generated_count} masks created.")
    return generated_count


def bootstrap_starter_set(num_samples=30):
    """
    Bootstrap an initial starter set of paired concrete crack images and
    binary masks for initial prototype development and U-Net training.
    """
    ensure_directories()
    sdnet_dir = PROJECT_ROOT / "Dataset" / "SDNET2018"
    if not sdnet_dir.exists():
        sdnet_dir = PROJECT_ROOT / "dataset" / "SDNET2018"

    if not sdnet_dir.exists():
        print(f"SDNET2018 directory not found at {sdnet_dir}.")
        return 0

    # Pick samples from cracked subfolders across Decks, Pavements, and Walls
    cracked_folders = [
        sdnet_dir / "D" / "CD",
        sdnet_dir / "P" / "CP",
        sdnet_dir / "W" / "CW",
    ]

    copied = 0
    for folder in cracked_folders:
        if not folder.exists():
            continue
        files = sorted(list(folder.glob("*.jpg")))
        samples_per_folder = max(1, num_samples // len(cracked_folders))
        # Take spaced samples to get variety
        step = max(1, len(files) // samples_per_folder)
        selected = files[::step][:samples_per_folder]

        for idx, src_file in enumerate(selected):
            dest_img = IMAGES_DIR / f"campus_demo_{src_file.parent.name}_{idx + 1:02d}.jpg"
            dest_mask = ANNOTATIONS_DIR / f"campus_demo_{src_file.parent.name}_{idx + 1:02d}.png"

            shutil.copy2(src_file, dest_img)
            mask = auto_generate_crack_mask(dest_img)
            cv2.imwrite(str(dest_mask), mask)
            copied += 1

    print(f"Successfully bootstrapped {copied} paired demonstration images and masks in {CAMPUS_DIR}.")
    return copied


def verify_campus_pairs():
    """Verify that every image in images/ has a corresponding binary mask in annotations/."""
    ensure_directories()
    images = sorted([f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS])
    masks = sorted(list(ANNOTATIONS_DIR.glob("*.png")))

    print("\n--- Campus Dataset Verification ---")
    print(f"Images found:      {len(images)} in {IMAGES_DIR}")
    print(f"Annotations found: {len(masks)} in {ANNOTATIONS_DIR}")

    matched_pairs = []
    missing_masks = []

    for img in images:
        expected_mask = ANNOTATIONS_DIR / f"{img.stem}.png"
        if expected_mask.exists():
            matched_pairs.append((img, expected_mask))
        else:
            missing_masks.append(img)

    print(f"Valid matched pairs: {len(matched_pairs)}")
    if missing_masks:
        print(f"Images missing masks ({len(missing_masks)}): {[f.name for f in missing_masks[:5]]}...")

    # Validate mask format and pixel values
    invalid_masks = []
    for _, mask_path in matched_pairs:
        mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_arr is None:
            invalid_masks.append(f"{mask_path.name} (unreadable)")
            continue
        unique_vals = set(np.unique(mask_arr))
        # Expect binary 0 and 255 (or single class if completely uncracked)
        if not unique_vals.issubset({0, 255}):
            invalid_masks.append(f"{mask_path.name} (contains non-binary values: {unique_vals})")

    if invalid_masks:
        print(f"Warning: {len(invalid_masks)} masks are non-binary.")
    else:
        print("All matched masks are valid binary grayscale images.")

    return len(matched_pairs)


def main():
    parser = argparse.ArgumentParser(description="Manage and annotate campus segmentation dataset")
    parser.add_argument("--verify", action="store_true", help="Verify existing campus image-mask pairs")
    parser.add_argument("--auto-annotate", action="store_true", help="Auto-generate candidate masks for images in campus/images/")
    parser.add_argument("--bootstrap-starter", type=int, default=0, help="Create N paired starter samples for prototyping")
    args = parser.parse_args()

    ensure_directories()

    if args.bootstrap_starter > 0:
        bootstrap_starter_set(args.bootstrap_starter)
        verify_campus_pairs()
    elif args.auto_annotate:
        process_unannotated_campus_images()
        verify_campus_pairs()
    else:
        verify_campus_pairs()


if __name__ == "__main__":
    main()

