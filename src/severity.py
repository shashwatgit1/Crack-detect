import cv2
import numpy as np


def estimate_severity(mask, min_component_area=5):
    """
    Estimate image-based crack severity from a binary mask.

    White pixels (255) represent predicted crack pixels.
    Calculates individual and total crack areas across all detected cracks.
    The returned category is a visual indicator, not a structural diagnosis.
    """
    if mask is None:
        raise ValueError("A crack mask is required.")

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    binary = (mask > 127).astype(np.uint8)

    # Detect distinct crack components using 8-connectivity
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)
    component_count = 0
    individual_crack_areas = []

    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            cleaned[labels == i] = 1
            component_count += 1
            individual_crack_areas.append(area)

    # Sort individual crack areas descending (largest crack first)
    individual_crack_areas.sort(reverse=True)

    # Total area covered in cracks is the sum of all individual crack areas
    total_crack_pixels = int(sum(individual_crack_areas))
    total_surface_pixels = int(cleaned.size)
    area_ratio = total_crack_pixels / max(total_surface_pixels, 1)

    # Total length proxy estimated by skeleton pixel count
    skeleton = cleaned.copy()
    skeleton_image = (skeleton * 255).astype(np.uint8)

    # Morphological skeletonization using iterative erosion
    skel = np.zeros_like(skeleton_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = skeleton_image.copy()

    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, residue)
        work = eroded

    estimated_length_pixels = int(cv2.countNonZero(skel))

    if total_crack_pixels == 0:
        category = "No visible crack detected"
    elif area_ratio < 0.005:
        category = "Minor"
    elif area_ratio < 0.02:
        category = "Moderate"
    else:
        category = "High"

    return {
        "severity": category,
        "total_crack_area_pixels": total_crack_pixels,
        "crack_area_pixels": total_crack_pixels,  # Backwards compatibility
        "individual_crack_areas": individual_crack_areas,
        "num_cracks": component_count,
        "connected_components": component_count,
        "total_surface_pixels": total_surface_pixels,
        "crack_area_percentage": round(area_ratio * 100, 3),
        "estimated_length_pixels": estimated_length_pixels,
        "note": (
            "Visual estimate only. Calibrated dimensions and professional "
            "inspection are required for engineering severity."
        )
    }


def estimate_severity_from_probability(probability):
    """
    Fallback when a segmentation mask is unavailable.
    This is a detection-confidence indicator, NOT crack severity.
    """
    if probability < 0.5:
        return "No crack detected"
    return "Crack detected; severity unavailable without a mask"
