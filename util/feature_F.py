from util.img_util import Record

from util import (
    np,
    cv2,
    frangi
)

def measure_streaks(record: 'Record') -> float:
    
    """
    Calculate the streak score for a given Record.
    If the lesion area is too small to analyze, returns 0.0.
    If no contours are found in the lesion mask, returns 0.0.
    The streak score is a measure of the presence of streaks in the lesion area.
    It is calculated as the mean of the vessel-like pixels in the border ring around the lesion.
    
    The streak score is computed based on the grayscale image, blurred image, and lesion mask.
    It involves the following steps:
    1. Extract the grayscale image, blurred image, and lesion mask from the record.
    2. Find the largest contour in the lesion mask to determine the area of interest.
    3. Create a filled lesion mask based on the largest contour.
    4. Create a border ring around the lesion mask.
    5. Apply the Frangi filter to the normalized grayscale image to enhance line-like structures.
    6. Normalize the Frangi output.
    7. Apply the border ring mask to the normalized vesselness image.
    8. Calculate the streak score as the mean of the vessel-like pixels in the ring.
    Args:
        record (Record): The Record object containing image data.
    Returns:
        float: The streak score, which is the mean of vessel-like pixels in the border ring.
    Raises:
        ValueError: If the lesion mask is not found in the record's image data.
    """

    gray = record.image_data.get("grayscale_img")

    lesion_mask = record.image_data.get("original_mask")

    # Get largest contour
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    lesion_contour = max(contours, key=cv2.contourArea)
    lesion_area = cv2.contourArea(lesion_contour)
    
    lesion_area = np.sum(lesion_mask > 0)
    if lesion_area < 10:  # Too small to analyze
        print("-----------------------------------------------------Lesion area too small to analyze:", lesion_area)
        return 0.0

    # Create border ring
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(lesion_mask, kernel, iterations=1)
    border_ring = cv2.subtract(dilated, lesion_mask)

    # Apply Frangi filter to enhance line-like structures
    norm_gray = gray / 255.0
    vesselness = frangi(norm_gray)

    # Normalize Frangi output
    vesselness_norm = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)

    # Ensure both arrays have the same size
    if vesselness_norm.shape != border_ring.shape:
        # Resize border_ring to match vesselness_norm's dimensions
        border_ring = cv2.resize(border_ring, (vesselness_norm.shape[1], vesselness_norm.shape[0]))

    # Apply ring mask
    streak_area = vesselness_norm * (border_ring / 255.0)

    # Sum of vessel-like pixels in the ring
    streak_score = np.mean(streak_area)

    return float(streak_score)