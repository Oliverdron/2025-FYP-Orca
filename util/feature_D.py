from util.img_util import Record
from util import (
    np,
    cv2
)


def hair_feat_extraction(record: 'Record'):
    """Extract a discrete hair‐density category from the Record’s hair mask.

Compute hair score by:
    1) Retrieving the thresholded hair mask from record.image_data["threshold_hair_mask"]
    2) Labeling connected components with cv2.connectedComponentsWithStats
    3) Building a clean mask by keeping only components whose area > 30 pixels
    4) Computing coverage as (number of hair pixels in clean mask) / (total pixels)
    5) Binning the coverage fraction:
        – return 2 if > 0.70 (high hair coverage)
        – return 1 if > 0.40 (medium hair coverage)
        – return 0 otherwise (low hair coverage)

Args:
    rec (Record): Record instance containing all preprocessed images and masks

Returns:
    int: hair‐density category (0 = low, 1 = medium, 2 = high)
"""
    
    hair_mask = record.image_data["threshold_hair_mask"]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask)
    clean_mask = np.zeros_like(hair_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 30:  # keep only long hair-like structures
            clean_mask[labels == i] = 255
    
    # Calculate the hair score based on the cleaned mask
    score = np.sum(clean_mask == 255) / clean_mask.size
    if score > 0.7:
        return 2
    elif score > 0.4:
        return 1
    else:
        return 0