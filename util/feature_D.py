from util.img_util import Record
from util import (
    np,
    cv2
)


def hair_feat_extraction(record: 'Record'):
    
    
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