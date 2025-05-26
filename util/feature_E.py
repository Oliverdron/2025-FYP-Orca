from util.img_util import Record
from util import (
    np,
    color,
    exposure,
    frangi
)

def vascular_score(record: 'Record') -> float:
    
    """Calculate the vascular score for a given Record.
    The vascular score is computed based on the red channel of the original RGB image,
    the grayscale image, and the vesselness of the grayscale image.
    Args:
        record (Record): The Record object containing image data.
    Returns:
        float: The vascular score, which is the sum of the masked vesselness values divided by 1000.
    """
    
    rgb_image = record.image_data.get("original_img")

    if rgb_image is None:
        print("[WARNING] - vascular_score - No original image found.")
        return 0.0
    
    # Enhance the red channel using gamma correction
    red_channel = rgb_image[:, :, 0]
    enhanced_red = exposure.adjust_gamma(red_channel, gamma=0.8)
    rgb_enhanced = rgb_image.copy()
    rgb_enhanced[:, :, 0] = enhanced_red

    # Convert the enhanced RGB image to HSV color space
    hsv = color.rgb2hsv(rgb_enhanced)
    
    # Define the lower and upper bounds for the red color in HSV
    # The red color in HSV can wrap around, so we define two ranges
    # 1) From 0 to 25 degrees (0 to 1 in hue)
    
    lower_red1 = np.array([0.0, 0.4, 0.0])
    upper_red1 = np.array([25/360.0, 1.0, 1.0])
    
    # 2) From 330 degrees to 360 degrees (0.9167 to 1 in hue)
    lower_red2 = np.array([330/360.0, 0.4, 0.0])
    upper_red2 = np.array([1.0, 1.0, 1.0])

    # Create masks for the red color ranges
    mask1 = np.logical_and(np.all(hsv >= lower_red1, axis=-1), np.all(hsv <= upper_red1, axis=-1))
    mask2 = np.logical_and(np.all(hsv >= lower_red2, axis=-1), np.all(hsv <= upper_red2, axis=-1))
    red_mask = np.logical_or(mask1, mask2)

    # getting gray img
    gray_img = record.image_data.get("grayscale_img")
    if gray_img is None:
        print("[WARNING] - vascular_score - No grayscale image found.")
        return 0.0

    # calculate vesselness using Frangi filter
    vesselness = frangi(gray_img) # binary mask

    # mask the vesselness with the red mask
    masked_vesselness = vesselness * red_mask

    score = float(np.sum(masked_vesselness)) / 1000.0

    return score
