from util.img_util import Record

from util import (
    np,
    color,
    exposure,
    frangi
)

def vascular_score(record: 'Record', gamma: float = 0.8) -> float:
    """
        The vascular score is computed based on the red channel of the original RGB image, the grayscale image, and the vesselness of the grayscale image.

        Args:
            record (Record): The Record object containing image data
            gamma (float): The gamma value for gamma correction applied to the red channel of the RGB image (default: 0.8)

        Returns:
            float: The vascular score, which is the sum of the masked vesselness values divided by 1000
    """
    # Retrieve the RGB image from the record object
    rgb_image = record.image_data["original_img"]
    
    # Extract the red channel (first channel of the RGB image)
    red_channel = rgb_image[:, :, 0]

    # Apply gamma correction to enhance the red channel (lower gamma values darken the image)
    enhanced_red = exposure.adjust_gamma(red_channel, gamma)

    # Make a copy of the original image to preserve other channels
    rgb_enhanced = rgb_image.copy()
    # Update the red channel in the enhanced image
    rgb_enhanced[:, :, 0] = enhanced_red

    # HSV conversion helps isolate the red color range more effectively
    # Makes color-based segmentation easier as hue (H) directly represents the color
    hsv = color.rgb2hsv(rgb_enhanced)
    
    # Define the lower and upper bounds for the red color in HSV
    # The red color in HSV can wrap around, so we define two ranges
    # - From 0 to 25 degrees (0 to 1 in hue)
    lower_red1 = np.array([0.0, 0.4, 0.0])
    upper_red1 = np.array([25/360.0, 1.0, 1.0])
    
    # - From 330 degrees to 360 degrees (0.9167 to 1 in hue)
    lower_red2 = np.array([330/360.0, 0.4, 0.0])
    upper_red2 = np.array([1.0, 1.0, 1.0])

    # Check whether each pixel in the HSV image lies within one of our specified range
    # (It has to be higher than the lower bound and lower than the upper bound)
    mask1 = np.logical_and(np.all(hsv >= lower_red1, axis=-1), np.all(hsv <= upper_red1, axis=-1))
    mask2 = np.logical_and(np.all(hsv >= lower_red2, axis=-1), np.all(hsv <= upper_red2, axis=-1))

    # Then, combine the two masks to create a single mask for red pixels
    red_mask = np.logical_or(mask1, mask2)

    # Retrieve the grayscale image from the record object
    gray_img = record.image_data["grayscale_img"]

    # Apply the Frangi filter to the grayscale image to detect vessels
    # The filter focuses on structures with the following characteristics:
    #   - Long and narrow (elongated), which are typical of blood vessels
    #   - High curvature or tube-like structures
    vesselness = frangi(gray_img)

    # Element-wise multiplication to keep only vessels in red areas
    masked_vesselness = vesselness * red_mask

    # Calculate the final score based on the masked vesselness
    # The score is divided by 1000 for normalization
    score = float(np.sum(masked_vesselness)) / 1000.0

    return score