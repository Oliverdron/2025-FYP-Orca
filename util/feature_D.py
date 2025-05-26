#Hair feature extraction code - Etele

from util.img_util import Record
from util import (
    np,
    cv2,
    frangi
)


def hair_feat_extraction(record: 'Record'):
    
    """
    The function computes the hairiness score based on the grayscale image and the blackhat image.
    The hairiness score is calculated as a combination of two metrics:
    1) The ratio of hair pixels to total pixels in the hair mask.
    2) The mean vesselness of the grayscale image using the Frangi filter.
    The final hairiness score is the average of these two metrics.
    The function modifies the record's features dictionary by adding a new key "hair_label" with the computed score.
    This function is designed to be used with a Record instance that contains the necessary image data.
    Args:
        record (Record): Record instance containing every bit of information about the image
        with the grayscale image, blackhat image, and hair mask.
    Returns:
        None: The function modifies the record in place, adding the hairiness score to its features.
    """    
    img = record.image_data["original_img"]
    gray = record.image_data["grayscale_img"]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blackhat = record.image_data["blackhat_img"]
    hair_mask = record.image_data["threshold_hair_mask"]
    inpainted = record.image_data["inpainted_img"]


    # Method 1:
    # Compute the hairiness score based on the difference between the original image and the inpainted image
    hair_only = cv2.absdiff(img, inpainted)
    gray_hair = cv2.cvtColor(hair_only, cv2.COLOR_BGR2GRAY)
    _, hair_cleaned = cv2.threshold(gray_hair, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    score = np.sum(hair_cleaned == 255) / hair_cleaned.size
    
    
    # Method 2:
    # Calculate the hairiness score based on the hair mask
    hair_pixels = np.sum(hair_mask > 0)
    total_pixels = hair_mask.size
    hairiness_score1 = hair_pixels / total_pixels
    
    # Apply the Frangi filter to the grayscale image
    # to compute the vesselness score
    vesselness = frangi(gray)
    hairiness_score2 = np.mean(vesselness)
    score = hairiness_score1 * 0.5 + hairiness_score2 * 0.5
    
    
    # Add the hairiness score to the record's features
    record.features["hair_label"] = score

