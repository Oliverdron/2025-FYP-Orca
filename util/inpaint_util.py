import cv2
import numpy as np


def removeHair(img_org, img_gray, kernel_size=25, threshold=14, radius=3):
    """
        Remove hair from the image using morphological filtering and inpainting
        
        Parameters:
            img_org (ndarray): The original RGB image
            img_gray (ndarray): The grayscale version of the original image
            kernel_size (int): Size of the kernel used for morphological operations
            threshold (int): Threshold value for binary thresholding
            radius (int): Radius for inpainting

        Returns:
            blackhat (ndarray): The blackhat filtered image
            thresh (ndarray): The thresholded mask of the image
            img_out (ndarray): The inpainted image with hair removed
            label (int): The label indicating the amount of hair in the image
    """
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)    
    
    hair_pixel_count = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]

    # Hair proportion
    hair_ratio = hair_pixel_count / total_pixels

    # Normalize to 0–2 range
    score = np.clip(hair_ratio * 6, 0, 2)
    #score = np.clip(np.sum(blackhat / 255) / (total_pixels/30), 0, 2) --- weight based on blackhat pixel count
    if score < 0.75:
        label = 0
    elif score < 1.5:
        label = 1
    else:
        label = 2

    return blackhat, thresh, img_out, label