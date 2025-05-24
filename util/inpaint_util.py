from util import (
    cv2, np
)

def removeHair(record: any, kernel_size=25, threshold=15, radius=5):
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
    blackhat = cv2.morphologyEx(record.img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(record.img_rgb, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out
