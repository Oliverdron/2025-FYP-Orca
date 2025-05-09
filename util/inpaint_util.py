from util import cv2

def removeHair(img_org, img_gray, kernel_size=25, threshold=10, radius=3):
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
    """
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out
