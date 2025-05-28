# Image Pre-process & Evaluation Strategy:
# Step 1: Apply "Bilateral Denoising" -> Reduce noise while preserving edges and fine details
#         - Bilateral denoising works by considering both spatial distance and intensity difference, smoothing out noise without blurring edges
# Step 2: Calculate Structural Similarity Index -> If SSIM is below the threshold (0.8), proceed with denoising
#         - SSIM evaluates structural, luminance, and contrast similarity between images
# Step 3.1: Calculate Peak Signal-to-Noise Ratio -> If PSNR is below the threshold (20 dB), flag for further observation
#         - PSNR measures the quality of the denoised image by comparing it to the original
#         - If PSNR is low, it may indicate high noise levels, incorrect denoising parameters, or artifacts introduced during denoising
# Step 3.2: If PSNR is below threshold, try "Non-Local Means Denoising" (NLMD)
#         - NLMD applies non-local averaging based on pixel similarity, further reducing noise without affecting details
# Step 4: Enhance image quality using 'Contrast Limited Adaptive Histogram Equalization' (CLAHE)
#         - CLAHE improves image contrast by equalizing histograms locally, enhancing visibility of fine details such as skin lesions
#         - In short, uses 'AHE' but with bilinear interpolation at the borders, to give smooth overall view
# Step 5: Hair removal using 'Morphological Black-Hat Transformation' and inpainting
#         - Removes unwanted hair from the image while maintaining skin structure, using a combination of morphological operations and inpainting
# Step 6: Apply edge detection (Sobel and Canny) to identify the skin lesion boundary
#         - 'Sobel' detects edges by calculating gradients in the x and y directions
#         - 'Canny' refines these edges to produce clear, binary edge maps, which help in lesion segmentation

from util import (
    cv2, np, ssim
)

class PreProcessor:
    def __init__(self, record):
        """
            The PreProcessor class applies a series of pre-processing steps to an image from the Record class instance.
            This class handles operations such as denoising, hair removal, and image enhancement to improve the quality of the input image.

            Param:
                - record (Record): Record class instance containing every bit of information

            Methods:
                - ssim (original_image, denoised_image): Calculates the Structural Similarity Index (SSIM)
                - psnr (original_image, denoised_image): Calculates the Peak Signal-to-Noise Ratio (PSNR)
                - bilateral_denoising (image, d, sigmaColor, sigmaSpace): Applies bilateral denoising on the image
                - nlmd (image, h, templateWindowSize, searchWindowSize): Applies Non-Local Means Denoising (NLMD) on the image
                - removeHair (img_org, img_gray, kernel_size, threshold, radius): Performs morphological black-hat filtering and inpainting to remove unwanted hair from the image
                - hist_equalization (image): Applies adaptive histogram equalization to enhance image contrast
                - sobel_edge_detection (image): Applies Sobel edge detection for detecting edges
                - canny_edge_detection (image): Applies Canny edge detection to refine edges
        """
        print("[INFO] - img_preprocess_util.py - Starting image pre-processing")
        self.results = {}  # To store the phases of the image during processing
        original_img = record.image_data["original_img"]

        # Step 1: Apply Bilateral Denoising
        print("    [INFO] - img_preprocess_util.py - Applying 'Bilateral denoising'")
        denoised_image = self.bilateral_denoising(original_img)
        record.image_data['denoised_image'] = denoised_image
        
        # Step 2: Calculate SSIM
        ssim_value = self.ssim(original_img, denoised_image)
        print(f"    [INFO] - img_preprocess_util.py - SSIM Value (Before denoising): {ssim_value}")

        # If SSIM is below threshold, check PSNR
        if ssim_value < 0.8:
            # Step 3.1: Calculate PSNR
            psnr_value = self.psnr(original_img, denoised_image)
            print(f"    [INFO] - img_preprocess_util.py - PSNR Value (After denoising): {psnr_value}")
            # Step 3.2: If PSNR is below threshold, apply NLMD
            if psnr_value < 20:
                print("    [INFO] - img_preprocess_util.py - PSNR below threshold, applying 'Non-Local Means Denoising'")
                nlmd_image = self.nlmd(denoised_image)
                record.image_data['nlmd_img'] = nlmd_image
                denoised_image = nlmd_image  # Update denoised image with NLMD applied
        else:
            print("    [INFO] - img_preprocess_util.py - SSIM above threshold, no denoising required.")
            denoised_image = original_img  # No denoising needed, keep original image
        
        # Store the final denoised image (after all processing steps)
        record.image_data['denoised_img'] = denoised_image
        
        # Step 4: Enhance image quality using Contrast Limited Adaptive Histogram Equalization
        print("    [INFO] - img_preprocess_util.py - Enhancing image quality with Contrast Limited Adaptive Histogram Equalization")
        enhanced_image = self.clahe_equalization(original_img)
        record.image_data['enhanced_img'] = enhanced_image

        # Step 5: Hair removal with Black-Hat transformation on the denoised-enhanced image
        # As Step 4) may enhance the contrast between the hair and the skin, potentially making it easier to detect hair
        # On the downside, it can also introduce artifacts or increase contrast in non-hair regions -> makes it harder to separate hair from other image features
        print("    [INFO] - img_preprocess_util.py - Removing hair with Black-Hat transformation")
        blackhat_image, thresh_hair, img_out = self.removeHair(denoised_image, cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY))
        
        # Store the results from hair removal
        record.image_data['blackhat_img'] = blackhat_image
        record.image_data['threshold_hair_mask'] = thresh_hair
        record.image_data['inpainted_img'] = img_out
        
        # Step 6: Apply edge detection to obtain the skin lesion binary mask for feature extractions
        print("    [INFO] - img_preprocess_util.py - Detecting edges using Sobel and Canny Edge Detection")
        sobel_edges, _ = self.sobel_edge_detection(img_out)
        canny_edges = self.canny_edge_detection(sobel_edges)
        
        # Store edge detection results
        record.image_data['sobel_edges'] = sobel_edges
        record.image_data['canny_edges'] = canny_edges

        # Printing status as pre-processing has finished
        print("[INFO] - img_preprocess_util.py - Image pre-processing completed")
    
    # Structural Similarity Index (SSIM)
    #   Measures structural, luminance, and contrast information
    # Threshold: SSIM > 0.8 is typically acceptable for high-quality images
    def ssim(self, original_image, denoised_image):
        # Convert images to grayscale version
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        denoised_gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_value, _ = ssim(original_gray, denoised_gray, full=True)
        
        return ssim_value

    # Peak Signal-to-Noise Ratio (PSNR)
    #   Measures the average error between the original and denoised image
    # Threshold: PSNR > 20 dB is typically considered acceptable
    def psnr(original_image, denoised_image):
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((original_image - denoised_image) ** 2)
        
        # If MSE is 0, that means no noise, PSNR is infinite
        if mse == 0:
            print("    [Warning] - img_preprocess_util.py - Mean squared error between 'original' and 'denoised' image is 0!")
            return float('inf')
        
        # Maximum possible pixel value (for 8-bit images)
        max_pixel = 255.0
        
        # Calculate PSNR using formula
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        
        return psnr

    # "Bilateral denoising" operates based on two main concepts:
    # 1) Spatial Distance (Gaussian Kernel):
    #   This refers to the pixel's physical distance in the image
    #   The closer two pixels are, the more influence they will have on each other (thanks to the Gaussian Kernel)
    # 2) Intensity Difference (Range Kernel):
    #   This measures the difference in pixel intensity (eg. color or grayscale) between the center pixel and its neighbors
    #   If the intensity difference is small, the neighboring pixel is weighted more heavily (meaning there is no edge)
    def bilateral_denoising(self, image, d=5, sigmaColor=25, sigmaSpace=100):
        """
            Apply bilateral denoising to the input image
            
            Parameters:
                - image: The input image (grayscale or color)
                - d: Diameter of the pixel neighborhood
                - sigmaColor: Filter's sensitivity to color differences
                - sigmaSpace: Filter's sensitivity to spatial distance
            
            Returns:
                - denoised_image: The denoised image
        """
        denoised_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)

        return denoised_image

    # "Non-Local Means Denoising" (NLMD):
    # 1) Non-locality: the filter relies on similar patches across the image that can be used to estimate the value of a pixel, even if they are far apart spatially
    # 2) Patch-based Similarity: the filter calculates the similarity of the patch of pixels being processed with patches of other pixels in the image
    # 3) Weighting Function: the estimated value is the weighted average of all pixels in the image, but similar pixel neighborhoods give larger weights
    def nlmd(self, image, h=10, templateWindowSize=7, searchWindowSize=21):
        """
            Apply Non-Local Means Denoising to an image.
            
            Parameters:
                - image: The input image (grayscale or color)
                - h: Filter strength. Higher values mean stronger denoising
                - templateWindowSize: Size of the patch used to calculate similarity
                - searchWindowSize: Size of the window used to search for similar patches
                
            Returns:
                - denoised_image: The denoised image
        """
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, templateWindowSize, searchWindowSize)

        return denoised_image

    # 1) Histogram equalization: uses the global contrast -> for instance values will range between 0 and 255
    #   Downside: Parts of the image will be too bright or too dark -> since pixels are forced/stretched to fill out range
    # 2) Adaptive histogram equalization: uses the same concept, but it divides the image into subparts (or superpixels)
    #   Advantage: This will lead to a better contrast as it performs histogram equalization on each tile independently
    #   Downside: The pixels at the border of each tile might have completely different values -> "nubbly"/not matching overall image 
    # 3) Contrast Limited Adaptive Histogram Equalization (CLAHE): uses 'AHE' but with bilinear interpolation at the borders -> smooth overall view
    #   Problem: we have an RGB color-channeled image -> need to convert it to LAB space
    #       - L: luminosity or intensity information
    #       - a: green-red axis
    #       - b: blue-yellow axis
    #   We need this, so that we can apply the histogram equalization on 'L' (basically seperating the colors), then mix the colors back and get the original image
    def clahe_equalization(self, image, clip_limit: float = 3.0, grid_size: int = 8):
        """
            Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel of an image

            Args:
                - image (ndarray): The input RGB image

            Returns:
                - enhanced_image (ndarray): The image after applying CLAHE (in RGB color space)
        """
        # Convert the image from RGB to LAB color space
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split the LAB image into L, a, b channels
        l, a, b = cv2.split(img_lab)
        
        # Create a CLAHE object with a contrast limit (clipLimit) and tile grid size (tileGridSize)
        clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = (grid_size, grid_size))
        
        # Apply CLAHE on the L channel (luminance channel)
        l_clahe = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel back with the original a and b channels
        img_lab_clahe = cv2.merge((l_clahe, a, b))
        
        # Convert the image back from LAB to RGB
        enhanced_image = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
        
        return enhanced_image
    
    # Removes hair from the image using morphological filtering and inpainting
    def removeHair(self, img_org, img_gray, kernel_size=25, threshold=20, radius=5):
        """
        Removes hair from the image using morphological filtering and inpainting.

        Args:
            img_org (ndarray): The original RGB image.
            img_gray (ndarray): The grayscale version of the original image.
            kernel_size (int): The size of the kernel used for the morphological operation (default is 25)
            threshold (int): The threshold value used for binarizing the hair contours (default is 10)
            radius (int): The radius used for inpainting the image (default is 4)
        
        Returns:
            tuple: A tuple containing:
                - blackhat (ndarray): The result of the blackHat morphological operation.
                - thresh (ndarray): The thresholded mask used for inpainting.
                - img_out (ndarray): The inpainted image with hair removed.
        """
        # the kernel is used to define the size and shape of the area over which the 'cv2.MORPH_BLACKHAT' is applied
        # larger kernel will affect a larger portion of the image, while a smaller kernel focuses on finer details
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        # grayscaling the image using the kernel to detect the contours of the hair
        # the result is an image where dark hair stands out in white on a mostly black background
        blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

        # after the BlackHat operation, the contours of the hair are intensified, but we need to separate them clearly from the background
        # thresholding is used to create a mask that highlights the hair as a binary image (hair = white, background = black)
        _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

        # inpainting is used to fill in the removed hair regions
        # the cv2.inpaint() function fills in the detected hair regions using neighboring pixels to make the inpainting as seamless as possible
        img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

        return blackhat, thresh, img_out
    
    # SOBEL EDGE DETECTION:
    # Applies Sobel filters with kernels for the 'x' and 'y' axes (orientations)
    # - The filters are responsive to steep intensity changes, detecting edges based on contrast differences
    # - The orientation of the gradient affects the sign of the output values
    # - The negative and positive values are stretched to the range of 0-255 for visualization
    # - The result is a mostly gray image with edges marked in black on one side of the gradient
    # - The magnitude of the gradient is calculated using Pythagoras' theorem: G = sqrt(Gx^2 + Gy^2)
    # - This removes the negative sign, allowing both large and small gradients to contribute equally
    # - Edge orientation is calculated as the arc-tangent of the gradient in the y-direction over the x-direction (atan2(Gy, Gx))

    # Note: Since Sobel operates on intensity, it requires the grayscale version of the image
    # Advisable: Apply 'Gaussian Blur' before Sobel edge detection to reduce high-frequency noise
    # - Gaussian blur uses a kernel with a normal distribution to give more weight to nearby pixels, smoothing the image
    def sobel_edge_detection(self, image, kernel_size=3, blur_kernel_size=5):
        """
            Applies Sobel edge detection on the grayscale image

            Args:
                - kernel_size (int): The size of the Sobel kernel
                - blur_kernel_size (int): The size of the Gaussian blur kernel before applying Sobel

            Returns:
                - edge_magnitude (ndarray): The gradient magnitude image
                - edge_orientation (ndarray): The gradient orientation (in degrees)
        """
        # Apply Gaussian Blur to reduce high frequency noise
        blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

        # Sobel operators (for x and y directions)
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=kernel_size)  # Gradient in X direction
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=kernel_size)  # Gradient in Y direction

        # Calculate gradient magnitude (Pythagoras' theorem)
        edge_magnitude = cv2.magnitude(sobel_x, sobel_y)

        # Normalize the magnitude image to fit the range [0, 255]
        edge_magnitude = np.uint8(np.absolute(edge_magnitude))

        # Calculate edge orientation (atan2 returns the angle in radians)
        edge_orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi  # Convert to degrees
        edge_orientation = np.uint8(np.absolute(edge_orientation))  # Convert to positive values

        return edge_magnitude, edge_orientation
    
    # CANNY EDGE DETECTION:
    # Canny uses the output of Sobel as input to detect edges
    # - It creates a binary image where edges are 1px wide, focusing on the position rather than the size of the edges
    # - The edges are refined using hysteresis thresholding:
    #     1. Upper threshold: Any edge gradient above this value is accepted
    #     2. Lower threshold: Any edge gradient below this value is rejected
    #     3. Values between the thresholds are accepted only if connected to edges above the upper threshold
    # - The goal is to identify the true edges by finding local maxima based on the gradient magnitude and orientation

    # The thresholds for hysteresis help determine which edges are significant and which are noise:
    # - A lower threshold keeps more edges, but may also introduce noise
    # - A higher threshold reduces noise but may also miss important edges
    def canny_edge_detection(self, sobel_edges, low_threshold=50, high_threshold=150):
        """
            Applies Canny edge detection based on the Sobel operator's gradient output

            Args:
                - sobel_edges (ndarray): Sobel edge detection output image
                - low_threshold (int): The lower threshold for edge detection
                - high_threshold (int): The higher threshold for edge detection.

            Returns:
                - canny_edges (ndarray): The edges detected by the Canny operator
        """
        # Apply Canny edge detection
        canny_edges = cv2.Canny(sobel_edges, low_threshold, high_threshold)

        return canny_edges