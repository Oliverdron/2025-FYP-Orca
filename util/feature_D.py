import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull
from skimage.feature import graycomatrix, graycoprops

def get_glcm_feature(img_path, mask_path, distances=[1], angles=[0], feature='contrast'):
    
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img_gray.shape != mask.shape:
    # Resize mask to match image dimensions if mismatch occurs
        mask = cv2.resize(mask, (img_gray.shape[1], img_gray.shape[0]), 
                       interpolation=cv2.INTER_NEAREST)
        
    _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    masked_img = np.zeros_like(img_gray)
    masked_img[mask_binary > 0] = img_gray[mask_binary > 0]
    masked_img = (masked_img * 255).astype(np.uint8)
    
    glcm = graycomatrix(masked_img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return graycoprops(glcm, feature)[0, 0]  # Returns a single value