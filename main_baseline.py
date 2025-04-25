# pull in the main functions from each of the features and iterate through them 
# to create a baseline model
# Feature A : mean_asymmetry()

# Pseudocode: Read File, Check with Masks, incorporate features - 

import os
import cv2
import numpy as np
import pandas as pd
from util.feature_A import mean_asymmetry
from util.feature_A import rotation_asymmetry
from util.feature_A import asymmetry
from util.feature_A import cut_mask
from util.feature_A import find_midpoint_v1

from util.feature_D import get_glcm_feature
from pathlib import Path

# Your previously defined functions: mean_asymmetry, rotation_asymmetry, asymmetry, cut_mask, find_midpoint_v1
# Make sure they are included above this block

def feat_d_extraction(data_path, dataset_path):
    df = pd.read_csv(dataset_path)
    feature_d = []
    i = 1
    for img_rel_path, mask_rel_path in zip(df["image_path"], df["image_mask_path"]):
        if not pd.isna(mask_rel_path):
            img_path = data_path / "images" / img_rel_path
            mask_path = data_path / "lesion_masks" / mask_rel_path
            feature_d.append(get_glcm_feature(img_path, mask_path))
        else:
            feature_d.append("NA")
        # progress
        print(f"{i}/{len(df["image_path"])}")
        i += 1
    df["feature_D"] = feature_d
    df.to_csv("dataset.csv")
        

def load_binary_mask(filepath):
    """Loads an image as a binary mask."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(bool)

def process_masks_in_folder(folder_path, rotations=30):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            path = os.path.join(folder_path, filename)
            mask = load_binary_mask(path)
            
            try:
                score = round(mean_asymmetry(mask, rotations=rotations), 4)
                results[filename] = score
                print(f"{filename}: {score}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                results[filename] = None

    return results

project_root = Path(__file__).parent
masks_path = project_root / "data" / "lesion_masks"
data_path = project_root / "data"
asymmetry_results = process_masks_in_folder(masks_path)
print(asymmetry_results)


feat_d_extraction(data_path, "dataset.csv")