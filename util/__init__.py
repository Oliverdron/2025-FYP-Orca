# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path
import os
import random
from math import sqrt, floor, ceil, nan, pi
from statistics import variance, stdev
import json

# ── Numerical & image processing ────────────────────────────────────────────
import cv2
import numpy as np
from scipy.stats import circmean, circvar, circstd
from scipy.spatial import ConvexHull, pdist

# ── Scikit-image ───────────────────────────────────────────────────────────────
from skimage import color, exposure, morphology
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize, rotate
from skimage.segmentation import slic

# ── Scikit-learn & Pandas ─────────────────────────────────────────────────────
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Submodule exports ─────────────────────────────────────────────────────────
from .img_util       import Dataset, Record
from .inpaint_util   import removeHair
from .feature_A      import asymmetry as extract_feature_A
from .feature_B      import border_irregularity as extract_feature_B
from .feature_C      import color_heterogeneity as extract_feature_C
from .classifier     import classifier_model