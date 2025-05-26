# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path
import os
import json
import time
# To avoid multithreading issues with OpenCV and NumPy, set the number of threads to 1
os.environ["OMP_NUM_THREADS"] = "1"
import random
from math import sqrt, floor, ceil, nan, pi, log10, sqrt
from statistics import variance, stdev
from datetime import datetime

# ── Numerical & image processing ────────────────────────────────────────────
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import circmean, circvar, circstd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

# ── Scikit-image ───────────────────────────────────────────────────────────────
from skimage import color, exposure, morphology
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import blob_log
from skimage.filters import threshold_otsu, frangi
from skimage.measure import label, regionprops
from skimage.transform import resize, rotate
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim

# ── Scikit-learn & Pandas ─────────────────────────────────────────────────────
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report