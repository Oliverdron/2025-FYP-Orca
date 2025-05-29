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
import joblib

# ── Numerical & image processing ────────────────────────────────────────────
import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, TunedThresholdClassifierCV, LearningCurveDisplay, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_validate, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier, XGBRegressor

# Classes
from util.img_util import Dataset, Record
from util.classifier import LoadClassifier, TrainClassifier

ALL_CLASSIFIERS = {
    "lr": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ]),
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_jobs=-1, random_state=42))
    ]), 
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(early_stopping=True, random_state=42))
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_jobs=-1))
    ]),
    "svc": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel='linear', random_state=42, verbose=0))
    ]),
    "gb": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42, verbose=0))
    ]), 
    "xgb": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(eval_metric='logloss', random_state=42))
    ]),
   
}

# Feature extraction
from util.feature_A import asymmetry            as extract_feature_A
from util.feature_B import border_irregularity  as extract_feature_B
from util.feature_C import color_heterogeneity  as extract_feature_C
from util.feature_D import hair_feat_extraction as extract_feature_D
from util.feature_E import vascular_score as extract_feature_E
from util.feature_F import measure_streaks as extract_feature_F

# Feature map
ALL_FEATURES = {
    "feat_A": extract_feature_A,
    "feat_B": extract_feature_B,
    "feat_C": extract_feature_C,
    "feat_D": extract_feature_D,
    "feat_E": extract_feature_E,
    "feat_F": extract_feature_F,
     
    # ALSO IMPORT EXTENDED FEATURES IN EXTENDED.py LATER!!!
}


# Hyperparameter grids for classifiers
ALL_PARAM_DISTR = {
    "mlp": 
    {
    'clf__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)], 
    'clf__activation': ['relu', 'tanh'], 
    'clf__alpha': [0.0001, 0.001, 0.01],
    'clf__learning_rate_init': [0.001, 0.0001],
    'clf__batch_size': [32, 64] 
    },
       "lr": [
        # l1 penalty
        {
            'clf__penalty': ['l1'],
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__solver': ['liblinear', 'saga'],
            'clf__max_iter': [100, 200, 500]
        },

        # l2 penalty
        {
            'clf__penalty': ['l2'],
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__solver': ['liblinear', 'saga'],
            'clf__max_iter': [100, 200, 500]
        },

        # elasticnet penalty
        {
            'clf__penalty': ['elasticnet'],
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__solver': ['saga'],
            'clf__l1_ratio': [0.1, 0.5, 0.9],
            'clf__max_iter': [100, 200, 500]
        },

        # no penalty (only supported with saga and scikit-learn ≥0.22)
        {
            'clf__penalty': [None],
            'clf__solver': ['saga'],
            'clf__max_iter': [100, 200, 500]
        }
    ]
    ,
    "rf" :
    {
    'clf__n_estimators': [100, 200, 500],                  # Number of trees
    'clf__max_depth': [None, 10, 20, 30],                  # Max depth of trees
    'clf__min_samples_split': [2, 5, 10],                  # Min samples to split an internal node
    'clf__min_samples_leaf': [1, 2, 4],                    # Min samples at a leaf node
    'clf__max_features': ['sqrt', 'log2', None],           # Number of features to consider at split
    'clf__bootstrap': [True, False],                       # Whether bootstrap samples are used
    'clf__criterion': ['gini', 'entropy']                  # Splitting criteria
    },
    "xgb" :
        {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.6, 1.0],
    'clf__gamma': [0, 0.1, 0.5],
    'clf__reg_alpha': [0, 0.01, 0.1],    # L1 regularization
    'clf__reg_lambda': [1, 2.0],    # L2 regularization
    'clf__scale_pos_weight': [1, 2]   # Useful if dataset is imbalanced
        },
    "knn" :
        {
    'clf__n_neighbors': [3, 5, 7, 11],
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metrics
    'clf__leaf_size': [30, 40, 50],  # Leaf size for KDTree or BallTree
    'clf__p': [1, 2]  # p=1 (manhattan), p=2 (euclidean); used with minkowski
    }
    
}

# Functions
def get_base_dir(path: Path) -> Path:
    """
        Returns the absolute Path to the directory containing this script
    """
    return path.parent.resolve()