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
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel

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
        #('feature_selector', SelectFromModel(LogisticRegression(penalty='l2'))),
        ("clf", RandomForestClassifier(max_depth=10,min_samples_leaf=5, class_weight='balanced', max_samples=0.7))
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
        ("clf", SVC(kernel='linear', random_state=42, verbose=0, probability=True))
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
    'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (30, 30, 30)],
    'clf__activation': ['relu', 'tanh'],  # Avoid 'logistic' for deep nets
    'clf__alpha': np.logspace(-5, -1, 20),  # L2 regularization
    'clf__learning_rate_init': [0.001, 0.005, 0.01],
    'clf__batch_size': [32, 64, 128],  # Smaller batches for medical data
    'clf__early_stopping': [True],  # Critical for medical applications
    'clf__validation_fraction': [0.1, 0.2],
    'clf__beta_1': [0.9, 0.95],  # Adam optimizer params
    'clf__beta_2': [0.999, 0.9999],
    'clf__solver': ['adam'],  # Better than 'sgd' for medical imaging
    'clf__max_iter': [500, 1000]  # Allow sufficient convergence
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

        # no penalty (only supported with saga solveR )
        {
            'clf__penalty': [None],
            'clf__solver': ['saga'],
            'clf__max_iter': [100, 200, 500]
        }
    ]
    ,
    "rf" :
    {
    'clf__max_depth': [10, 15, 20],  
    'clf__min_samples_split': [10, 20],  
    'clf__min_samples_leaf': [5, 10],  
    'clf__max_features': ['sqrt'],  # Reduced complexity
    'clf__n_estimators': [100],  # Fixed for stability
    'clf__max_samples': [0.6, 0.7],  # Subsampling
    'clf__ccp_alpha': [0.0, 0.01, 0.1]  
    },
    "xgb" :
    {
    'clf__n_estimators': [100, 200, 300],  # Fewer trees than default (medical data is often smaller)
    'clf__max_depth': [3, 5, 7, 10],  # Shallower trees prevent overfitting to image artifacts
    'clf__learning_rate': [0.01, 0.05, 0.1],  # Lower rates for stable convergence
    'clf__subsample': [0.7, 0.8, 0.9],  # Stochastic sampling
    'clf__colsample_bytree': [0.7, 0.8, 1.0],
    'clf__gamma': [0, 0.1, 0.2],  # Minimum loss reduction (pruning)
    'clf__reg_alpha': [0, 0.1, 1],  # L1 regularization
    'clf__reg_lambda': [0, 0.1, 1],  # L2 regularization
    'clf__scale_pos_weight': [1, 1.5, 2]  # Adjust for slight class imbalance
    },
    "knn" :
        {
    'clf__n_neighbors': [3, 5, 7, 10],  # Fewer neighbors for rare cancer cases
    'clf__weights': ['uniform', 'distance'],  # Distance weighting helps with outliers
    'clf__metric': ['euclidean', 'manhattan', 'minkowski'],
    'clf__p': [1, 2],  # Power for Minkowski metric (1=manhattan, 2=euclidean)
    'clf__algorithm': ['auto', 'ball_tree'],  # Ball_tree better for high-dim medical features
    'clf__leaf_size': [15, 30, 45]
},
    "svc" : {
    'clf__C': [0.01, 0.1, 1, 10, 30],  # 10 values (0.001 to 100)
    'clf__kernel': ['rbf', 'linear'],  # Poly often too slow for medical images
    'clf__gamma': ['scale', 'auto'] + list(np.logspace(-1, 1, 3)),  # 7 options
    'clf__class_weight': [None, 'balanced', {0:1, 1:2.5}],
    'clf__shrinking': [True, False]
},
    "gb": {
    # Core parameters
    'clf__n_estimators': [100, 200, 300],  # Fewer trees than default (medical data is often smaller)
    'clf__learning_rate': [0.01, 0.05, 0.1],  # Lower rates for stable convergence
    'clf__max_depth': [3, 4, 5],  # Shallower trees prevent overfitting to image artifacts
    'clf__min_samples_split': [10, 20],  # Higher than default for medical robustness
    'clf__min_samples_leaf': [5, 10],  # Conservative leaf sizes
    
    # Medical-specific tuning
    'clf__max_features': ['sqrt', None],  # Fewer features for high-dim medical data
    'clf__subsample': [0.8, 0.9],  # Stochastic gradient boosting
    'clf__ccp_alpha': [0.0, 0.001, 0.01],  # Post-pruning for simpler trees
    'clf__loss': ['log_loss', 'exponential'],  # Alternative losses
    
    # Class imbalance handling
    'clf__init': [None, 'zero'],  # For severe imbalance
    'clf__validation_fraction': [0.1, 0.2]  # Early stopping
}
    
}

# Functions
def get_base_dir(path: Path) -> Path:
    """
        Returns the absolute Path to the directory containing this script
    """
    return path.parent.resolve()