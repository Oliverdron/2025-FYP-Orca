from util import (
    Path, pd, os, np,
    Classifier,
    Dataset,
    get_base_dir,
    ALL_CLASSIFIERS,
    ALL_FEATURES,
    ALL_PARAM_GRIDS,
)

# ── Feature extraction ────────────────────────────────────────────────────
SELECTED_FEATURES = ["feat_B"]  # Choose a subset by name
FEATURES = {k: ALL_FEATURES[k] for k in SELECTED_FEATURES}


# ── Classifiers ──────────────────────────────────────────────────────────────
SELECTED_CLASSIFIERS = ["mlp","lr"] # Choose a subset by name
CLASSIFIERS = {k: ALL_CLASSIFIERS[k] for k in SELECTED_CLASSIFIERS}



# ── Hyperparameter grids ───────────────────────────────────────────────────
PARAM_GRIDS = {k: ALL_PARAM_GRIDS[k] for k in SELECTED_CLASSIFIERS}



    LogisticRegression,
    GradientBoostingClassifier,
    RandomForestClassifier,
    KNeighborsClassifier,
    SVC,
    MLPClassifier,
    RFE,
    plt,
    Pipeline,
    StandardScaler
)

FEATURE_MAP = {
    "feat_A": extract_feature_A,
    "feat_B": extract_feature_B,
    "feat_C": extract_feature_C,
    #"feat_D": hair_extraction,
    #"feat_D": extract_feature_D, 
    # ALSO IMPORT EXTENDED FEATURES IN EXTENDED.py LATER!!!
}

# Set classifiers according to discussion with the team. Using Pipelines for clear and consistent preprocessing and model training. 
# Data leakage is avoided by using StandardScaler within the Pipeline, ensuring that scaling is applied only to the training data during cross-validation.
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
   
}

# Choose a subset by name
SELECTED = ["mlp","lr"]
CLASSIFIERS = {k: ALL_CLASSIFIERS[k] for k in SELECTED}

PARAM_GRIDS = {
    "mlp": 
    {
    'clf__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'clf__activation': ['relu', 'tanh'],
    'clf__alpha': [0.0001, 0.001, 0.01],
    'clf__learning_rate_init': [0.001, 0.0001],
    'clf__batch_size': [32, 64]
    },
    "lr" :
    {
    'clf__C': np.logspace(-3, 3, 7),         
    'clf__l1_ratio': [0, 0.15, 0.5, 0.85, 1]    
    }
}

def get_base_dir() -> Path:
    """
        Returns the absolute Path to the directory containing this script
    """
    return Path(__file__).parent.resolve()

def main():
    # 1) Retrieve the absolute path to the directory containing this script
    base = get_base_dir(Path(__file__))

    # 2) Set up result output path
    output_path = base / "result"

    # 4) Build Dataset so we can train/evaluate/test our model directly (this also calls Record.load() for each image)
    # This should only run if the dataset.csv file does not exist yet, as the dataset.csv file should contain the extracted feature values
    
    # ------- FOR TESTING -------
    #if not os.path.exists(os.path.join(base, "dataset.csv")):
    #    print("[INFO] - main_demo.py - Dataset not found, creating new one")
    ds = Dataset(feature_extractors=FEATURES, base_dir=base, shuffle=True, limit=10)
    ds = Dataset(feature_extractors=FEATURE_MAP, base_dir=base, shuffle=False, limit=5)
    # Later on 'record.image_data["threshold_segm_mask"]' should never be None
    # But now is, which will lead to a 'NoneType' attribute access error
    # ---------------------------

    # 5) Pass the Dataset to the classifier model for training and evaluation
    # SAVE THE BEST CLASSIFIER BASED ON AVERAGE VALIDATION PERFORMANCE AT MULTIPLE TRAINING SET SIZES (learning curves)
    # EVALUATE ACCURACY: true label VS probabilities
    # REPORT MEAN AND STD
    # FLAG THE INCORRECTLY CLASSIFIED SAMPLE ("most incorrect" ones) -> any patterns?

    # Aim for similar performance on training and validation sets
    # For classifiers relying on distances, scale the features using StandardScaler from scikit-learn
    # Record class now have an ID attr., indicating the patient's ID as there could be more image for one
 
    clf = Classifier(
        base,
        list(FEATURES.keys()),
        CLASSIFIERS,
        test_size=0.2,
        random_state=42,
        output_path=output_path
    )
    clf.load_split_data()
    clf.hyperparameter_tuning(PARAM_GRIDS, scoring="roc_auc")  
    clf.optimize_thresholds(scoring="roc_auc")  

    
    print(clf.trained_models)
    
    # Only run this when classifiers are trained and saved to the maximum extent
    clf.evaluate_classifiers()  
    print(clf.trained_models["mlp"].best_threshold_)

    # IF CLASSIFIER_CONFIG.JSON EXISTS, IMPORT AND HAVE METHOD TO CLASSIFY OTHER DATASET!
    # VISUALIZE THE TRAINED CLASSIFIER (see: https://stackoverflow.com/questions/41138706/recreating-decision-boundary-plot-in-python-with-scikit-learn-and-matplotlib)
    
    

    
    # In classifier.py -> Save the best model's results/parameters

"""    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(hierarchicalClassifier.X, hierarchicalClassifier.Y_binary)
    importances = pd.Series(model.feature_importances_, index=hierarchicalClassifier.X.columns)
    importances = importances.sort_values(ascending=False)

    # Plot feature importances
    importances[:2].plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(f"Top {2} Features by Random Forest Importance")
    plt.show()"""

if __name__ == "__main__":
    main()