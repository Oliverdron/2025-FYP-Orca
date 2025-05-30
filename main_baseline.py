from util import (
    Path, pd, os, np,
    LoadClassifier, TrainClassifier,
    Dataset,
    get_base_dir,
    ALL_CLASSIFIERS,
    ALL_FEATURES,
    ALL_PARAM_DISTR
)

# ── Feature extraction ────────────────────────────────────────────────────
SELECTED_FEATURES = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E", "feat_F"]  # Choose a subset by name
FEATURES = {k: ALL_FEATURES[k] for k in SELECTED_FEATURES}

# ── Classifiers ──────────────────────────────────────────────────────────────
SELECTED_CLASSIFIERS = ["mlp","lr"] # Choose a subset by name
CLASSIFIERS = {k: ALL_CLASSIFIERS[k] for k in SELECTED_CLASSIFIERS}

# ── Hyperparameter distributions ───────────────────────────────────────────────────
PARAM_DISTR = {k: ALL_PARAM_DISTR[k] for k in SELECTED_CLASSIFIERS}

def main():
    # 1) Retrieve the absolute path to the directory containing this script
    base = get_base_dir(Path(__file__))

    # 2) Set up result output path
    output_path = base / "result"

    # 3) Build Dataset so we can train/evaluate/test our model directly (this also calls Record.load() for each image)
    # This should only run if the dataset.csv file does not exist yet, as the dataset.csv file should contain the extracted feature values
    if not os.path.exists(os.path.join(base, "dataset.csv")):
        print("[INFO] - main_demo.py - Dataset not found, creating new one")
        Dataset(feature_extractors=FEATURES, base_dir=base)

    # 4) Initialize and set up the classifier
    # (FLAG THE INCORRECTLY CLASSIFIED SAMPLE ("most incorrect" ones) -> any patterns?)
    # Create the TrainClassifier object for training models from scratch
    clf = TrainClassifier(
        base_dir=base,
        feature_names=list(FEATURES.keys()),
        classifiers=CLASSIFIERS,
        output_path=output_path
    )

    # 5) Load and split data into train, validation, and test sets
    clf.load_split_data()
    
    # 6) Hyperparameter tuning and training (RandomizedSearchCV) on the classifiers
    # Then, save the results and probabilities from hyperparameter tuning and training
    clf.save_result_and_probabilities(
        *clf.training_hyperparameter_tuning(PARAM_DISTR, scoring="roc_auc"),
        type="tuning"
    )
    
    # 7) Visualize the results of hyperparameter tuning and training
    clf.visualize(clf.X_val, clf.y_val, "training")
    clf.visualize_CV_boxplots("roc_auc")
    
    # 8) Optimize classification thresholds (TunedThresholdClassifierCV)
    # Then, save the results and probabilities from threshold optimization
    clf.save_result_and_probabilities(
        *clf.optimize_thresholds(scoring="roc_auc"),
        type="threshold"
    )
    
    # 9) Evaluate classifiers (Evaluates on the test set)
    clf.save_result_and_probabilities(
        *clf.evaluate_classifiers(clf.X_test, clf.y_test),
        type="evaluation"
    )
    
    # 10) Visualize the results of evaluation
    clf.visualize(clf.X_test, clf.y_test, "final")

    
    # 11) Save the trained models to disk
    clf.save_models()


if __name__ == "__main__":
    main()