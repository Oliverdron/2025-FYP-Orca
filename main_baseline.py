""" Main script for running baseline experiments with selected basic classifiers and  A, B, C features.
    """



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
SELECTED_FEATURES = ["feat_A", "feat_B", "feat_C"]  # Choose a subset by name
FEATURES = {k: ALL_FEATURES[k] for k in SELECTED_FEATURES}


# ── Classifiers ──────────────────────────────────────────────────────────────
SELECTED_CLASSIFIERS = ["mlp","lr"] # Choose a subset by name
CLASSIFIERS = {k: ALL_CLASSIFIERS[k] for k in SELECTED_CLASSIFIERS}



# ── Hyperparameter grids ───────────────────────────────────────────────────
PARAM_GRIDS = {k: ALL_PARAM_GRIDS[k] for k in SELECTED_CLASSIFIERS}



def main():
    # 1) Retrieve the absolute path to the directory containing this script
    base = get_base_dir(Path(__file__))
    
    # 2) Set up result output path
    output_path = base / "result"
    
    # 3) Build Dataset so we can train/evaluate/test our model directly (this also calls Record.load() for each image)
    # This should only run if the dataset.csv file does not exist yet, as the dataset.csv file should contain the extracted feature values
    if not os.path.exists(os.path.join(base, "dataset.csv")):
        print("[INFO] - main_demo.py - Dataset not found, creating new one")
        ds = Dataset(feature_extractors=FEATURES, base_dir=base)

    # 4) Pass the Dataset to the classifier model for training and evaluation
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
    
    
if __name__ == "__main__":
    main()