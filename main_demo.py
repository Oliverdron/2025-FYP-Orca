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
SELECTED_FEATURES = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E", "feat_F"]  # Choose a subset by name
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

    # 4) Build Dataset so we can train/evaluate/test our model directly (this also calls Record.load() for each image)
    # This should only run if the dataset.csv file does not exist yet, as the dataset.csv file should contain the extracted feature values
    
    # ------- FOR TESTING -------
    #if not os.path.exists(os.path.join(base, "dataset.csv")):
    #    print("[INFO] - main_demo.py - Dataset not found, creating new one")
    ds = Dataset(feature_extractors=FEATURES, base_dir=base)
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
############################### TRYING OUT FEATURE ANALYSIS ###############################
from util.featureanalysis import plot_feature_distributions, plot_feature_correlations
import pandas as pd

# Load dataset.csv created during Dataset export
df = pd.read_csv("dataset.csv")

# Call your plotting functions
plot_feature_distributions(df)
plot_feature_correlations(df)

print("[INFO] - Feature plots generated in /results/")
