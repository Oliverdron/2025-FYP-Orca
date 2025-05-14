# TODO:
# 1) Validate the original mask (and threshold mask) for emptiness and log warnings if no foreground is found!
# 2) Implement adaptive hair-removal method
# 3) # "- Implement automatic methods for hair feature extraction. You can use annotation agreement to adapt / evaluate your developed method" ???
# 4) Implement two or more different classifier models on the same feature matrix
#       Logistic Regression: Baseline; Fast, well-understood; Linear decision boundary
#       Random Forest: Captures non-linear interactions; Handles outliers, gives feature importances; Can overfit on noisy features
#       Gradient Boosting: State-of-the-art on tabular data; Very high accuracy; Training can be slower
# Evaluate the models

from util.img_util       import Dataset, Record
from util.feature_A      import asymmetry as extract_feature_A
from util.feature_B      import border_irregularity as extract_feature_B
from util.feature_C      import color_heterogeneity as extract_feature_C
from util.classifier     import classifier_model

from util import (
    sys, Path, pd,
    LogisticRegression,
    GradientBoostingClassifier,
    confusion_matrix,
    classification_report
)

FEATURE_MAP = {
    "feat_A": extract_feature_A,
    "feat_B": extract_feature_B,
    "feat_C": extract_feature_C,
    # ALSO IMPORT EXTENDED FEATURES IN EXTENDED.py LATER!!!
}

CLASSIFIERS = {
    "lr": LogisticRegression(max_iter = 1000, random_state = 42, verbose = 0),
    "gb": GradientBoostingClassifier(random_state = 42, verbose = 0)
    # Set classifiers according to discussion with the team
}

def get_base_dir() -> Path:
    """
        Returns the absolute Path to the directory containing this script
    """
    return Path(__file__).parent.resolve()

def main():
    # 1) Retrieve the absolute path to the directory containing this script
    base = get_base_dir()

    # 2) Locate & validate dataset.csv
    csv_path = base / "dataset.csv"
    if not csv_path.exists():
        sys.exit(f"Error: dataset.csv not found in {base}")

    # 3) Set up image folder and result output path
    data_dir = base / "data"
    output_path = base / "result" / "result_baseline.csv"

    # 4) Build Dataset so we can train/evaluate/test our model directly (this also calls Record.load() for each image)
    ds = Dataset(FEATURE_MAP, csv_path, data_dir)

    # 5) Pass the Dataset to the classifier model for training and evaluation
    result = classifier_model(ds, list(FEATURE_MAP.keys()), CLASSIFIERS, test_size=0.3, random_state=42, output_path=output_path)

    # 6) Represent test accuracy, write results to CSV and possibly display predictions on a plot


    # acc = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # print("Test Accuracy:", acc)
    # print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()