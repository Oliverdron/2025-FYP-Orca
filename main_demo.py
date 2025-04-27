"""
main.py

    1.) Finds "dataset.csv" next to this script
    2.) Loads metadata and images using "util/img_util.py"
    3.) Applies hair removal + feature A/B/C extraction
    4.) Trains & evaluates a logistic regression classifier
    5.) Writes predictions to result/result_baseline.csv

"""

# --- Standard library imports ---
from util import (
    sys, Path, pd,
    Dataset,
    extract_feature_A,
    extract_feature_B,
    extract_feature_C,
    # ALSO IMPORT EXTENDED FEATURES IN EXTENDED.py LATER!!!
    #classifier_model
)

FEATURE_MAP = {
    "feat_A": extract_feature_A,
    "feat_B": extract_feature_B,
    "feat_C": extract_feature_C,
    # ALSO IMPORT EXTENDED FEATURES IN EXTENDED.py LATER!!!
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
    ds = Dataset(FEATURE_MAP, csv_path, data_dir, filename_col="image_path", label_col="label")

    # 5) Pass the Dataset to the classifier model for training and evaluation


    # NEED TO WORK ON CLASSIFIER.PY -> BUILD MODEL
    # THE RECORD CLASS COULD STORE PREDICTED VALUES IF NEEDED FOR A BETTER RESULT ANALYSIS




    # select only the baseline features.
    # baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
    # x_all = data_df[baseline_feats]
    # y_all = data_df["label"]

    # split the dataset into training and testing sets.
    # x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)

    # train the classifier (using logistic regression as an example)
    # clf = LogisticRegression(max_iter=1000, verbose=1)
    # clf.fit(x_train, y_train)

    # test the trained classifier
    # y_pred = clf.predict(x_test)
    # acc = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # print("Test Accuracy:", acc)
    # print("Confusion Matrix:\n", cm)

    # write test results to CSV.
    # result_df = data_df.loc[x_test.index, ["filename"]].copy()
    # result_df['true_label'] = y_test.values
    # result_df['predicted_label'] = y_pred
    # result_df.to_csv(save_path, index=False)
    # print("Results saved to:", save_path)

if __name__ == "__main__":
    main()