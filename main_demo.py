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
from util.classifier     import HierarchicalClassifier


# For some reason i cant seem to import from util module, can someone check this?
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from matplotlib import pyplot as plt

from util import (
    Path, pd, os,
    LogisticRegression,
    GradientBoostingClassifier,
)

FEATURE_MAP = {
    #"feat_A": extract_feature_A,
    "feat_B": extract_feature_B,
    "feat_C": extract_feature_C,
    #"feat_D": any
    # ALSO IMPORT EXTENDED FEATURES IN EXTENDED.py LATER!!!
}

# Set classifiers according to discussion with the team
CLASSIFIERS_LEVEL1 = {
    #"lr": LogisticRegression(max_iter = 1000, random_state = 42, verbose = 0),
    #"gb": GradientBoostingClassifier(random_state = 42, verbose = 0),
    #"rf": RandomForestClassifier(random_state = 42, verbose = 0),
    #"knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    #"svc": SVC(kernel='linear', random_state=42, verbose=0),
    "mlp": MLPClassifier(random_state=42, verbose=0, max_iter=1000, hidden_layer_sizes=(100,50)),

}
CLASSIFIERS_LEVEL2_CANCER = {
    #"lr": LogisticRegression(max_iter = 1000, random_state = 42, verbose = 0),
    #"gb": GradientBoostingClassifier(random_state = 42, verbose = 0),
    #"rf": RandomForestClassifier(random_state = 42, verbose = 0),
    #"knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    #"svc": SVC(kernel='linear', random_state=42, verbose=0),
    "mlp": MLPClassifier(random_state=42, verbose=0, max_iter=1000, hidden_layer_sizes=(100,50)),

}
CLASSIFIERS_LEVEL2_NON_CANCER = {
    #"lr": LogisticRegression(max_iter = 1000, random_state = 42, verbose = 0),
    #"gb": GradientBoostingClassifier(random_state = 42, verbose = 0),
    #"rf": RandomForestClassifier(random_state = 42, verbose = 0),
    #"knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    #"svc": SVC(kernel='linear', random_state=42, verbose=0),
     "mlp": MLPClassifier(random_state=42, verbose=0, max_iter=1000, hidden_layer_sizes=(100,50)),

}



def get_base_dir() -> Path:
    """
        Returns the absolute Path to the directory containing this script
    """
    return Path(__file__).parent.resolve()

def main():
    # 1) Retrieve the absolute path to the directory containing this script
    base = get_base_dir()

    # 2) Set up result output path
    output_path = base / "result" 

    # 4) Build Dataset so we can train/evaluate/test our model directly (this also calls Record.load() for each image)
    # This should only run if the dataset.csv file does not exist yet, as the dataset.csv file should contain the extracted feature values
    if not os.path.exists(os.path.join(base, "dataset.csv")):
        ds = Dataset(FEATURE_MAP, base)

    # 5) Pass the Dataset to the classifier model for training and evaluation
    print("Training classifier...")
    hierarchicalClassifier = HierarchicalClassifier(base, 
                                                    list(FEATURE_MAP.keys()),
                                                    CLASSIFIERS_LEVEL1,
                                                    CLASSIFIERS_LEVEL2_CANCER,
                                                    CLASSIFIERS_LEVEL2_NON_CANCER,
                                                    test_size=0.3,
                                                    random_state=42, 
                                                    output_path=output_path)

    hierarchicalClassifier.run()
    hierarchicalClassifier.save_config()
    
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength for Logistic Regression
    'solver': ['liblinear', 'saga'],  # Solvers to use in Logistic Regression
    'max_iter': [100, 200, 300]  # Max iterations for optimization
    }
    #hierarchicalClassifier.tune_hyperparameters("level1", "lr", param_grid)   
    
        # Save the best model
     




    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(hierarchicalClassifier.X, hierarchicalClassifier.Y_binary)
    importances = pd.Series(model.feature_importances_, index=hierarchicalClassifier.X.columns)
    importances = importances.sort_values(ascending=False)

    # Plot feature importances
    importances[:2].plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(f"Top {2} Features by Random Forest Importance")
    plt.show()


if __name__ == "__main__":
    main()