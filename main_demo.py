from util.img_util       import Dataset, Record
from util.feature_A      import asymmetry           as extract_feature_A
from util.feature_B      import border_irregularity as extract_feature_B
from util.feature_C      import color_heterogeneity as extract_feature_C
from util.classifier     import HierarchicalClassifier

from util import (
    Path, pd, os,
    LogisticRegression,
    GradientBoostingClassifier,
    RandomForestClassifier,
    KNeighborsClassifier,
    SVC,
    MLPClassifier,
    RFE,
    plt
)

FEATURE_MAP = {
    "feat_A": extract_feature_A,
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
        print("[INFO] - main_demo.py - LINE 86 - Dataset not found, creating new one")
        ds = Dataset(FEATURE_MAP, base)

    # 5) Pass the Dataset to the classifier model for training and evaluation
    print("[INFO] - main_demo.py - LINE 90 - Called hierarchical classifier")
    hierarchicalClassifier = HierarchicalClassifier(base, 
                                                    list(FEATURE_MAP.keys()),
                                                    CLASSIFIERS_LEVEL1,
                                                    CLASSIFIERS_LEVEL2_CANCER,
                                                    CLASSIFIERS_LEVEL2_NON_CANCER,
                                                    test_size=0.3,
                                                    random_state=42, 
                                                    output_path=output_path)
    
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength for Logistic Regression
    'solver': ['liblinear', 'saga'],  # Solvers to use in Logistic Regression
    'max_iter': [100, 200, 300]  # Max iterations for optimization
    }
    #hierarchicalClassifier.tune_hyperparameters("level1", "lr", param_grid)   
    
        # In classifier.py -> Save the best model's results/parameters

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