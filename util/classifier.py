from util import (
    datetime,
    json,
    np,
    os,
    pd,
    BaseEstimator,
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
    StandardScaler,
    accuracy_score,
    classification_report
)

class HierarchicalClassifier:
    """ 
        HierarchicalClassifier is a class that implements a two-level hierarchical classification system.
            - The first level classifies samples into binary labels (cancer or non-cancer)
            - The second level classifies samples into specific cancer types or non-cancer types based on the first level's predictions
        
        Args:
            base_dir (str): Base directory's absolute path
            feature_names (list): List containing feature extractor names (Eg. "feat_A", "feat_B"...)
            classifiers_level1 (dict): Dictionary mapping classifier names and models for level 1 classification 
            classifiers_level2_cancer (dict): Dictionary mapping classifier names and models for level 2-cancer classification 
            classifiers_level2_non_cancer (dict): Dictionary mapping classifier names and models for level 2-non-cancer classification
            test_size (float): Fraction of data to hold out for testing
            random_state (int): Random seed for reproducibility
            output_path (str): If provided, save the results CSV here

        Attributes:
            base_dir (str): Base directory's absolute path
            feature_names (list): List containing feature extractor names (Eg. "feat_A", "feat_B"...)
            classifiers_level1 (dict): Dictionary mapping classifier names and models for level 1 classification 
            classifiers_level2_cancer (dict): Dictionary mapping classifier names and models for level 2-cancer classification 
            classifiers_level2_non_cancer (dict): Dictionary mapping classifier names and models for level 2-non-cancer classification
            test_size (float): Fraction of data to hold out for testing
            random_state (int): Random seed for reproducibility
            output_path (str): If provided, save the results CSV here
            trained_models (dict): Dictionary to store trained models for each level
            results (dict): Dictionary to store results for each level
            df (pd.DataFrame): DataFrame containing the dataset
            X (pd.DataFrame): DataFrame containing the features
            Y_binary (pd.Series): Series containing the binary labels
            Y_categorical (pd.Series): Series containing the categorical labels
            X_test (pd.DataFrame): DataFrame containing the test features
            X_train (pd.DataFrame): DataFrame containing the training features
            Y_test_binary (pd.Series): Series containing the test binary labels
            Y_train_binary (pd.Series): Series containing the training binary labels
            Y_test_categorical (pd.Series): Series containing the test categorical labels
            Y_train_categorical (pd.Series): Series containing the training categorical labels

        Methods:
            Methods are not called outside of the class, so making them private
            _load_dataset(): Loads the dataset from the specified base directory.
            _split_data(): Splits the dataset into training and testing sets using the specified parameters.
            _evaluate_model(model, X_test, Y_test, model_name): Evaluates the model on the test set and returns a DataFrame with predictions and accuracy.
            _train_and_evaluate_classifiers(classifiers, X_train, Y_train, X_test, Y_test, group_name): Trains classifiers and evaluates them.
            _level1(): Trains and evaluates the Level 1 classifier on the binary labels.
            _level2(group): Trains and evaluates the Level 2 classifier on the non-cancer or cancer labels.
            _run(): Runs the training and evaluation process on both levels.
            _save_results(): Saves the results to a CSV file if an output path is provided.
            save_config() : Saves the configuration to a JSON file if an output path is provided.
            tune_hyperparameters(level, model_name, param_grid): Hyperparameter tuning for a specific model using GridSearchCV.
            
    """
    def __init__(
        self,
        base_dir: str,
        feature_names: list,
        classifiers_level1: dict[str, BaseEstimator],
        classifiers_level2_cancer: dict[str, BaseEstimator],
        classifiers_level2_non_cancer: dict[str, BaseEstimator],
        test_size: float = 0.3,
        random_state: int = 42,
        output_path: str = None
    ) -> None:
        self.base_dir = base_dir
        self.feature_names = feature_names
        self.classifiers_level1 = classifiers_level1
        self.classifiers_level2_cancer = classifiers_level2_cancer
        self.classifiers_level2_non_cancer = classifiers_level2_non_cancer
        self.test_size = test_size
        self.random_state = random_state
        self.output_path = output_path
        self.trained_models = {}
        self.reports = {}
        self.results = {}
        
        # Load the dataset
        self._load_dataset()
        
        # Split the dataset into training and testing sets
        self._split_data()

        # Run the training and evaluation process on both levels
        self.results = self._run()

        # Save the results to a CSV file if an output path is provided
        self._save_results()

        # Save the classifier configuration, parameters and dataset stats to a JSON file
        self.save_config()

    def _load_dataset(self) -> None:
        """
            Loads the dataset from the specified base directory and extracts features.
            The features are stored in self.X, and the labels in self.Y_binary and self.Y_categorical.
        """
        # Load the dataset, which stores pre-calculated features and labels
        self.df = pd.read_csv(os.path.join(self.base_dir, "dataset.csv"))
        # Extract features and labels
        self.X = self.df[self.feature_names]
        self.Y_binary = self.df['label_binary']
        self.Y_categorical = self.df['label_categorical']
        
    def _split_data(self) -> None:
        """
            Splits the dataset into training and testing sets using the specified parameters.
            The training and testing sets are stored in self.X_train, self.X_test, self.Y_train_binary, self.Y_test_binary, self.Y_train_categorical, and self.Y_test_categorical.
            The features are standardized using StandardScaler.
        """
        # Split the dataset into training and testing sets using the provided parameters
        self.X_train, self.X_test, self.Y_train_binary, self.Y_test_binary, self.Y_train_categorical, self.Y_test_categorical = train_test_split(
            self.X, self.Y_binary, self.Y_categorical,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify = list(zip(self.Y_binary, self.Y_categorical)) # Stratify by both binary and categorical labels to ensure that both distributions are representative
        )
        # Standardize the features using StandardScaler
        scaler = StandardScaler()
        # .fit_transform() method computes the mean and standard deviation of the training data and scales it
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        # the .transform() method is applied to the testing data using the mean and standard deviation computed from the training data earlier
        # This ensures that the testing data is scaled in the same way as the training data
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index)
        
    def _evaluate_model(self, model, X_test, Y_test, model_name: str, group_name: str) -> pd.DataFrame:
        """
            Evaluates the model on the test set and returns a DataFrame with predictions and accuracy.
            The DataFrame contains the filename, truth label, predicted label, and accuracy.
        """
        # Create a DataFrame to hold the results, starting with aligning the indices
        results = pd.DataFrame(index=X_test.index)
        # Add the filename for easy identification
        results["filename"] = self.df.loc[X_test.index, "filename"]
        # Add the truth labels and the predicted labels from both models
        results["truth_label"] = Y_test.values
        preds = model.predict(X_test)
        # Make report with classification_report and save it
        self.reports[f"{group_name}_{model_name}"] = classification_report(Y_test, preds, output_dict=True, zero_division=0)
        results[f"{model_name}_pred"] = preds
        # Calculate overall accuracy and add it to the results rounded to 3 deciamls
        results[f"{model_name}_accuracy"] = np.round(accuracy_score(Y_test, preds), 3)
        # Calculate and add label probabilities to the results
        # But first, check if the classifier has the predict_proba method
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            proba = np.round(proba, 3)  # Round probabilities to 3 decimal places
            for i, cls in enumerate(model.classes_):
                results[f"{model_name}_proba_{cls}"] = proba[:, i]

        return results

    def _train_and_evaluate_classifiers(
        self, 
        classifiers: dict[str, BaseEstimator], 
        X_train: pd.DataFrame, 
        Y_train: pd.Series, 
        X_test: pd.DataFrame, 
        Y_test: pd.Series, 
        group_name: str
    ) -> tuple[dict, pd.DataFrame]:
        """
        Trains classifiers and evaluates them. Returns trained models and a combined results DataFrame.
        """
        # Initialize a dictionary to store the trained models
        trained_models = {}
        combined_results_df = None

        # Iterate through the classifiers
        for name, clf in classifiers.items():
            print(f"[INFO] - LINE 183 - classifier.py - Training {name} classifier")
            # Pass the training values to the initialized objects, which then fits the models
            model = clf.fit(X_train, Y_train)

            # Save the created model
            trained_models[name] = model

            # Evaluate the model on the test set and store the results
            # The evaluate_model function returns a DataFrame with predictions and accuracy
            result_df = self._evaluate_model(model, X_test, Y_test, name, group_name)

            # Combine the results into one DataFrame
            if combined_results_df is None:
                combined_results_df = result_df
            else:
                result_df = result_df.drop(columns=["filename", "truth_label"])
                combined_results_df = combined_results_df.join(result_df)

        # Add disagreement column to the combined results DataFrame
        pred_cols = [f"{name}_pred" for name in trained_models.keys()]
        combined_results_df[f"disagreement_{group_name}"] = (combined_results_df[pred_cols].nunique(axis=1) > 1).astype(int)

        return trained_models, combined_results_df

    def _level1(self) -> pd.DataFrame:
        """
            Trains and evaluates the Level 1 classifier on the binary labels.
            The trained models are stored in self.trained_models["level1"].
            The predictions and accuracy are stored in results
        """
        print("[INFO] - classifier.py - LINE 213 - Training Level 1 classifier in progress")
        # Get the models and results from _train_and_evaluatec
        models, results_df = self._train_and_evaluate_classifiers(
            self.classifiers_level1,
            self.X_train,
            self.Y_train_binary,
            self.X_test,
            self.Y_test_binary,
            group_name="level1"
        )

        # Save the models
        self.trained_models["level1"] = models
        # Return the results
        return results_df

    def _level2(self, group: str) -> pd.DataFrame:
        """
            Trains and evaluates the Level 2 classifier on the categorical labels
            The trained models are stored in self.trained_models["level2_cancer/non_cancer"].
            The predictions and accuracy are stored in results
        """
        print(f"[INFO] - classifier.py - LINE 235 - Training Level 2 classifier in progress for group '{group}'")

        # Makes masks and chooses the right classifiers depending on which group we run it on(cancer/non-cancer)
        if group == "cancer":
            # Create filter for cancer samples
            mask_train = self.Y_train_binary == True
            mask_test = self.Y_test_binary == True
            classifiers = self.classifiers_level2_cancer
        else:
            # Create filter for non-cancer samples
            mask_train = self.Y_train_binary == False
            mask_test = self.Y_test_binary == False
            classifiers = self.classifiers_level2_non_cancer

        # Filters out cancer/non-cancer training and testing sets
        X_train_subset = self.X_train[mask_train]
        X_test_subset = self.X_test[mask_test]
        Y_train_subset = self.Y_train_categorical[mask_train]
        Y_test_subset = self.Y_test_categorical[mask_test]

        # Get the models and results from _train_and_evaluatec
        models, results_df = self._train_and_evaluate_classifiers(
            classifiers,
            X_train_subset,
            Y_train_subset,
            X_test_subset,
            Y_test_subset,
            group_name=f"level2_{group}"
        )
        # Save the models
        self.trained_models[f"level2_{group}"] = models
        # Return the results
        return results_df

    def _run(self) -> dict[str, pd.DataFrame]:
        """
            Runs the training and evaluation process on both levels.
            The results are stored in self.results.
            This method is only called internally and is not meant to be called outside of the class.
        """
        print("[INFO] - classifier.py - LINE 275 - Hierarchical classifier is running")
        # Initialize a dictionary to store the results for each level
        combined_results = {}

        # First, we train the Level 1 classifier, so only for binary label
        combined_results["level1"] =  self._level1()

        # Then, we train the Level 2 classifier for both cancer and non-cancer labels
        combined_results["level2_cancer"] = self._level2("cancer")
        combined_results["level2_non_cancer"] = self._level2("non_cancer")

        return combined_results

    def _save_results(self) -> None:
        """
            Saves the results to a CSV file if an output path is provided.
        """
        if self.output_path:
            print("[INFO] - classifier.py - LINE 293 - Saving results to CSV files")
            # Save the DataFrame to CSV files
            for group, results in self.results.items():
                # Save each group's results to a separate CSV file
                group_output_path = os.path.join(self.output_path, f"results_{group}.csv")
                results.to_csv(group_output_path, index=False)
                print(f"[INFO] - classifier.py - LINE 299 - Model results for {group} saved to {group_output_path}")

    def save_config(self) -> None:
        """
            Saves the classifier configuration, parameters and dataset stats to a JSON file
        """
        # Helper function to extract fitted parameters from a classifier
        def extract_fitted_params(clf):
            """Extracts fitted parameters from a classifier"""
            extracted_params = {}
            # Check if the classifier has the specified common attributes and add them to the dictionary
            if hasattr(clf, "get_params"):
                extracted_params["get_params"] = clf.get_params()
            if hasattr(clf, "n_features_in_"):
                extracted_params["n_features_in_"] = clf.n_features_in_
            if hasattr(clf, "classes_"):
                extracted_params["classes_"] = clf.classes_.tolist()
            if hasattr(clf,"coef_"):
                extracted_params["coef_"] = clf.coef_.tolist()
            if hasattr(clf,"intercept_"):
                extracted_params["intercept_"] = clf.intercept_.tolist()
            if hasattr(clf,"feature_importances_"):
                extracted_params["feature_importances_"] = clf.feature_importances_.tolist()
            
            # Model specific attributes
            return extracted_params
        
        # Build the configuration dictionary
        config = {
            "timestamp": datetime.now().isoformat(),
            "dataset_stats": {
                "num_samples": len(self.df),
                "num_features": len(self.feature_names),
                "training_samples": len(self.X_train),
                "testing_samples": len(self.X_test),
            },
            "base_dir": str(self.base_dir),
            "feature_names": self.feature_names,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "classification_reports": { name: { atr: value for atr, value in report.items()} for name, report in self.reports.items()},
            "classifiers_level1": {
                name: extract_fitted_params(clf) for name, clf in self.classifiers_level1.items()
            },
            "classifiers_level_level2_cancer": {
                name: extract_fitted_params(clf) for name, clf in self.classifiers_level2_cancer.items()
            },
            "classifiers_level_level2_non_cancer": {
                name: extract_fitted_params(clf) for name, clf in self.classifiers_level2_non_cancer.items()
            }
        }
        
        # save the config to a JSON file
        config_path = os.path.join(self.output_path, "classifier_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"[INFO] - classifier.py LINE 356 - Classifier configuration saved to {config_path}")

    def tune_hyperparameters(self, level: str, model_name: str, param_grid: dict) -> None:
        """
            Hyperparameter tuning for a specific model using GridSearchCV.
        """
        if level == "level1":
            models = self.classifiers_level1
        elif level == "level2_cancer":
            models = self.classifiers_level2_cancer
        elif level == "level2_non_cancer":
            models = self.classifiers_level2_non_cancer
        # Check if the model name is valid
        if model_name not in models:
            raise ValueError(f"[ERROR] - classifier.py - LINE 370 - Model '{model_name}' not found in the specified level '{level}'")
        
        # Get the model
        model = models[model_name]
        
        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        if level == "level1":
            # Use the training data for level 1
            grid_search.fit(self.X_train, self.Y_train_binary)
        elif level == "level2_cancer":
            # Use the training data for level 2 cancer
            mask_train = self.Y_train_binary == True
            X_train_subset = self.X_train[mask_train]
            Y_train_subset = self.Y_train_categorical[mask_train]
            grid_search.fit(X_train_subset, Y_train_subset)
        elif level == "level2_non_cancer":
            # Use the training data for level 2 non-cancer
            mask_train = self.Y_train_binary == False
            X_train_subset = self.X_train[mask_train]
            Y_train_subset = self.Y_train_categorical[mask_train]
            grid_search.fit(X_train_subset, Y_train_subset)

        # Save the best parameters
        best_params = grid_search.best_params_
        # Save the best model
        best_model = grid_search.best_estimator_
        self.trained_models[level][model_name] = best_model

        # Print the best parameters
        print(f"[INFO] - classifier.py - LINE 400 - Best parameters for {model_name}: {grid_search.best_params_}")
        y_pred = best_model.predict(self.X_test)
        print(f"[INFO] - classifier.py - LINE 402 - Accuracy: {accuracy_score(self.Y_test_binary, y_pred)}")