from util import (
    joblib, json, np, os, pd, plt, sns,
    StratifiedGroupKFold, BaseEstimator,
    GridSearchCV, RandomizedSearchCV, LearningCurveDisplay,
    StandardScaler, TunedThresholdClassifierCV,
    Pipeline, PCA, DecisionBoundaryDisplay,
    accuracy_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report, calibration_curve,
    clone, precision_score, recall_score, permutation_importance,
    f1_score, roc_auc_score, 
)

class Classifier:
    def __init__(self, output_path: str = None, random_state: int = 42):
        """
            Parent class for shared classifier logic

            Parameters:
                output_path (str): Path to save the results or predictions
                random_state (int): Random state for reproducibility

            Methods:
                evaluate_classifiers(X_test: pd.DataFrame, y_test: pd.Series) -> None: Evaluates the trained classifiers on the test set
                make_predictions(X_new: pd.DataFrame) -> dict: Make predictions using the loaded models
                save_models() -> None: Save the trained models to disk
                save_probabilities(results, probabilities, type: str) -> None: Save results and probabilities to CSV/JSON
                visualize(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame) -> None: Visualize model decision boundaries
                visualize_CV_boxplots(scoring: str) -> None: Visualize cross-validation scores using boxplots
        """
        self.output_path = output_path
        self.random_state = random_state
        self.trained_models = {}

    def evaluate_classifiers(self, X_test, y_test, scoring_functions: dict = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score,
        'confusion_matrix': confusion_matrix
    }) -> dict:
        """
            Evaluate the trained classifier on the test set
            
            Param:
                X_test (pd.DataFrame): Test features
                y_test (pd.Series): Test labels

            Returns:
                model_results (dict): Dictionary containing evaluation results for each model
                probabilities (pd.DataFrame): DataFrame containing predicted probabilities and labels
        """
        # Check if 'self.df' exists before using it (it may not exist in the case of loaded models)
        if hasattr(self, 'df') and self.df is not None:
            # If self.df exists, use it to get the 'image_fname' column for display
            probabilities = pd.DataFrame({
                'image_fname': self.df.loc[X_test.index, 'image_fname'],
                'y_true': y_test
            })
        else:
            # If self.df doesn't exist, create the probabilities with only the truth labels
            probabilities = pd.DataFrame({
                'y_true': y_test
            })

        # Create a dictionary to store evaluation results
        model_results = {}

        # Iterate through each trained model and evaluate its performance
        for name, model in self.trained_models.items():
            print( f"[INFO] - classifier.py - Evaluating classifier: {name}")
            
            # Check if the model is fitted
            if not hasattr(model, 'predict'):
                print(f"[WARNING] - classifier.py - Model {name} is not fitted. Skipping evaluation.")
                continue
                
            # Predict on the test set
            y_pred = model.predict(X_test)

            # Store predicted labels in the
            probabilities[f"{name}_y_pred"] = y_pred
            
            # If the model supports predict_proba, get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = np.round(model.predict_proba(X_test)[:, 1], 3)
                # Store predicted probabilities in the
                probabilities[f"{name}_y_proba"] = y_proba
            
            # Store model evaluation results using provided or default scoring functions
            model_results[name] = {}
            for metric_name, metric_func in scoring_functions.items():
                try:
                    if metric_name == "roc_auc" and y_proba is None:
                        # Skip AUC if no probabilities are available
                        model_results[name][metric_name] = None
                    else:
                        # Compute the metric (accuracy, precision, etc.)
                        model_results[name][metric_name] = metric_func(y_test, y_pred)
                except Exception as e:
                    print(f"[WARNING] - Error computing {metric_name} for {name}: {e}")
            
            # Lastly, save the model's best threshold if available
            if hasattr(model, 'best_threshold_'):
                model_results[name]['best_threshold'] = model.best_threshold_

            print(f"    [INFO] - classifier.py - Evaluation results for {name}:\n{classification_report(y_test, y_pred)}")

        # Return the evaluation results
        return model_results, probabilities

    def save_result_and_probabilities(self, results: dict, probabilities: dict, type: str, save_visible: bool=False) -> None:
        """
            Save the results and probabilities of the classifier evaluation to CSV/JSON

            Parameters:
                results (dict): Dictionary containing evaluation results
                probabilities (dict): Dictionary containing predicted probabilities
                type (str): Type of evaluation (e.g., "tuning", "test", "validation", "final")
                save_visible (bool): Whether to save the results and probabilities to repo
        """
        
        if save_visible is None or not save_visible:
            output_path = os.path.join(self.output_path, "other")
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_path
            
        # Construct output directory path
        output_file = os.path.join(output_path, f"probabilities_{type}.csv")
        # Save probabilities to CSV

        probabilities.to_csv(output_file, index=False)
        print(f"[INFO] - classifier.py - {type} probabilities saved to {output_file}")

        # Construct output directory path for results
        output_file = os.path.join(output_path, f"results_{type}.json")
        # Save evaluation results to JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"[INFO] - classifier.py - {type} results saved to {output_file}")

    def make_predictions(self, X_new: pd.DataFrame) -> dict:
        """
            Given a new dataset (X_new), make predictions using the loaded models
        """
        # Dictionary to store predictions from each model
        predictions = {}
        
        # If self.trained_models is empty (should not be, but a fall-back), it means no models have been trained or loaded
        if not self.trained_models:
            raise ValueError("[ERROR] - classifier.py - No trained models available. Please train or load models before making predictions.")
        # Iterate through each trained model and make predictions
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X_new)
            print(f"    [INFO] - classifier.py - Predictions made with {name}")
        
        # Return the dictionary containing predictions from all models
        return predictions

    def save_models(self, version) -> None:
        """
            Save the trained models to disk
        """
        if not self.output_path:
            raise ValueError("[ERROR] - classifier.py - Output path is not set. Please provide a valid output path to save models.")
        
        
        models_dir = os.path.join(self.output_path, "models")
        # Check if directory exists, if not create it
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        

        
        # Iterate through each trained model and save it to the specified output path
        for name, model in self.trained_models.items():
            model_file = os.path.join(models_dir, f"{name}_{version}_model.pkl")
            try:
                joblib.dump(model, model_file)
                print(f"    [INFO] - classifier.py - Saved model {name} to {model_file}")
            except Exception as e:
                print(f"[WARNING] - classifier.py - Failed to save model {name}: {e}")

    def visualize(self, X: pd.DataFrame, y: pd.Series, version: str) -> None:
        """
            Visualize model decision boundaries using PCA for dimensionality reduction
        """

        images_dir = "images"
        # Check if directory exists, if not create it
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        for name, model in self.trained_models.items():
            print(f"[INFO] - classifier.py - Visualizing model: {model.__class__.__name__}")
            # Check if X has more than 2 features, if so, apply PCA to reduce to 2D
            if X.shape[1] > 2:
                # Initialize PCA to reduce to 2 components
                pca = PCA(n_components=2)
                # Preform the PCA transformation on both training and validation sets
                X_plot = pca.fit_transform(X)
                # Clone the model for plotting purposes (to avoid modifying the original model)
                model_for_plot = clone(model)  
                model_for_plot.fit(X_plot, y)
            else:
                # If X already has 2 features, no need for PCA
                X_plot = X
                model_for_plot = clone(model)
                model_for_plot.fit(X_plot, y)
            
            # Standardize the training and validation data
            scaler = StandardScaler()
            X_plot= scaler.fit_transform(X_plot)
            
            # Plot decision boundary using sklearn's from_estimator
            disp = DecisionBoundaryDisplay.from_estimator(
                model_for_plot,
                X_plot,
                response_method="predict",
                cmap="coolwarm",
                alpha=0.5
            )
            # Overlay actual data points using Seaborn
            sns.scatterplot(
                x=X_plot[:, 0],
                y=X_plot[:, 1],
                hue=y,
                palette="Set2",
                edgecolor="k",
                s=50,
                ax=disp.ax_
            )
            
            disp.ax_.set_title(f"Decision Boundary: {name} (PCA Reduced)")
            disp.ax_.set_xlabel("Component 1")
            disp.ax_.set_ylabel("Component 2")
            plt.tight_layout()
            #plt.show()
            plt.savefig(f"images/{name}_{version}_decision_boundary.png", dpi=300, bbox_inches='tight')

            # Plot ROC curve for hyperparameter tuning
            y_proba = np.round(model.predict_proba(X)[:, 1],3)
            sns.set_style("whitegrid")
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            # Plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            #plt.show()
            plt.savefig(f"images/{name}_{version}_roc_curve.png", dpi=300, bbox_inches='tight')        

            # Calibration curve
            prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10, strategy="uniform")

            plt.figure(figsize=(6, 5))
            plt.plot(prob_pred, prob_true, marker='o')
            plt.plot([0, 1], [0, 1], '--', color='gray')  # perfect calibration line
            plt.title('Calibration Curve')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Probability')
            plt.legend(loc ='best')
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            #plt.show()
            plt.savefig(f"images/{name}_{version}_calibration_curve.png", dpi=300, bbox_inches='tight')
            
    def create_importance_dashboard(self, X, y, feature_names):
        """ 
            Create feature importance plots for the trained models.
            This method calculates feature importances for tree-based models, coefficients for linear models,
            and uses permutation importance for other models. It then plots the top features across all models.
        """
        
        
        results = {}
        
        for name, model in self.trained_models.items():
            if hasattr(model.named_steps['clf'], 'feature_importances_'):
                # Tree-based
                imp = model.named_steps['clf'].feature_importances_
            elif hasattr(model.named_steps['clf'], 'coef_'):
                # Linear models
                imp = np.abs(model.named_steps['clf'].coef_[0])
            else:
                # Others use permutation importance
                r = permutation_importance(model, X, y, n_repeats=5)
                imp = r.importances_mean
                
            results[name] = pd.Series(imp, index=feature_names)
            results = pd.DataFrame(results).sort_index()
            # Plot top 10 features across models
            plt.figure(figsize=(12,8))
            results.mean(axis=1).sort_values(ascending=False)[:6].plot.barh()
            plt.title('Consensus Top Features Across Models')
            plt.tight_layout()
            plt.savefig("images/feature_importance.png", dpi=300, bbox_inches='tight')

    def visualize_CV_boxplots(self, scoring: str) -> None:
            """
            Visualize the cross-validation scores using boxplots.
            """
            images_dir = "images"
            # Check if directory exists, if not create it
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(
                x='model', 
                y='scores', 
                data=self.scores_df, 
                palette='Set2',
                hue="model",
                showfliers=False,
            )
            sns.stripplot(
                x='model', 
                y='scores', 
                data=self.scores_df, 
                color='black', 
                alpha=0.3, 
                jitter=True, 
                size=3
            )
            plt.title('Cross-Validation Scores by Model', fontsize=14)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel(scoring.replace("_", " ").title(), fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            #plt.show()
            fig.savefig("images/boxplot_cv.png", dpi=300, bbox_inches='tight')
            plt.close(fig)  # Clean up
     
class TrainClassifier(Classifier):
    def __init__(self, base_dir: str, feature_names: list, classifiers: dict[str, Pipeline], output_path: str = None):
        """
            Child class for training models from scratch
            Parent class: Classifier
                Methods:
                    - evaluate_classifiers(X_test: pd.DataFrame, y_test: pd.Series) -> None: Evaluates the trained classifiers on the test set
                    - make_predictions(X_new: pd.DataFrame) -> dict: Make predictions using the loaded models
                    - save_models() -> None: Save the trained models to disk
                    - save_probabilities(results, probabilities, type: str) -> None: Save results and probabilities to CSV/JSON
                    - visualize(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame) -> None: Visualize model decision boundaries
                    - visualize_CV_boxplots(scoring: str) -> None: Visualize the cross-validation scores using boxplots
    
            Param:
                base_dir (str): Base directory where the dataset is located
                feature_names (list): List of feature names to be used for training
                classifiers (dict[str, Pipeline]): Dictionary of classifiers to be trained
                output_path (str): Path to save the results or predictions

            Methods:
                - load_split_data(source: str = "dataset.csv") -> None: Load and split the dataset into training, validation, and test sets
                - training_hyperparameter_tuning(param_distr: dict[str, BaseEstimator], cv_splits: int = 5, scoring: str = "roc_auc") -> None: Perform training and hyperparameter tuning for the classifiers
                - train() -> None: Train the classifiers using the entire training and validation set
                - optimize_thresholds(scoring: str = 'f1') -> None: Optimize classification thresholds for the classifiers
        """
        super().__init__(output_path)
        self.base_dir = base_dir
        self.feature_names = feature_names
        self.classifiers = classifiers

    def load_split_data(self, source: str = "dataset.csv", label_col: str = 'label_binary', patient_col: str = 'patient_id') -> None:
        """
            Load and split the dataset into training, validation, and test sets
            1) Initially split the dataset into training (80%) and testing (20%) sets
            2) Further split the training set into a training subset (64%) and validation set (16%)
                Using StratifiedGroupKFold to maintain the distribution of labels and patient groups across splits

            Param:
                source (str): Path to the dataset CSV file (default is "dataset.csv")
                label_col (str): Name of the column containing the labels (default is 'label_binary')
                patient_col (str): Name of the column containing patient IDs for stratified splitting (default is 'patient_id')
        """
        print(f"[INFO] - classifier.py - Loading dataset from {source} and splitting into train/validation/testing sets")
        # Check if the path with source exists
        if not os.path.exists(os.path.join(self.base_dir, source)):
            raise FileNotFoundError(f"[ERROR] - classifier.py - Dataset file {source} not found in {self.base_dir}. Please check the path or ensure the dataset is available.")

        # Read the dataset from the specified CSV file
        self.df = pd.read_csv(os.path.join(self.base_dir, source))
        # Seperate features and labels (features on the X-axis, labels on the Y-axis)
        self.X = self.df[self.feature_names]
        self.y = self.df[label_col]
        # Extract patient groups for stratified splitting
        groups = self.df[patient_col]

        # 1) Initial split: Train (80%) + Test (20%)
        # (Note: n_splits=5 equals approximately 80% train and 20% test split)
        # This is a cross-validator that splits data into train and test sets, ensuring that the splits are stratified by the labels (y) -> should not split e.g. multiple samples from the same patient
        sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        # Now, as sgkf_test.split(self.X, self.y, groups) returns a generator object
        # We use 'next()' function to retrieve the first batch (i.e., the first train-test split) from the generator
        # This will give us the indices for the training and testing sets
        self.train_idx, self.test_idx = next(sgkf_test.split(self.X, self.y, groups))
        # Then, we use these indices to create the training and testing sets
        self.X_train = self.X.iloc[self.train_idx]
        self.X_test = self.X.iloc[self.test_idx]
        # And the corresponding labels
        self.y_train = self.y.iloc[self.train_idx]
        self.y_test = self.y.iloc[self.test_idx]

        # 2) Split train into train_sub (64%) + val (16%)
        # (Note: val_size is 20% of TRAINING data (0.2 * 0.8 = 0.16 of total))
        # Seperate the training groups (the patient ids) for stratified splitting
        train_groups = groups.iloc[self.train_idx]
        # Perform the same stratified group k-fold split for the training data
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state + 1)
        # Retrieve the first batch of train-validation split indices
        self.train_sub_idx, self.val_idx = next(sgkf_val.split(self.X_train, self.y_train, train_groups))
        # Now, we can create the training subset and validation set
        self.X_train_sub = self.X_train.iloc[self.train_sub_idx]
        self.X_val = self.X_train.iloc[self.val_idx]
        # And the corresponding labels
        self.y_train_sub = self.y_train.iloc[self.train_sub_idx]
        self.y_val = self.y_train.iloc[self.val_idx]
        print(f"[INFO] - classifier.py - Finished loading and splitting dataset")

        # Verify the splits to ensure they are valid
        self._verify_splits(groups)

    def _verify_splits(self, groups, sample_threshold: int = 20) -> None:
        """
            Validate the group separation and class balance after splitting
        """
        print("[INFO] - classifier.py - Verifying splits for group separation and class balance")
        # Get groups for train and validation
        train_groups = set(groups.iloc[self.train_idx].iloc[self.train_sub_idx])
        val_groups = set(groups.iloc[self.train_idx].iloc[self.val_idx])

        # Check no overlap between the training and validation sets
        if not train_groups.isdisjoint(val_groups):
            raise ValueError("[ERROR] - classifier.py - Overlap found between training and validation sets. Please check the splitting logic.")

        # Check minimunum number of samples
        assert len(self.y_val) >= sample_threshold, "[ERROR] - classifier.py - Training subset size is below threshold (20 samples). Please check the dataset or splitting logic."

        # Print information about the splits
        print(f"[INFO] - classifier.py - Class distributions:")
        print(f"    [INFO] - classifier.py - Train: {self.y_train_sub.value_counts(normalize=True)}")
        print(f"    [INFO] - classifier.py - Val: {self.y_val.value_counts(normalize=True)}")
        print(f"    [INFO] - classifier.py - Test: {self.y_test.value_counts(normalize=True)}")

        print(f"\n[INFO] - classifier.py - Split Verification:")
        print(f"    [INFO] - classifier.py - Train: {len(self.train_idx)}, Val: {len(self.val_idx)}, Test: {len(self.test_idx)}")
        print(f"    [INFO] - classifier.py - Total: {len(self.train_sub_idx)+len(self.val_idx)+len(self.test_idx)} vs Original: {len(self.df)}")

    def training_hyperparameter_tuning(self, param_distr: dict[str, BaseEstimator], cv_splits: int = 5, scoring: str = "roc_auc") -> None:
        """
            Perform hyperparameter tuning and training for the classifiers, saves the results to CSV files in the output path and saves plots for each.

            Param:
                param_distr (dict[str, BaseEstimator]): Dictionary containing parameter grids for each classifier
                cv_splits (int): Number of cross-validation splits (default is 5)
                scoring (str): Scoring metric to optimize during tuning (default is "roc_auc")
            
            Returns:
                model_results (dict): Dictionary containing the best parameters and scores for each model
                probabilities (dict): Dictionary containing predicted probabilities and labels for the validation set
        """
        # Check if 'self.df' exists before using it (it may not exist in the case of loaded models)
        if hasattr(self, 'df') and self.df is not None:
        # If self.df exists, use it to get the 'image_fname' column for display
            probabilities = pd.DataFrame({
                'image_fname': self.df.loc[self.X_val.index, 'image_fname'],
                'y_true': self.y_val
            })
        else:
            # If self.df does not exist, raise an error (should not happen as this is the training class' method)
            raise ValueError("[ERROR] - classifier.py - 'df' is not available. Please ensure that the dataset is loaded before proceeding.")

        all_scores = [] # for the CV results
        model_results = {} # model results

        for name, pipeline in self.classifiers.items():
            if name not in param_distr:
                print(f"[INFO] - classifier.py - No parameter distribution for {name}, skipping training and tuning.")
                continue
            
            print(f"[INFO] - classifier.py - Training and tuning hyperparameters for {name}")
            # Clone the pipeline to avoid modifying the original one
            cv = StratifiedGroupKFold(
                n_splits=cv_splits,
                shuffle=True,
                random_state=self.random_state
            )
            # Create a RandomizedSearchCV object with the cloned pipeline and parameter distribution
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distr[name],
                cv=cv.split(
                    self.X_train_sub,
                    self.y_train_sub,
                    groups=self.df.loc[self.train_idx].iloc[self.train_sub_idx]['patient_id']  # Use patient_id for stratified splitting
                ),
                scoring=scoring,
                n_iter=50,  # Number of parameter combinations to try
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores for parallel processing
                verbose=1,  # Show progress during tuning
            )

            # Fit the distribution search on the training subset
            search.fit(self.X_train_sub, self.y_train_sub)
            # Store the best estimator in the trained_models dictionary
            self.trained_models[name] = search.best_estimator_
            model = self.trained_models[name]
            # Get the cross-validation results
            cv_results = search.cv_results_
            # Extract the scores from the first split (split0) and concatenate them for all splits
            scores = cv_results["split0_test_score"]
            for i in range(1, cv_splits):  # Adjusting based on your cv_splits
                scores = np.concatenate((scores, cv_results[f'split{i}_test_score']))           
            
            all_scores.append(pd.DataFrame({
                "model": name,
                "scores": scores,
            }))
            
            # Predict on the validation set using
            y_pred = search.predict(self.X_val)
            # Store predicted labels in the dictionary
            probabilities[f"{name}_y_pred"] = y_pred
            
            # If the model supports predict_proba, get probabilities
            if hasattr(search, 'predict_proba'):
                y_proba = np.round(search.predict_proba(self.X_val)[:, 1], 3)
                # Store predicted probabilities in the DataFrame
                probabilities[f"{name}_y_proba"] = y_proba

            # Store model tuning results
            model_results[name] = {
                'best_score': search.best_score_,
                'best_params': search.best_params_
            }

            # Save the randomized search results to CSV file in the output path
            output_file = os.path.join(self.output_path / "cv_results", f"{name}_cv_results.csv")
            # Check if the directory exists, if not create it
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Save the randomized search results to CSV
            pd.DataFrame(search.cv_results_).to_csv(output_file, index=False)
            print(f"[INFO] - classifier.py - Randomized search results for {name} saved to {output_file}")

            scoring_name = scoring.replace('_', ' ').title()  # Format the scoring name for display
            images_dir = "images"
            # Check if directory exists, if not create it
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            # Learning curve
            display = LearningCurveDisplay.from_estimator(
                estimator=model,
                X=self.X_train_sub,
                y=self.y_train_sub,
                cv=cv,
                scoring=scoring,
                groups= self.df.loc[self.train_idx].iloc[self.train_sub_idx]['patient_id'],  
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                score_name=scoring_name
                )
            display.plot()
            plt.title("Learning Curve for " + name + " (Scoring: " + scoring_name + ")")
            plt.xlabel('Training Size')
            plt.ylabel(scoring_name)
            plt.legend(loc='best')
            #plt.show()
            plt.savefig(f"images/{name}_learning_curve.png", dpi=300, bbox_inches='tight')
        self.scores_df = pd.concat(all_scores, ignore_index=True)
        
        # Return the model results and probabilities
        return model_results, probabilities

    def optimize_thresholds(self, scoring: str = 'f1') -> None:
        """
            Optimize the classification thresholds for each trained model using TunedThresholdClassifierCV.
            This method assumes that the models have been trained and are available in self.trained_models.

            Param:
                scoring (str): Scoring metric to optimize the thresholds (default is 'f1')
                This can be any valid scoring metric supported by scikit-learn, such as 'roc_auc', 'accuracy', etc
        
            Returns:
                model_results (dict): Dictionary containing the best thresholds and scores for each model
                probabilities (dict): Dictionary containing predicted probabilities and labels for the validation set
        """
        # Check if 'self.df' exists before using it (it may not exist in the case of loaded models)
        if hasattr(self, 'df') and self.df is not None:
        # If self.df exists, use it to get the 'image_fname' column for display
            probabilities = pd.DataFrame({
                'image_fname': self.df.loc[self.X_val.index, 'image_fname'],
                'y_true': self.y_val
            })
        else:
            # If self.df does not exist, raise an error (should not happen as this is the training class' method)
            raise ValueError("[ERROR] - classifier.py - 'df' is not available. Please ensure that the dataset is loaded before proceeding.")

        model_results = {}
        
        for name, model in self.trained_models.items():
            print(f"[INFO] - classifier.py - Optimizing threshold for {name}")
            # Check if the model is fitted
            if not hasattr(model, 'predict'):
                print(f"[WARNING] - classifier.py - Model {name} is not fitted. Skipping evaluation.")
                continue

            # Initialize TunedThresholdClassifierCV with the prefit model
            tuner = TunedThresholdClassifierCV(
                estimator=model,
                scoring=scoring,
                cv="prefit",
                n_jobs=-1,
                refit=False # Use prefit models
            )
            # Fit the tuner on the validation set
            tuner.fit(self.X_val, self.y_val)
            self.trained_models[name] = tuner.estimator_
            
            # Predict the validation set using the tuned model
            y_val_pred = tuner.estimator_.predict(self.X_val)
            # Then store the predicted labels in the probabilities
            probabilities[f"{name}_y_pred"] = y_val_pred

            # Check if model supports predict_proba
            if not hasattr(tuner, 'predict_proba'):
                print(f"[WARNING] - classifier.py - Model {name} does not support predict_proba.")
                continue
            # Else, get the predicted probabilities
            y_val_proba = np.round(tuner.predict_proba(self.X_val)[:, 1],3)
            # And store the predicted probabilities
            probabilities[f"{name}_y_proba"] = y_val_proba
            
            # Save the models optimization results
            model_results[name] = {
                "best_threshold": tuner.best_threshold_,
                "scoring": scoring,
                "best_score": tuner.best_score_,
                "best_params": tuner.get_params()
            }
            images_dir = "images"
            # Check if directory exists, if not create it
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            # Plot Precision-Recall vs Threshold
            precision, recall, thresholds = precision_recall_curve(self.y_val, y_val_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
            plt.plot(thresholds, recall[:-1], label='Recall', color='green')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title(f'Precision-Recall vs Threshold for {name}')
            plt.legend(loc='best')
            #plt.show()
            plt.savefig(f"images/{name}_precision_recall_vs_threshold.png", dpi=300, bbox_inches='tight')

        
        # Lastly, return the model results and probabilities (eg. for saving)
        return model_results, probabilities
    
class LoadClassifier(Classifier):
    def __init__(self, model_path: str, output_path: str = None, base_dir: str = None, feature_names: list = None):
        """
            Child class for loading pre-trained models and making predictions

            Parent class: Classifier
                Methods:
                - evaluate_classifiers(X_test: pd.DataFrame, y_test: pd.Series) -> None: Evaluates the trained classifiers on the test set
                - make_predictions(X_new: pd.DataFrame) -> dict: Make predictions using the loaded models
                - save_models() -> None: Save the trained models to disk
                - save_probabilities(results, probabilities, type: str) -> None: Save results and probabilities to CSV/JSON
                - visualize(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame) -> None: Visualize model decision boundaries
                - visualize_CV_boxplots(scoring: str) -> None: Visualize the cross-validation scores using boxplots
                
            Parameters:
                model_path (str): Path to the directory containing the pre-trained models
                output_path (str): Path to save the results or predictions
            
            Methods:
                load_models(model_path: str) -> None: Load pre-trained models from the specified directory
        """
        super().__init__(output_path)
        self.base_dir = base_dir
        self.feature_names = feature_names
        self.load_models(model_path)

    def load_models(self, model_path: str) -> None:
        """
            Load pre-trained models from the specified directory
        """
        print(f"[INFO] - classifier.py - Loading models from {model_path}")
        # Load all model files from the specified directory ending with .pkl
        model_files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]
        # If no models found, raise an error
        if not model_files:
            raise FileNotFoundError(f"[ERROR] - classifier.py - No model files found in {model_path}. Please check the path or ensure models are saved.")
        
        # Else load each model and store it in the trained_models dictionary
        for model_file in model_files:
            # Extract model name from the filename (assuming format like "model_name.pkl")
            model_name = model_file.split(".")[0]
            model_full_path = os.path.join(model_path, model_file)
            # Load the model using joblib
            self.trained_models[model_name] = joblib.load(model_full_path)
            # Print confirmation message
            print(f"[INFO] - Loaded model {model_name} from {model_full_path}")
            
    def load_dataset(self, source: str = "dataset.csv", label_col: str = 'label_binary', patient_col: str = 'patient_id') -> None:
        """
            Load the dataset from the specified CSV file and prepare the features and labels for evaluation

            Param:
                source (str): Path to the dataset CSV file (default is "dataset.csv")
                label_col (str): Name of the column containing the labels (default is 'label_binary')
                patient_col (str): Name of the column containing patient IDs for stratified splitting (default is 'patient_id')
        """
        print(f"[INFO] - classifier.py - Loading dataset from {source}")
        # Check if the path with source exists
        if not os.path.exists(os.path.join(self.base_dir, source)):
            raise FileNotFoundError(f"[ERROR] - classifier.py - Dataset file {source} not found in {self.base_dir}. Please check the path or ensure the dataset is available.")
        
        # Read the dataset from the specified CSV file
        self.df = pd.read_csv(os.path.join(self.base_dir, source))
        # Seperate features and labels (features on the X-axis, labels on the Y-axis)
        self.X = self.df[self.feature_names]
        self.y = self.df[label_col]
        # Store the indices of all samples for later use
        self.all_idx = self.df.index.tolist()
        
        print(f"[INFO] - classifier.py - Finished loading dataset")