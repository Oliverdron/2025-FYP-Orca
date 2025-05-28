from util import (
    datetime,
    joblib,
    json,
    np,
    os,
    pd,
    np,
    plt,
    StratifiedGroupKFold,
    BaseEstimator,
    GridSearchCV,
    RandomizedSearchCV,
    StandardScaler,
    TunedThresholdClassifierCV,
    Pipeline,
    PCA,
    DecisionBoundaryDisplay,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    cross_validate,
    clone,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

class Classifier:
    def __init__(
        self,
        base_dir: str,
        feature_names: list,
        classifiers: dict[str, Pipeline],
        test_size: float = 0.3,
        random_state: int = 42,
        output_path: str = None
    ) -> None:
        self.base_dir = base_dir
        self.feature_names = feature_names
        self.classifiers = classifiers
        self.test_size = test_size
        self.random_state = random_state
        self.output_path = output_path 
        self.trained_models = {}
        
    def load_split_data(self, filename: str = "dataset.csv") -> None:
        """
        Load the dataset from a CSV file and split it into training, validation, and test sets.
        """
        
        # Couldn't seem to find a way to have stratified shuffled group single split, so we do it with 
        # StratifiedGroupKFold with n_splits=5, which gives us approximately 80% train and 20% test split
        
        print(f"[INFO] - classifier.py - Line 45 - Loading dataset from {filename} and splitting into train, val, and test sets.")
        # Load the dataset, which stores pre-calculated features and labels
        self.df = pd.read_csv(os.path.join(self.base_dir, filename))
        self.X = self.df[self.feature_names]
        self.y = self.df['label_binary']
        groups = self.df['patient_id']
        print(self.X)
        # 1. Initial split: Train (80%) + Test (20%)
        # n_splits=5 equals approximately 80% train and 20% test split
        sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.train_idx, self.test_idx = next(sgkf_test.split(self.X, self.y, groups))
        self.X_train = self.X.iloc[self.train_idx]
        self.X_test = self.X.iloc[self.test_idx]
        self.y_train = self.y.iloc[self.train_idx]
        self.y_test = self.y.iloc[self.test_idx]

        # 2. Split train into train_sub (64%) + val (16%) 
        # val_size is 20% of TRAINING data (0.2 * 0.8 = 0.16 of total)
        train_groups = groups.iloc[self.train_idx]
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state + 1)
        self.train_sub_idx, self.val_idx = next(sgkf_val.split(self.X_train, self.y_train, train_groups))
        self.X_train_sub = self.X_train.iloc[self.train_sub_idx]
        self.X_val = self.X_train.iloc[self.val_idx]
        self.y_train_sub = self.y_train.iloc[self.train_sub_idx]
        self.y_val = self.y_train.iloc[self.val_idx]
        


        # Check class balance
        self._verify_splits(groups)
        
        
    def _verify_splits(self, groups):
        """Validate group separation and class balance."""
        print("[INFO] - classifier.py - Line 110 - Verifying splits for group separation and class balance.")
        
        # Check no patient overlap
        train_groups = set(groups.iloc[self.train_idx].iloc[self.train_sub_idx])
        val_groups = set(groups.iloc[self.train_idx].iloc[self.val_idx])
        assert train_groups.isdisjoint(val_groups), "Group leakage detected!"
        
        # Check minimum samples
        assert min(self.y_val.value_counts()) >= 20, "Increase val_size!"
        
        print("Class distributions:")
        print(f"Train: {self.y_train_sub.value_counts(normalize=True)}")
        print(f"Val: {self.y_val.value_counts(normalize=True)}")
        print(f"Test: {self.y_test.value_counts(normalize=True)}")
        
        
        print(f"\nSplit Verification:")
        print(f"Train: {len(self.train_idx)}, Val: {len(self.val_idx)}, Test: {len(self.test_idx)}")
        print(f"Total: {len(self.train_sub_idx)+len(self.val_idx)+len(self.test_idx)} vs Original: {len(self.df)}")     
    
    
    def _get_train_sub_groups(self):
        """Helper to get patient IDs for train-sub."""
        return self.df.loc[self.train_idx].iloc[self.train_sub_idx]['patient_id']
    
    

    def hyperparameter_tuning(self, param_grids: dict[str,BaseEstimator], cv_splits: int = 5, scoring: str = "roc_auc") -> None:
        
        tuning_results = {}
        tuning_probabilities = pd.DataFrame({
            'filename': self.df.loc[self.X_val.index, 'filename'],
            'y_true': self.y_val
        })
        
        
        for name, pipeline in self.classifiers.items():
            if name not in param_grids:
                print(f"[WARNING] No parameter grid for {name}. Skipping.")
                continue
        
            print(f"[INFO] - classifier.py - Line 165 - Starting hyperparameter tuning for {name}")
            cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            grid = GridSearchCV(
                pipeline,
                param_grid=param_grids[name],
                cv=cv.split(self.X_train_sub, self.y_train_sub, groups=self._get_train_sub_groups()),
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            grid.fit(self.X_train_sub, self.y_train_sub)
            self.trained_models[f"{name}"] = grid.best_estimator_
            
            tuning_results[name] = {
                'best_score': grid.best_score_,
                'scoring': scoring,
                'best_params': grid.best_params_
            }
            self._save_grid_search_results(name, grid)
            
            y_val_pred = grid.predict(self.X_val)
            y_val_proba = np.round(grid.predict_proba(self.X_val)[:, 1],3)
            tuning_probabilities[f"{name}_y_pred"] = y_val_pred
            tuning_probabilities[f"{name}_y_proba"] = y_val_proba
            
            # Plot ROC curve for hyperparameter tuning
            fpr, tpr, _ = roc_curve(self.y_val, y_val_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()
            self._visualize(grid.best_estimator_, self.X_train_sub, self.y_train_sub, self.X_val, self.y_val)
            
        self._save_results_probabilities(tuning_results, tuning_probabilities, "tuning")
        
    def _save_grid_search_results(self, name: str, grid: GridSearchCV) -> None:
        """
        Save the grid search results to a csv file in the output directory.
        """
        if not self.output_path:
            raise ValueError("Output path is not set.")
        
        output_file = os.path.join(self.output_path / "cv_results", f"{name}_cv_results.csv")
        results_df = pd.DataFrame(grid.cv_results_)
        results_df.to_csv(output_file, index=False)
        print(f"[INFO] - classifier.py - Line 180 - Grid search results for {name} saved to {output_file}")

    def _visualize(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        """ Visualize multiple things:
        -  Decision boundary of the model using PCA
        """ 
        print(f"[INFO] - classifier.py - Line 190 - Visualizing model: {model.__class__.__name__}")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            X_val = pca.transform(X_val)
            
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_val = scaler.transform(X_val)
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        
        # mesh grid
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max()),
            np.linspace(X[:, 1].min(), X[:, 1].max())
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        y_pred = model.predict(grid_points)
        y_pred = np.reshape(y_pred, xx.shape)
        
        display_train = DecisionBoundaryDisplay(
            xx0 = xx,
            xx1 = yy,
            response = y_pred
        )
        display_train.plot()
        plt.show()
            
        
        
        


    def optimize_thresholds(self, scoring: str = 'f1') -> None:
        """
        Optimize the classification thresholds for each trained model using TunedThresholdClassifierCV.
        This method assumes that the models have been trained and are available in self.trained_models.
        """
        
        thresholded_results = {}
        thresholded_probabilities = pd.DataFrame({
            'filename': self.df.loc[self.X_val.index, 'filename'],
            'y_true': self.y_val
        })
        
        for name, model in self.trained_models.items():
            print(f"[INFO] Optimizing threshold for {name}.")
            tuner = TunedThresholdClassifierCV(
                estimator=model, 
                scoring=scoring,
                cv="prefit",
                n_jobs=-1,
                refit=False # Use prefit models
                )
            tuner.fit(self.X_val, self.y_val)
            self.trained_models[name] = tuner
            
            y_val_pred = tuner.predict(self.X_val)
            y_val_proba = np.round(tuner.predict_proba(self.X_val)[:, 1],3)
            thresholded_probabilities[f"{name}_y_pred"] = y_val_pred
            thresholded_probabilities[f"{name}_y_proba"] = y_val_proba
            
            thresholded_results[name] = {
                "best_threshold": tuner.best_threshold_,
                "scoring": scoring,
                "best_score": tuner.best_score_,
                "best_params": tuner.get_params()
            }
            # Plot Precision-Recall vs Threshold
            precision, recall, thresholds = precision_recall_curve(self.y_val, y_val_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
            plt.plot(thresholds, recall[:-1], label='Recall', color='green')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title(f'Precision-Recall vs Threshold for {name}')
            plt.legend(loc='best')
            plt.show()
        
            
            
            
        self._save_results_probabilities(thresholded_results, thresholded_probabilities, "threshold")


    def train_on_full(self) -> None:
        """
        Train the classifiers on the full training set (train_sub + val) and save them to trained_models.
        This method assumes that the model is already fitted with hyperparameters and optimized thresholds.
        """
        for name, model in self.trained_models.items():
            model = clone(model)  # Clone the model to avoid modifying the original
            print(f"[INFO] - classifier.py - Line 205 - Training {name} on full training set.")
            model.fit(self.X_train, self.y_train)
            self.trained_models[f"{name}_fulltrained"] = model
        

    

    def evaluate_classifiers(self) -> None:
        """
        Evaluate the trained classifier with optional thresholds on the test set and store the results.
        If the classifier is not trained, it raises an error.
        """
        
        final_probabilities = pd.DataFrame({
            'filename': self.df.loc[self.X_test.index, 'filename'],
            'y_true': self.y_test
        })
        final_results = {}
        
        for name, model in self.trained_models.items():
            print( f"[INFO] - classifier.py - Line 220 - Evaluating classifier: {name}")    
            
            y_pred = model.predict(self.X_test)
            y_proba = np.round(model.predict_proba(self.X_test)[:, 1], 3)
            
            final_probabilities[f"{name}_y_pred"] = y_pred
            final_probabilities[f"{name}_y_proba"] = y_proba
            
            final_results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_proba),
                'threshold': model.best_threshold_ if hasattr(model, 'best_threshold_') else None,
                "roc_auc": roc_auc_score(self.y_test, y_proba)
            }
        
        
            print("[INFO] - classifier.py - Line 245 - Evaluation results for", name)
            print(classification_report(self.y_test, y_pred))
            
            
        assert all(final_probabilities['y_true'] == self.y_test.loc[self.X_test.index].values)
        # Save the final results to a CSV file
        self._save_results_probabilities(final_results, final_probabilities, "final")
    
    def _save_results_probabilities(self, results, probabilities, type: str) -> None:
        
        # Save the probabilities of the classifier evalution to a CSV file
        output_file = os.path.join(self.output_path, f"{type}_probabilities.csv")
        probabilities.to_csv(output_file, index=False)
        print(f"[INFO] - classifier.py - Line 278 - {type} probabilities saved to {output_file}")
        
        # Save the results of the classifier evaluations to a JSON file
        output_file = os.path.join(self.output_path, f"{type}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"[INFO] - classifier.py - Line 284 - {type} results saved to {output_file}")
        
        
        
    def save_model(self, model: BaseEstimator, name: str) -> None:
        """
        Save a trained model to a file.
        Args:
            model (BaseEstimator): The trained model to save.
            name (str): The name of the model for saving.
        """
        if not self.output_path:
            raise ValueError("Output path is not set.")
        
        output_file = os.path.join(self.output_path, f"{name}_model.pkl")
        with open(output_file, 'wb') as f:
            joblib.dump(model, f)
        print(f"[INFO] - classifier.py - Line 262 - Model {name} saved to {output_file}")
        
        
    def load_trained_model(self, name: str) -> BaseEstimator:
        model_path = os.path.join(self.output_path, f"{name}_best_model.pkl")
        self.trained_models[name] = joblib.load(model_path)
        print(f"[INFO] - classifier.py - Line 280 - Loaded model {name} from {model_path}")