from util import (
    datetime,
    json,
    np,
    os,
    pd,
    StratifiedGroupKFold,
    BaseEstimator,
    GridSearchCV,
    RandomizedSearchCV,
    StandardScaler,
    TunedThresholdClassifierCV,
    Pipeline,
    accuracy_score,
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
        self.final_results = {}
        self.final_probabilities = None
        self.best_thresholds = {}
        
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
        
        # 1. Initial split: Train (80%) + Test (20%)
        # n_splits=5 equals approximately 80% train and 20% test split
        sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.train_idx, self.test_idx = next(sgkf_test.split(self.X, self.y, groups))
        X_train = self.X.iloc[self.train_idx]
        self.X_test = self.X.iloc[self.test_idx]
        y_train = self.y.iloc[self.train_idx]
        self.y_test = self.y.iloc[self.test_idx]

        # 2. Split train into train_sub (64%) + val (16%) 
        # val_size is 20% of TRAINING data (0.2 * 0.8 = 0.16 of total)
        train_groups = groups.iloc[self.train_idx]
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state + 1)
        self.train_sub_idx, self.val_idx = next(sgkf_val.split(X_train, y_train, train_groups))
        self.X_train_sub = X_train.iloc[self.train_sub_idx]
        self.X_val = X_train.iloc[self.val_idx]
        self.y_train_sub = y_train.iloc[self.train_sub_idx]
        self.y_val = y_train.iloc[self.val_idx]
        


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
    
    

    def hyperparameter_tuning(self, param_grids, cv_splits: int = 5, scoring: str = "roc_auc") -> None:
        
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
        print(f"[INFO] - classifier.py - Line 177 - Best parameters for {name}: {grid.best_params_}")
        print(f"[INFO] - classifier.py - Line 178 - Best {scoring} score: {grid.best_score_:.4f}")
        
    def optimize_thresholds(self, scoring: str = 'f1') -> None:
        """
        Optimize the classification thresholds for each trained model using TunedThresholdClassifierCV.
        This method assumes that the models have been trained and are available in self.trained_models.
        """
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
            self.best_thresholds[name] = tuner.best_threshold_
            print(f"[INFO] - classifier.py - Line 215 - Best threshold for {name}: {tuner.best_threshold_:.4f}")
            print(f"[INFO] - classifier.py - Line 216 - Best {scoring} score: {tuner.best_score_:.4f}")
            print(f"[INFO] - classifier.py - Line 217 - Best parameters: {tuner.get_params()}")


    def evaluate_classifiers(self) -> None:
        """
        Evaluate the trained classifier with optional thresholds on the test set and store the results.
        If the classifier is not trained, it raises an error.
        """
        
        self.final_probabilities = pd.DataFrame({
            'filename': self.df.loc[self.X_test.index, 'filename'],
            'y_true': self.y_test
        })
        
        for name, model in self.trained_models.items():
            print( f"[INFO] - classifier.py - Line 220 - Evaluating classifier: {name}")    
            
            y_pred = model.predict(self.X_test)
            y_proba = np.round(model.predict_proba(self.X_test)[:, 1], 3)
            
            self.final_probabilities[f"{name}_y_pred"] = y_pred
            self.final_probabilities[f"{name}_y_proba"] = y_proba
            
            self.final_results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_proba),
                'threshold': self.best_thresholds.get(name, 0.5),
                "roc_auc": roc_auc_score(self.y_test, y_proba)
            }
           
           
            print("[INFO] - classifier.py - Line 245 - Evaluation results for", name)
            print(classification_report(self.y_test, y_pred))
            
            
        assert all(self.final_probabilities['y_true'] == self.y_test.loc[self.X_test.index].values)
        # Save the final results to a CSV file
        self._save_final_results_probabilities()
        
        
        
    
    def _save_final_results_probabilities(self) -> None:
        """
        This method saves the final probabilities to a CSV file and the evaluation results to a JSON file.
        """
        output_file = os.path.join(self.output_path, "final_probabilities.csv")
        self.final_probabilities.to_csv(output_file, index=False)
        print(f"[INFO] - classifier.py - Line 267 - Final probabilities saved to {output_file}")
        
        # Save the final results of the classifier evaluations to a JSON file
        output_file = os.path.join(self.output_path, "final_results.json")
        with open(output_file, 'w') as f:
            json.dump(self.final_results, f, indent=4, default=str)
        print(f"[INFO] - classifier.py - Line 274 - Final results saved to {output_file}")
            
