from util.img_util import Dataset

from util import (
    os,
    pd,
    BaseEstimator,
    train_test_split,
    accuracy_score
)

def classifier_model(dataset: 'Dataset', feature_names: list, classifiers: dict[str, BaseEstimator], test_size: float = 0.3, random_state: int = 42, output_path: str = None) -> dict[str, any]:
    """
        Trains the given classifiers on extracted features, collects per-sample predictions & probabilities, and optionally saves a detailed results CSV

        Args:
            dataset (Dataset): Dataset instance containing features and labels
            feature_names (list): List of feature names used for feature extraction
            classifiers (dict): Dictionary mapping classifier names to initialized classifier objects
            test_size (float): Fraction of data to hold out for testing
            random_state (int): Random seed for reproducibility
            output_path (str): If provided, save the results CSV here

        Returns:
            dict: {
                - "models": dict of trained estimators
                - "results": DataFrame with filename, true_label, <model>_pred, <model>_proba_<class>, <model>_accuracy for each model, disagreement
            }
    """
    # Start with converting the Record instances to a DataFrame
    df = dataset.records_to__dataframe()
    
    # Then, need to filter out the features we want to use for training (should be the same as the extracted ones)
    X = df[feature_names]
    Y = df['label']

    # Now, split the dataset into training and testing sets using the provided parameters
    # test_size: fraction of data to hold out for testing
    # random_state: seed for reproducibility
    # stratify: ensures that the same proportion of labels is maintained in sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state, stratify = Y)

    # Construct a DataFrame to hold the results, starting with aligning the indices
    results = pd.DataFrame(index=X_test.index)
    # Add the filename for easy identification
    results['filename'] = df.loc[X_test.index, 'filename']
    # Add the true labels and the predicted labels from both models
    results['true_label'] = Y_test.values

    # Use a dictionary for easy access saving
    trained: dict[str, BaseEstimator] = {}

    # Iterate through the classifiers
    for name, clf in classifiers.items():
        # Pass the training values to the initialized objects, which then fits the models
        trained[name] = clf.fit(X_train, Y_train)

        # Calculate and add the predictions to the results
        pred = trained[name].predict(X_test)
        results[f"{name}_pred"] = pred

        # Calculate and add label probabilities to the results
        # But first, check if the classifier has the predict_proba method
        if hasattr(clf, "predict_proba"):
            proba = trained[name].predict_proba(X_test)
            for i, cls in enumerate(trained[name].classes_):
                    results[f"{name}_proba_{cls}"] = proba[:, i]

        # Calculate overall accuracy and add it to the results
        results[f"{name}_accuracy"] = accuracy_score(Y_test, pred)

    # Select the columns that contain the predictions for disagreement calculation
    pred_cols = [f"{name}_pred" for name in trained.keys()]

    # Then, look across each row and count the unique predictions
    # If there is more than one unique prediction, it means disagreement between classifiers: 1 means disagreement, 0 means agreement
    results["disagreement"] = (results[pred_cols].nunique(axis=1) > 1).astype(int)

    # Save the results if an output path is provided
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Then save the DataFrame to a CSV file
        results.to_csv(output_path, index=False)
        print(f"Classifier model results saved to {output_path}")

    # Return the results
    return {"models": trained, "results": results}